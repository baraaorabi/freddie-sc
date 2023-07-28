#!/usr/bin/env python3
import functools
from multiprocessing import Pool
import os
import glob
import gzip

from itertools import groupby
import argparse
import re
from math import exp, ceil
import bisect

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from tqdm import tqdm


class SplitRegex:
    cinterval_re = "([0-9]+)-([0-9]+)"
    cigar_op_re = "([0-9]+)([MIDNSHPX=])"
    cigar_re = "((%(c)s)+)" % {"c": cigar_op_re}
    rinterval_re = "%(i)s:%(i)s:%(c)s" % {"i": "([0-9]+)-([0-9]+)", "c": cigar_re}
    cinterval_prog = re.compile(cinterval_re)
    rinterval_prog = re.compile(rinterval_re)
    cigar_prog = re.compile(cigar_op_re)

    read_prog = re.compile(
        r"%(rid_re)s\t%(name_re)s\t%(chr_re)s\t%(strand_re)s\t%(cid_re)s\t%(intervals_re)s\n$"
        % {
            "rid_re": "(?P<rid>[0-9]+)",
            "name_re": "(?P<name>[!-?A-~]{1,254})",
            "chr_re": "(?P<chr>[0-9A-Za-z!#$%&+./:;?@^_|~-][0-9A-Za-z!#$%&*+./:;=?@^_|~-]*)",
            "strand_re": "(?P<strand>[+-])",
            "cid_re": "(?P<cid>[0-9]+)",
            "intervals_re": r"(?P<intervals>%(i)s(\t%(i)s)*)" % {"i": rinterval_re},
        }
    )

    tint_prog = re.compile(
        r"#%(chr_re)s\t%(cid_re)s\t%(intervals_re)s\t%(read_count_re)s\n$"
        % {
            "chr_re": "(?P<chr>[0-9A-Za-z!#$%&+./:;?@^_|~-][0-9A-Za-z!#$%&*+./:;=?@^_|~-]*)",
            "cid_re": "(?P<cid>[0-9]+)",
            "intervals_re": "(?P<intervals>%(i)s(,%(i)s)*)" % {"i": cinterval_re},
            "read_count_re": "(?P<read_count>[0-9]+)",
        }
    )


rev_comp = dict(A="T", C="G", G="C", T="A")
dna_id = dict(A="A", C="C", G="G", T="T")


class SegmentationParams:
    def __init__(
        self,
        sigma=5.0,
        threshold_rate=0.9,
        variance_factor=3.0,
        max_problem_size=50,
        min_read_support_outside=3,
        ignore_ends=False,
    ):
        self.sigma: float = sigma
        self.threshold_rate: float = threshold_rate
        self.variance_factor: float = variance_factor
        self.max_problem_size: int = max_problem_size
        self.min_read_support_outside: int = min_read_support_outside
        self.ignore_ends: bool = ignore_ends

        smoothed_threshold: list[float] = list()
        while True:
            x = len(smoothed_threshold)
            y = threshold_rate / (1 + ((threshold_rate - 0.5) / 0.5) * exp(-0.05 * x))
            if x > 5 and x * (threshold_rate - y) < 0.5:
                break
            smoothed_threshold.append(round(y, 2))
            assert len(smoothed_threshold) < 1000

        self.smoothed_threshold: list[float] = smoothed_threshold


segmentation_params: SegmentationParams = SegmentationParams()


class Tint:
    def __init__(self, split_header):
        re_dict = SplitRegex.tint_prog.match(split_header)
        assert re_dict != None
        re_dict = re_dict.groupdict()

        self.id = int(re_dict["cid"])
        self.chr: str = re_dict["chr"]
        self.intervals: list[tuple[int, int]] = [
            (int(x[0]), int(x[1]))
            for x in SplitRegex.cinterval_prog.findall(re_dict["intervals"])
        ]
        self.read_count: int = int(re_dict["read_count"])
        self.reads: list[Read] = list()
        # read_reps is a list of items (key,val)
        # key is a tuple of intervals tuples (start, end) of the read
        # val is a list of read indices that have the same intervals
        self.read_reps: list[
            tuple[
                tuple[ # Read intervals
                    tuple[int, int], # (start, end)
                    ...
                ],  
                list[int]] # list of read indices
        ] = list()
        # final_positions is a list of positions
        self.final_positions: list[int] = list()
        # segs is a list of tuples (start, end) of segments
        self.segs: list[
            tuple[int, int] # (start, end)
        ] = list()
        assert all(
            a[1] < b[0] for a, b in zip(self.intervals[:-1], self.intervals[1:])
        ), self.intervals
        assert all(s < e for s, e in self.intervals)

    def compute_read_reps(self):
        read_reps = dict()
        for ridx, read in enumerate(self.reads):
            k = tuple(((ts, te) for ts, te, _, _, _ in read.intervals))
            if not k in self.read_reps:
                read_reps[k] = list()
            read_reps[k].append(ridx)
        self.read_reps = list(read_reps.items())

    def get_as_record(self, sep: str = "\t"):
        record = list()
        record.append("#{}".format(self.chr))
        record.append(str(self.id))
        record.append(",".join(map(str, self.final_positions)))
        return sep.join(record)


class Read:
    def __init__(self, split_line: str):
        re_dict = SplitRegex.read_prog.match(split_line)
        assert re_dict != None
        re_dict = re_dict.groupdict()
        self.id = int(re_dict["rid"])
        self.name = re_dict["name"]
        self.chr = re_dict["chr"]
        self.strand = re_dict["strand"]
        self.tint = int(re_dict["cid"])
        self.intervals = [
            (
                int(x[0]),
                int(x[1]),
                int(x[2]),
                int(x[3]),
                [(int(c[0]), c[1]) for c in SplitRegex.cigar_prog.findall(x[4])],
            )
            for x in SplitRegex.rinterval_prog.findall(re_dict["intervals"])
        ]
        self.seq: str = ""
        self.length: int = 0
        self.gaps: list = list()
        self.data: list = list()
        assert all(
            aet <= bst and aer <= bsr
            for (_, aet, _, aer, _), (bst, _, bsr, _, _) in zip(
                self.intervals[:-1], self.intervals[1:]
            )
        )
        assert all(st < et and sr < er for (st, et, sr, er, _) in self.intervals)

    def get_as_record(self, sep: str = "\t"):
        record = list()
        record.append(str(self.id))
        record.append(self.name)
        record.append(self.chr)
        record.append(self.strand)
        record.append(str(self.tint))
        record.append("".join(map(str, self.data)))
        record.append("".join("{},".format(g) for g in self.gaps))
        return sep.join(record)


def parse_args():
    parser = argparse.ArgumentParser(description="Cluster aligned reads into isoforms")
    parser.add_argument(
        "-s",
        "--split-dir",
        type=str,
        required=True,
        help="Path to Freddie split directory of the reads",
    )
    parser.add_argument(
        "--consider-ends",
        type=str_to_bool,
        nargs="?",
        const=True,
        default=False,
        help="Consider the start and end splice sites in segmentation",
    )
    parser.add_argument(
        "-o",
        "--outdir",
        type=str,
        default="freddie_segment/",
        help="Path to output directory. Default: freddie_segment/",
    )
    parser.add_argument(
        "-t",
        "--threads",
        type=int,
        default=1,
        help="Number of threads for multiprocessing. Default: 1",
    )
    parser.add_argument(
        "-sd",
        "--sigma",
        type=float,
        default=5.0,
        help="Sigma value for gaussian_filter1d",
    )
    parser.add_argument(
        "-tp",
        "--threshold-rate",
        type=float,
        default=0.90,
        help="Threshold rate above which the read will be considered as covering a segment. Low threshold is 1-threshold_rate. Anything in between is considered ambigious. Default: 0.9. Note: the stricter threshold for a given segment length will be used.",
    )
    parser.add_argument(
        "-vf",
        "--variance-factor",
        type=float,
        default=3.0,
        help="The stdev factor to fix a candidate peak. The threshold is set as > mean(non-zero support for splicing postions)+variance_factor*stdev(non-zero support for splicing postions). Default 3.0",
    )
    parser.add_argument(
        "-mps",
        "--max-problem-size",
        type=int,
        default=50,
        help="Maximum number of candidate breakpoints allowed per segmentation problem",
    )
    parser.add_argument(
        "-lo",
        "--min-read-support-outside",
        type=int,
        default=3,
        help="Minimum reads support for splice site to support a breakpoint",
    )
    args = parser.parse_args()

    assert 1 >= args.threshold_rate >= 0.5
    assert 10 > args.variance_factor > 0
    assert 50 >= args.sigma > 0
    assert args.max_problem_size > 3
    assert args.min_read_support_outside >= 0
    assert args.threads > 0
    global segmentation_params
    segmentation_params = SegmentationParams(
        sigma=args.sigma,
        threshold_rate=args.threshold_rate,
        variance_factor=args.variance_factor,
        max_problem_size=args.max_problem_size,
        min_read_support_outside=args.min_read_support_outside,
        ignore_ends=not args.consider_ends,
    )
    return args


def str_to_bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in {"false", "f", "0", "no", "n"}:
        return False
    elif value.lower() in {"true", "t", "1", "yes", "y"}:
        return True
    raise ValueError(f"{value} is not a valid boolean value")


def read_split(split_tsv):
    if split_tsv.endswith(".gz"):
        split_tsv = gzip.open(split_tsv, "rt")
    else:
        split_tsv = open(split_tsv)
    header = str(split_tsv.readline())
    tint = Tint(header)

    for line in split_tsv:
        read = Read(str(line))
        tint.reads.append(read)
    assert len(tint.reads) == tint.read_count
    tint.compute_read_reps()

    return tint


def read_sequence(tint: Tint, reads_tsv: str):
    rid_to_seq = dict()
    for line in open(reads_tsv):
        line = line.rstrip().split("\t")
        rid = int(line[0])
        seq = line[3]
        rid_to_seq[rid] = seq
    assert len(rid_to_seq) == len(tint.reads), tint
    for read in tint.reads:
        read.seq = rid_to_seq[read.id]
        read.length = len(read.seq)
    return


def get_cumulative_coverage(
    read_reps, candidate_y_idxs, y_idx_to_pos, pos_to_Yy_idx, Y_idx
):
    C = np.zeros(
        (
            len(candidate_y_idxs)
            + 1,  # Cumulative coverage upto but excluding each candidate y_idx
            len(read_reps),  # for each read_rep
        ),
        dtype=np.uint32,
    )

    # First, build a dictionary {pos : (Y_C index, coverage on that entry) }
    total_coverage = 0
    pos_to_C_idx = dict()
    for read_rep, _ in read_reps:
        for ts, te in read_rep:
            Y_idx_s, y_idx_s = pos_to_Yy_idx[ts]
            Y_idx_e, y_idx_e = pos_to_Yy_idx[te]
            assert Y_idx_s == Y_idx_e
            # Checking if this interval of the read covers this tint interval
            if Y_idx_s != Y_idx:
                continue
            total_coverage += te - ts + 1
            if not ts in pos_to_C_idx:
                pos_to_C_idx[ts] = bisect.bisect_right(
                    candidate_y_idxs,
                    y_idx_s,
                )
            if not te in pos_to_C_idx:
                pos_to_C_idx[te] = bisect.bisect_right(
                    candidate_y_idxs,
                    y_idx_e,
                    lo=pos_to_C_idx[ts],
                )
    for rr_idx, (read_rep, _) in enumerate(read_reps):
        for ts, te in read_rep:
            if pos_to_Yy_idx[ts][0] != Y_idx:
                continue
            C_idx_s = pos_to_C_idx[ts]
            C_idx_e = pos_to_C_idx[te]
            assert 0 < C_idx_s <= C_idx_e <= len(candidate_y_idxs), (
                C_idx_s,
                C_idx_e,
                ts,
                te,
            )
            if C_idx_s == C_idx_e:
                C[C_idx_s][rr_idx] += te - ts + 1
                continue

            cov_s = y_idx_to_pos[candidate_y_idxs[C_idx_s]] - ts
            assert cov_s > 0, cov_s
            cov_e = te - y_idx_to_pos[candidate_y_idxs[C_idx_e - 1]] + 1
            assert cov_e > 0, cov_e
            # Adding coverge of the first candidate segment
            C[C_idx_s][rr_idx] += cov_s
            # Adding coverge of the last candidate segment
            C[C_idx_e][rr_idx] += cov_e
            # Adding full coverage for all segments in between
            for C_idx in range(C_idx_s + 1, C_idx_e):
                seg_len = candidate_y_idxs[C_idx] - candidate_y_idxs[C_idx - 1]
                assert seg_len > 0, seg_len
                C[C_idx][rr_idx] += seg_len
    # Doubling checking we added the slack correctly
    assert C.sum() == total_coverage, (C.sum(), total_coverage)
    # Summing cumulatively over the candidate segments
    for C_idx in range(1, C.shape[0]):
        C[C_idx] += C[C_idx - 1]
    return C


def refine_segmentation(y_raw, y_idxs, sigma, skip=20, min_internal_splice=20):
    refine_y_idxs = list()
    for s_yidx, e_yidx in zip(y_idxs[:-1], y_idxs[1:]):
        if e_yidx - s_yidx <= 2 * skip:
            continue
        i_vals = [x for x in y_raw[s_yidx:e_yidx]]
        for i in range(0, skip):
            i_vals[i] = 0.0
            i_vals[-i - 1] = 0.0
        if sum(i_vals) < min_internal_splice:
            continue
        i_gauss = gaussian_filter1d(
            i_vals, sigma, mode="constant", cval=0.0, truncate=1.0
        )
        for i in find_peaks(i_gauss, distance=skip)[0]:
            if (
                sum(i_gauss[int(round(i - sigma)) : int(round(i + sigma + 1))])
                < min_internal_splice
            ):
                continue
            refine_y_idxs.append(i + s_yidx)
    return refine_y_idxs


def get_high_threshold(
    seg_len: int, smoothed_threshold: list[float], threshold_rate: float
):
    if seg_len < len(smoothed_threshold):
        h = smoothed_threshold[seg_len]
    else:
        h = threshold_rate
    return h


def forward_thread_cigar(cigar, t_goal, t_pos, q_pos):
    assert t_pos <= t_goal
    cig_idx = 0
    while t_pos < t_goal:
        c, t = cigar[cig_idx]
        c = min(c, t_goal - t_pos)
        if t in ["M", "X", "="]:
            t_pos += c
            q_pos += c
        if t in ["D"]:
            t_pos += c
        if t in ["I"]:
            q_pos += c
        cig_idx += 1
    assert t_pos == t_goal
    return q_pos


def get_interval_start(start, read):
    """
    Finds the first position in the read alignment intervals that aligns to start or after.
    Any (negative) offset is reported as slack.
    """
    for t_start, t_end, q_start, q_end, cigar in read.intervals:
        # We haven't passed the start location yet
        if t_end < start:
            continue
        if start < t_start:
            q_pos = q_start
            slack = start - t_start
        else:
            q_pos = forward_thread_cigar(
                cigar=cigar, t_goal=start, t_pos=t_start, q_pos=q_start
            )
            slack = 0
        assert slack <= 0, (slack, t_start, start)
        assert q_start <= q_pos <= q_end, (q_start, q_pos, q_end)
        return q_pos, slack
    assert False


def get_interval_end(end, read):
    """
    Finds the last location on the read that aligns on or before end.
    Any (negative) offset is reported as slack.
    """
    for t_start, t_end, q_start, q_end, cigar in reversed(read.intervals):
        # We haven't passed end yet
        if t_start > end:
            continue
        # the right most interval that covers end, ends before end
        if t_end < end:
            q_pos = q_end
            slack = t_end - end
        else:
            q_pos = forward_thread_cigar(
                cigar=cigar, t_goal=end, t_pos=t_start, q_pos=q_start
            )
            slack = 0
        assert slack <= 0, (slack, t_end, end)
        assert 0 <= q_pos <= q_end, (q_start, q_pos, q_end)
        return q_pos, slack
    assert False


def find_longest_poly(seq, s, e, step, match_score=1, mismatch_score=-2, char="A"):
    if e - s == 0:
        return
    if seq[s] == char:
        scores = [match_score]
    else:
        scores = [0]
    for m in (
        match_score if c == char else mismatch_score for c in seq[s + step : e : step]
    ):
        scores.append(max(0, scores[-1] + m))
    for k, g in groupby(enumerate(scores), lambda x: x[1] > 0):
        if not k:
            continue
        i, S = list(zip(*g))
        max_s, max_i = max(zip(S, i))
        l = max_i + 1 - i[0]
        yield i[0], l, seq[s:e:step][i[0] : i[0] + l].count(char) / l


def get_unaligned_gaps_and_polyA(read: Read, segs):
    gaps = set()
    if not 1 in read.data:
        return
    intervals = list()
    for d, group in groupby(enumerate(read.data), lambda x: x[1]):
        if d != 1:
            continue
        group = list(group)
        f_seg_idx = group[0][0]
        l_seg_idx = group[-1][0]
        intervals.append((f_seg_idx, l_seg_idx))
    assert len(intervals) > 0, read.data
    (f_seg_idx, _) = intervals[0]
    start = segs[f_seg_idx][0]
    q_ssc_pos, _ = get_interval_start(start=start, read=read)
    (_, l_seg_idx) = intervals[-1]
    end = segs[l_seg_idx][1]
    q_esc_pos, _ = get_interval_end(end=end, read=read)
    assert 0 <= q_ssc_pos <= q_esc_pos <= read.length, (
        q_ssc_pos,
        q_esc_pos,
        read.length,
        start,
        end,
        segs,
        read,
    )
    s_polys = list()
    for char in ["A", "T"]:
        s = 0
        e = q_ssc_pos
        step = 1
        sc_char = char
        if read.strand == "-":
            s = -s - 1
            e = -e - 1
            step = -1
            sc_char = rev_comp[char]
        for i, l, p in find_longest_poly(read.seq, s=s, e=e, step=step, char=sc_char):
            if l < 20 or p < 0.85:
                continue
            assert 0 <= i < q_ssc_pos, (i, q_ssc_pos, read.length)
            s_polys.append((i, l, p, char))
    if len(s_polys) > 0:
        i, l, p, char = max(s_polys, key=lambda x: x[2])
        poly_to_gene_gap_size = q_ssc_pos - i - l
        assert 0 <= poly_to_gene_gap_size < q_ssc_pos
        gaps.add("S{}_{}:{}".format(char, l, poly_to_gene_gap_size))
        gaps.add("SSC:{}".format(i))
    else:
        gaps.add("SSC:{}".format(q_ssc_pos))
    e_polys = list()
    for char in ["A", "T"]:
        s = q_esc_pos
        e = read.length
        step = 1
        sc_char = char
        if read.strand == "-":
            s = -s - 1
            e = -e - 1
            step = -1
            sc_char = rev_comp[char]
        for i, l, p in find_longest_poly(read.seq, s=s, e=e, step=step, char=sc_char):
            if l < 20 or p < 0.85:
                continue
            assert 0 <= i < read.length - q_esc_pos, (i, q_esc_pos, read.length)
            e_polys.append((i, l, p, char))
    if len(e_polys) > 0:
        i, l, p, char = max(e_polys, key=lambda x: x[2])
        poly_to_gene_gap_size = i
        assert 0 <= poly_to_gene_gap_size < read.length - q_esc_pos, (
            q_esc_pos,
            i,
            l,
            p,
            read.length,
            poly_to_gene_gap_size,
        )
        gaps.add("E{}_{}:{}".format(char, l, poly_to_gene_gap_size))
        gaps.add("ESC:{}".format(read.length - q_esc_pos - poly_to_gene_gap_size))
        assert read.length - q_esc_pos - poly_to_gene_gap_size > 0
    else:
        gaps.add("ESC:{}".format(read.length - q_esc_pos))
    for i1, i2 in zip(intervals[:-1], intervals[1:]):
        (_, i1_l_seg_idx) = i1
        i1_end = segs[i1_l_seg_idx][1]
        q_gap_start, start_slack = get_interval_end(end=i1_end, read=read)
        (i2_f_seg_idx, _) = i2
        i2_start = segs[i2_f_seg_idx][0]
        q_gap_end, end_slack = get_interval_start(start=i2_start, read=read)
        assert 0 < q_gap_start <= q_gap_end < read.length, (
            q_gap_start,
            q_gap_end,
            read.length,
        )
        q_gap_size = q_gap_end - q_gap_start
        q_gap_size = max(0, q_gap_size + start_slack + end_slack)
        assert 0 <= q_gap_size < read.length, (q_gap_size, start_slack, end_slack)
        assert i1_l_seg_idx < i2_f_seg_idx
        gaps.add(
            "{}-{}:{}".format(i1_l_seg_idx, i2_f_seg_idx, q_gap_size),
        )
    read.gaps = sorted(gaps)



def optimize(
    candidate_y_idxs,
    C,
    start,
    end,
    W,
    smoothed_threshold,
    threshold_rate,
    read_support,
):
    yea_mem = dict()
    nay_mem = dict()
    amb_mem = dict()
    for i in range(start, end):
        for j in range(i, end + 1):
            seg_len = candidate_y_idxs[j] - candidate_y_idxs[i] + 1
            h = get_high_threshold(seg_len, smoothed_threshold, threshold_rate)
            l = 1 - h
            c = (C[j] - C[i]) / seg_len
            yea_mem[(i, j)] = c > h
            nay_mem[(i, j)] = c < l
            amb_mem[(i, j)] = W * np.logical_not(
                np.logical_or(yea_mem[(i, j)], nay_mem[(i, j)])
            )
    in_mem = dict()

    def inside(i, j):
        if not (i, j) in in_mem:
            if i == j:
                in_mem[(i, j)] = 0
            else:
                in_mem[(i, j)] = -1 * amb_mem[(i, j)].sum()
        return in_mem[(i, j)]

    out_mem = dict()

    def outside(i, j, k):
        if not (i, j, k) in out_mem:
            if i == j or j == k:
                out_mem[(i, j, k)] = 0
            else:
                X1 = np.logical_and(yea_mem[(i, j)], nay_mem[(j, k)])
                X2 = np.logical_and(nay_mem[(i, j)], yea_mem[(j, k)])
                X3 = np.logical_or(X1, X2)
                X4 = W * X3
                out_mem[(i, j, k)] = np.sum(X4)
                if out_mem[(i, j, k)] < read_support:
                    out_mem[(i, j, k)] = float("-inf")
        return out_mem[(i, j, k)]

    D = dict()
    B = dict()

    def dp(i, j, k):
        # memoization
        if (i, j, k) in D or (i, j, k) in B:
            assert (i, j, k) in D and (i, j, k) in B
            return D[(i, j, k)]
        max_b = (-1, -1, -1)
        max_d = float("-inf")
        # Segment too small: y_idx[j]-y_idx[i] < 5 or
        if (
            candidate_y_idxs[j] - candidate_y_idxs[i] < 5
            or candidate_y_idxs[k] - candidate_y_idxs[j] < 5
        ):
            D[(i, j, k)] = max_d
            B[(i, j, k)] = max_b
            return D[(i, j, k)]
        # Base case: i<j<k=END: k is at the end so no more segmentation
        if k == end:
            D[(i, j, k)] = inside(i, j) + outside(i, j, k) + inside(j, k)
            B[(i, j, k)] = (-1, -1, -1)
            return D[(i, j, k)]
        # Does further segmentation give us better score?
        for k_ in range(k + 1, end + 1):
            cur_b = (j, k, k_)
            cur_d = inside(i, j) + outside(i, j, k) + dp(*cur_b)
            if cur_d > max_d:
                max_d = cur_d
                max_b = cur_b
        D[(i, j, k)] = max_d
        B[(i, j, k)] = max_b
        return D[(i, j, k)]

    # Lower bound on score is no segmentation
    max_d = inside(start, end)
    max_b = (-1, -1, -1)
    for j in range(start + 1, end):
        for k in range(j + 1, end + 1):
            if dp(start, j, k) > max_d:
                max_b = (start, j, k)
                max_d = dp(*max_b)
    del yea_mem, nay_mem, amb_mem, in_mem, out_mem, D
    return B, max_b


def run_optimize(
    candidate_y_idxs,
    fixed_c_idxs,
    coverage,
    read_rep_weights,
    smoothed_threshold,
    threshold_rate,
    min_read_support_outside,
):
    final_c_idxs = set(fixed_c_idxs)
    for idx, (start, end) in enumerate(zip(fixed_c_idxs[:-1], fixed_c_idxs[1:])):
        B, max_b = optimize(
            candidate_y_idxs=candidate_y_idxs,
            C=coverage,
            start=start,
            end=end,
            W=read_rep_weights,
            smoothed_threshold=smoothed_threshold,
            threshold_rate=threshold_rate,
            read_support=min_read_support_outside,
        )
        while max_b != (-1, -1, -1):
            final_c_idxs.update(max_b)
            max_b = B[max_b]
        del B, max_b
    return sorted(final_c_idxs)


def non_desert(y, jump=10):
    l = list()
    for k, group in groupby(enumerate(y), lambda x: x[1] > 0):
        if not k:
            continue
        group = list(group)
        f_idx = group[0][0]
        l_idx = group[-1][0]
        if len(l) == 0:
            l.append([f_idx, l_idx])
        elif l_idx - l[-1][-1] < jump:
            l[-1][-1] = l_idx
        else:
            l.append([f_idx, l_idx])
    return l


def candidates_from_peaks(y):
    c, _ = find_peaks(y)
    c = list(c)
    c.append(0)
    c.append(len(y) - 1)
    c = sorted(set(c))
    return c


def break_large_problems(candidate_y_idxs, fixed_c_idxs, y, max_problem_size, window=5):
    fixed_c_idxs_pairs = sorted(fixed_c_idxs)
    fixed_c_idxs_pairs = [
        (s, e) for s, e in zip(fixed_c_idxs_pairs[:-1], fixed_c_idxs_pairs[1:])
    ]
    new_problems_total_count = 0
    for c_idx_s, c_idx_e in fixed_c_idxs_pairs:
        problem_size = c_idx_e - c_idx_s + 1
        if problem_size <= max_problem_size:
            continue
        new_problems_count = ceil(problem_size / max_problem_size)
        new_problems_size = problem_size / new_problems_count
        new_problems_total_count += new_problems_count - 1
        for i in range(1, new_problems_count):
            mid_anchor = int(c_idx_s + i * new_problems_size)
            max_c_idx_y_v = float("-inf")
            max_c_idx = None
            for c_idx in range(mid_anchor - window, mid_anchor + window):
                if y[candidate_y_idxs[c_idx]] > max_c_idx_y_v:
                    max_c_idx_y_v = y[candidate_y_idxs[c_idx]]
                    max_c_idx = c_idx
            assert max_c_idx_y_v > 0
            fixed_c_idxs.add(max_c_idx)
    return fixed_c_idxs, new_problems_total_count


def process_splicing_data(tint: Tint, ignore_ends: bool):
    pos_to_Yy_idx = dict()
    Yy_idx_to_pos = list()
    Y_raw = list()
    for s, e in tint.intervals:
        y_idx_to_pos = list()
        for p in range(s, e + 1):
            assert not p in pos_to_Yy_idx
            pos_to_Yy_idx[p] = (len(Yy_idx_to_pos), len(y_idx_to_pos))
            y_idx_to_pos.append(p)
        Yy_idx_to_pos.append(y_idx_to_pos)
        Y_raw.append(np.zeros(len(y_idx_to_pos)))
    assert len(pos_to_Yy_idx) == sum(len(y_idx_to_pos) for y_idx_to_pos in Y_raw)
    for read_rep, r_idxes in tint.read_reps:
        for rid in r_idxes:
            tint.reads[rid].data = list()
        for idx, (ts, te) in enumerate(read_rep):
            Y_idx_s, y_idx_s = pos_to_Yy_idx[ts]
            Y_idx_e, y_idx_e = pos_to_Yy_idx[te]
            assert Y_idx_s == Y_idx_e, (Y_idx_s, Y_idx_e)
            Y_idx = Y_idx_s
            if not (ignore_ends and idx == 0):
                Y_raw[Y_idx][y_idx_s] += len(r_idxes)
            if not (ignore_ends and idx == len(read_rep) - 1):
                Y_raw[Y_idx][y_idx_e] += len(r_idxes)
    return (
        pos_to_Yy_idx,
        Yy_idx_to_pos,
        Y_raw,
    )


def run_segment(run_segment_args):
    (
        split_tsv,
        reads_tsv,
        outdir,
        contig,
        tint_id,
    ) = run_segment_args
    tint = read_split(split_tsv)
    read_sequence(tint, reads_tsv)
    segment(tint)
    with open(f"{outdir}/{contig}/segment_{contig}_{tint_id}.tsv", "w+") as out_file:
        print(tint.get_as_record(), file=out_file)
        for read in tint.reads:
            print(read.get_as_record(), file=out_file)
    del tint


def segment(
    tint: Tint,
):
    sigma = segmentation_params.sigma
    smoothed_threshold = segmentation_params.smoothed_threshold
    threshold_rate = segmentation_params.threshold_rate
    variance_factor = segmentation_params.variance_factor
    max_problem_size = segmentation_params.max_problem_size
    min_read_support_outside = segmentation_params.min_read_support_outside
    ignore_ends = segmentation_params.ignore_ends
    (
        pos_to_Yy_idx,
        Yy_idx_to_pos,
        Y_raw,
    ) = process_splicing_data(tint, ignore_ends)
    # Creating a smoothed version of the splicing signal
    Y = [gaussian_filter1d(y, sigma, truncate=4.0) for y in Y_raw]
    # Extracting non-zero values to detect extremely high splicing signals
    Y_none_zero_vals = np.array([v for y in Y for v in y if v > 0])
    variance_threhsold = (
        Y_none_zero_vals.mean() + variance_factor * Y_none_zero_vals.std()
    )

    tint.final_positions = list()
    datas = [list() for _ in tint.read_reps]
    for Y_idx in range(len(Y)):
        y = Y[Y_idx]
        # Getting candidate splice locations for the current tint interval
        candidate_y_idxs = candidates_from_peaks(y)
        # Computing the cumulative coverage for each read rep on the intervals
        # defined by the candidate locations
        C = get_cumulative_coverage(
            read_reps=tint.read_reps,
            candidate_y_idxs=candidate_y_idxs,
            y_idx_to_pos=Yy_idx_to_pos[Y_idx],
            pos_to_Yy_idx=pos_to_Yy_idx,
            Y_idx=Y_idx,
        )
        fixed_c_idxs = set()
        # First and last candidates are always fixed
        fixed_c_idxs.add(0)
        fixed_c_idxs.add(len(candidate_y_idxs) - 1)
        # Any candidate with signal above the threshold is fixed
        for c_idx, y_idx in enumerate(candidate_y_idxs):
            if y[y_idx] > variance_threhsold:
                fixed_c_idxs.add(c_idx)
        # If the number of consecutive unfixed candidate is large,
        # then add some fixed candidates in between
        fixed_c_idxs, new_problems_total_count = break_large_problems(
            candidate_y_idxs, fixed_c_idxs, y, max_problem_size
        )
        fixed_c_idxs = sorted(fixed_c_idxs)
        # for s, e in zip(fixed_c_idxs[:-1], fixed_c_idxs[1:]):
        #     assert e-s <= max_problem_size+5, (s,e,max_problem_size)

        # Getting final candidate indices by runin g a DP formulation
        final_c_idxs = run_optimize(
            candidate_y_idxs=candidate_y_idxs,
            fixed_c_idxs=fixed_c_idxs,
            coverage=C,
            read_rep_weights=np.array([len(ridxs) for _, ridxs in tint.read_reps]),
            smoothed_threshold=smoothed_threshold,
            threshold_rate=threshold_rate,
            min_read_support_outside=min_read_support_outside,
        )
        final_y_idxs = [candidate_y_idxs[c_idx] for c_idx in final_c_idxs]
        refine_y_idxs = refine_segmentation(Y_raw[Y_idx], final_y_idxs, sigma)
        final_y_idxs.extend(refine_y_idxs)
        final_y_idxs.sort()
        tint.final_positions.extend(
            [Yy_idx_to_pos[Y_idx][y_idx] for y_idx in final_y_idxs]
        )
        C = get_cumulative_coverage(
            read_reps=tint.read_reps,
            candidate_y_idxs=final_y_idxs,
            y_idx_to_pos=Yy_idx_to_pos[Y_idx],
            pos_to_Yy_idx=pos_to_Yy_idx,
            Y_idx=Y_idx,
        )
        for seg_idx, (s_yidx, e_yidx) in enumerate(
            zip(final_y_idxs[:-1], final_y_idxs[1:])
        ):
            seg_len = e_yidx - s_yidx + 1
            h = get_high_threshold(seg_len, smoothed_threshold, threshold_rate)
            l = 1 - h
            for rr_idx in range(len(tint.read_reps)):
                cov_ratio = (C[seg_idx + 1][rr_idx] - C[seg_idx][rr_idx]) / seg_len
                assert 0 <= cov_ratio <= 1, (rr_idx, seg_idx, s_yidx, e_yidx, seg_len)

                if cov_ratio > h:
                    datas[rr_idx].append(1)
                elif cov_ratio < l:
                    datas[rr_idx].append(0)
                else:
                    datas[rr_idx].append(2)
        for data in datas:
            data.append(0)

    for data, (_, ridxs) in zip(datas, tint.read_reps):
        for ridx in ridxs:
            tint.reads[ridx].data = data.copy()
    tint.segs = [
        (s, e) for s, e in zip(tint.final_positions[:-1], tint.final_positions[1:])
    ]
    for read in tint.reads:
        read.data.pop()
        assert len(read.data) == len(tint.segs), (
            len(read.data),
            len(tint.segs),
        )
        get_unaligned_gaps_and_polyA(read=read, segs=tint.segs)
    return tint.id


def main():
    args = parse_args()
    args.split_dir = args.split_dir.rstrip("/")

    segment_args = list()
    for split_tsv in glob.iglob(f"{args.split_dir}/*/split_*.tsv*"):
        # Split file name is args.split_dir/<contig>/split_<contig>_<tint_id>.(tsv|tsv.gz)
        contig, tint_id = split_tsv.split("/")[-2:]
        tint_id = tint_id.split("_")[1].split(".")[0]
        tint_id = int(tint_id)
        reads_tsv = f"{args.split_dir}/{contig}/reads_{contig}_{tint_id}.tsv"
        assert os.path.exists(reads_tsv), reads_tsv
        segment_args.append(
            (
                split_tsv,
                reads_tsv,
                args.outdir,
                contig,
                tint_id,
            )
        )
    with Pool(args.threads) as threads_pool:
        for _ in tqdm(
            threads_pool.imap_unordered(
                run_segment,
                segment_args,
                chunksize=1,
            ),
            total=len(segment_args),
        ):
            pass


if __name__ == "__main__":
    main()
