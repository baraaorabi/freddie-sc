#!/usr/bin/env python3
from collections import Counter, defaultdict
import copy
import enum
from itertools import groupby
from multiprocessing import Pool
import os
import glob
import gzip

import argparse
import re
from typing import Generator, NamedTuple

import numpy as np
from tqdm import tqdm
import cgranges


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
    args = parser.parse_args()
    args.split_dir.rstrip("/")
    assert args.threads > 0
    return args


class SplitRegex:
    interval_re = "([0-9]+)-([0-9]+)"
    rinterval_re = "%(i)s:%(i)s" % {"i": interval_re}
    chr_re = "[0-9A-Za-z!#$%&+./:;?@^_|~-][0-9A-Za-z!#$%&*+./:;=?@^_|~-]*"
    tint_interval_prog = re.compile(interval_re)
    read_interval_prog = re.compile(rinterval_re)
    tint_prog = re.compile(
        r"#%(contig)s\t" % {"contig": f"(?P<contig>{chr_re})"}
        + r"%(idx)s\t" % {"idx": "(?P<tint_id>[0-9]+)"}
        + r"%(count)s\t" % {"count": "(?P<read_count>[0-9]+)"}
        + r"%(intervals)s\n"
        % {"intervals": "(?P<intervals>%(i)s(,%(i)s)*)" % {"i": interval_re}}
    )
    read_prog = re.compile(
        r"%(idx)s\t" % {"idx": "(?P<rid>[0-9]+)"}
        + r"%(name)s\t" % {"name": "(?P<name>[!-?A-~]{1,254})"}
        + r"%(strand)s\t" % {"strand": "(?P<strand>[+-])"}
        + r"%(intervals)s\n$"
        % {"intervals": r"(?P<intervals>%(i)s(\t%(i)s)*)" % {"i": rinterval_re}}
    )


class Tint:
    def __init__(self, tint_lines: list[str]):
        split_header = tint_lines[0]
        re_dict = SplitRegex.tint_prog.match(split_header)
        assert re_dict != None
        re_dict = re_dict.groupdict()

        self.contig: str = re_dict["contig"]
        self.id = int(re_dict["tint_id"])
        self.read_count: int = int(re_dict["read_count"])
        self.intervals: list[tuple[int, int]] = [
            (int(x[0]), int(x[1]))
            for x in SplitRegex.tint_interval_prog.findall(re_dict["intervals"])
        ]
        assert all(
            a[1] < b[0] for a, b in zip(self.intervals[:-1], self.intervals[1:])
        ), self.intervals
        assert all(s < e for s, e in self.intervals), self.intervals
        self.reads: list[Read] = list()
        for line in tint_lines[1:]:
            self.reads.append(Read(line))
        assert len(self.reads) == self.read_count, (len(self.reads), self.read_count)


class Read:
    def __init__(self, split_line: str):
        re_dict = SplitRegex.read_prog.match(split_line)
        assert re_dict != None
        re_dict = re_dict.groupdict()
        self.id = int(re_dict["rid"])
        self.name = re_dict["name"]
        self.strand = re_dict["strand"]
        self.intervals = [
            (int(ts), int(te), int(qs), int(qe))
            for (ts, te, qs, qe) in SplitRegex.read_interval_prog.findall(
                re_dict["intervals"]
            )
        ]
        self.seq: str = ""
        self.length: int = 0
        self.data: list = list()
        assert all(
            aet <= bst and aer <= bsr
            for (_, aet, _, aer), (bst, _, bsr, _) in zip(
                self.intervals[2:-1], self.intervals[3:]
            )
        )
        assert all(st < et and sr < er for (st, et, sr, er) in self.intervals[3:])

    def get_as_record(self, sep: str = "\t"):
        record = list()
        record.append(str(self.id))
        record.append(self.name)
        record.append(self.strand)
        record.append("".join(map(str, self.data)))
        return sep.join(record)


class CanonicalItervals:
    cinterval = NamedTuple(
        "cinterval",
        [
            ("start", int),
            ("end", int),
            ("intronic_rids", set[int]),
            ("exonic_rids", set[int]),
        ],
    )

    class IntervalType(enum.IntEnum):
        UNALIGNED = 0
        INTRONIC = 1
        EXONIC = 2

    def __init__(self, tint: Tint):
        self.intervals = self.make_cintervals(tint.reads)
        self.read_count = tint.read_count

    @staticmethod
    def make_cintervals(reads: list[Read]) -> list[cinterval]:
        result: list[CanonicalItervals.cinterval] = list()
        breakpoints_set: set[int] = set()
        g = cgranges.cgranges()
        for idx, read in enumerate(reads, 1):
            for start, end, _, _ in read.intervals:
                g.add("", start, end, idx)
                breakpoints_set.add(start)
                breakpoints_set.add(end)
            for (_, start, _, _), (end, _, _, _) in zip(
                read.intervals[:-1], read.intervals[1:]
            ):
                g.add("", start, end, -idx)
        g.index()
        breakpoints: list[int] = sorted(breakpoints_set)
        for start, end in zip(breakpoints[:-1], breakpoints[1:]):
            intronic_rids: set[int] = set()
            exonic_rids: set[int] = set()
            for _, _, ridx in g.overlap("", start, end):
                if ridx > 0:
                    exonic_rids.add(ridx)
                else:
                    intronic_rids.add(-ridx)
            result.append(
                CanonicalItervals.cinterval(
                    start=start,
                    end=end,
                    intronic_rids=intronic_rids,
                    exonic_rids=exonic_rids,
                )
            )
        return result

    # extend the first/last exons of the reads
    # as far as no intron of another read is crossed
    def extend(self):
        def do_extend(intervals: list[CanonicalItervals.cinterval]):
            for idx, interval in enumerate(self.intervals[:-1]):
                next_interval = intervals[idx + 1]
                # no exons
                if len(interval.exonic_rids) == 0:
                    continue
                # check if any read starts a new intron
                if any(
                    rid not in interval.intronic_rids
                    for rid in next_interval.intronic_rids
                ):
                    continue
                # expands the reads ending at the end of the current interval
                next_interval.exonic_rids.update(interval.exonic_rids)

        do_extend(self.intervals)
        self.intervals.reverse()
        do_extend(self.intervals)
        self.intervals.reverse()

    # merge adjacent intervals with the same sets of reads
    def compress(self):
        result: list[CanonicalItervals.cinterval] = list()
        last = self.intervals[0]
        for curr in self.intervals[1:]:
            if (
                curr.exonic_rids != last.exonic_rids
                or curr.intronic_rids != last.intronic_rids
            ):
                result.append(last)
                last = curr
            else:
                last = CanonicalItervals.cinterval(
                    start=last.start,
                    end=curr.end,
                    intronic_rids=curr.intronic_rids,
                    exonic_rids=curr.exonic_rids,
                )
        result.append(last)
        self.intervals = result

    # convert the intervals to a matrix
    def to_matrix(self) -> np.ndarray:
        M = np.zeros((len(self.intervals), self.read_count), dtype=np.uint8)
        for i, interval in enumerate(self.intervals):
            for j in interval.exonic_rids:
                M[i, j] = CanonicalItervals.IntervalType.EXONIC
            for j in interval.intronic_rids:
                M[i, j] = CanonicalItervals.IntervalType.INTRONIC
        return M

    # remove intervals shorter than min_len
    # if extend run extend() at the end
    # if compress run compress() at the end
    def pop(self, min_len, extend=True, compress=True):
        def enum_cint_len(args: tuple[int, CanonicalItervals.cinterval]):
            _, cint = args
            return cint.end - cint.start

        result = copy.deepcopy(self.intervals)
        drop_idxs = set()
        for is_short, g in groupby(
            enumerate(self.intervals), key=lambda x: enum_cint_len(x) < min_len
        ):
            is_short: bool
            if not is_short:
                continue
            idxs: list[int] = list()
            for idx, _ in g:
                idxs.append(idx)
            if len(idxs) == 1:
                idx = idxs[0]
                curr_interval = result[idx]
                next_interval = result[idx + 1]
                prev_interval = result[idx - 1]
                if idx == 0:
                    result[idx + 1] = CanonicalItervals.cinterval(
                        start=curr_interval.start,
                        end=next_interval.end,
                        intronic_rids=next_interval.intronic_rids,
                        exonic_rids=next_interval.exonic_rids,
                    )
                elif idx == len(result) - 1:
                    result[idx - 1] = CanonicalItervals.cinterval(
                        start=prev_interval.start,
                        end=curr_interval.end,
                        intronic_rids=prev_interval.intronic_rids,
                        exonic_rids=prev_interval.exonic_rids,
                    )
                else:
                    cost_prev = len(
                        prev_interval.intronic_rids - curr_interval.intronic_rids
                    ) + len(prev_interval.exonic_rids - curr_interval.exonic_rids)
                    cost_next = len(
                        next_interval.intronic_rids - curr_interval.intronic_rids
                    ) + len(next_interval.exonic_rids - curr_interval.exonic_rids)
                    if cost_prev < cost_next:
                        result[idx - 1] = CanonicalItervals.cinterval(
                            start=prev_interval.start,
                            end=curr_interval.end,
                            intronic_rids=prev_interval.intronic_rids,
                            exonic_rids=prev_interval.exonic_rids,
                        )
                    else:
                        result[idx + 1] = CanonicalItervals.cinterval(
                            start=curr_interval.start,
                            end=next_interval.end,
                            intronic_rids=next_interval.intronic_rids,
                            exonic_rids=next_interval.exonic_rids,
                        )
                drop_idxs.add(idx)
            else:
                counter = defaultdict(Counter)
                start = self.intervals[idxs[0]].start
                end = self.intervals[idxs[-1]].end
                exonic_rids: set[int] = set()
                intronic_rids: set[int] = set()
                for interval in self.intervals[idxs[0] : idxs[-1] + 1]:
                    length = interval.end - interval.start
                    for rid in interval.exonic_rids:
                        counter[rid][CanonicalItervals.IntervalType.EXONIC] += length
                    for rid in interval.intronic_rids:
                        counter[rid][CanonicalItervals.IntervalType.INTRONIC] += length
                for rid in counter:
                    if (
                        counter[rid][CanonicalItervals.IntervalType.EXONIC]
                        > counter[rid][CanonicalItervals.IntervalType.INTRONIC]
                    ):
                        exonic_rids.add(rid)
                    else:
                        intronic_rids.add(rid)
                result[idxs[0]] = CanonicalItervals.cinterval(
                    start=start,
                    end=end,
                    intronic_rids=intronic_rids,
                    exonic_rids=exonic_rids,
                )
                drop_idxs.update(idxs[1:])
        self.intervals = [x for i, x in enumerate(result) if i not in drop_idxs]
        if extend:
            self.extend() 
        if compress:
            self.compress()


def generate_tint_lines(split_dir: str) -> Generator[list[str], None, None]:
    for split_tsv in glob.iglob(f"{split_dir}/*.split.tsv*"):
        if split_tsv.endswith(".gz"):
            infile = gzip.open(split_tsv, "rt")
        else:
            infile = open(split_tsv)
        line: str = infile.readline()  # type: ignore
        assert line.startswith("#")
        tint_lines = [line]
        for line in infile:  # type: ignore
            if line.startswith("#"):
                yield tint_lines
                tint_lines = [line]
            else:
                tint_lines.append(line)
        yield tint_lines


def segment_and_cluster(tint_lines: list[str]):
    tint = Tint(tint_lines)
    del tint_lines
    cints = CanonicalItervals(tint)
    for i in range(10):
        cints.pop(i)

    M = cints.to_matrix()
    print(len(cints.intervals))


def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    with Pool(args.threads) as threads_pool:
        for _ in tqdm(
            threads_pool.imap_unordered(
                segment_and_cluster,
                generate_tint_lines(args.split_dir),
                chunksize=1,
            ),
            desc="Segmenting and clustering transcriptional intervals",
        ):
            pass


if __name__ == "__main__":
    main()
