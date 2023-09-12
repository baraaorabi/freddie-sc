#!/usr/bin/env python3
import argparse
from itertools import groupby
import os
import re
import functools
from collections import deque
from typing import Generator, NamedTuple
from multiprocessing import Pool
import gzip

from tqdm import tqdm
import pysam
import networkx as nx
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract alignment information from BAM/SAM file and splits reads"
        + " into distinct transcriptional intervals",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-b",
        "--bam",
        type=str,
        required=True,
        help="Path to sorted and indexed BAM file of reads. "
        + " Assumes splice aligner is used to the genome (e.g. minimap2 -x splice)",
    )
    parser.add_argument(
        "-t",
        "--threads",
        default=1,
        type=int,
        help="Number of threads. Max # of threads used is # of contigs.",
    )
    parser.add_argument(
        "--contig-min-size",
        default=1_000_000,
        type=int,
        help="Minimum contig size. Any contig with less size will not be processes.",
    )
    parser.add_argument(
        "-o",
        "--outdir",
        type=str,
        default="freddie_split/",
        help="Path to output directory.",
    )
    args = parser.parse_args()
    assert args.threads > 0
    return args


cigar_re = re.compile(r"(\d+)([M|I|D|N|S|H|P|=|X]{1})")

query_consuming = [
    pysam.CINS,
    pysam.CSOFT_CLIP,
    pysam.CMATCH,
    pysam.CEQUAL,
    pysam.CDIFF,
]
target_consuming = [
    pysam.CDEL,
    pysam.CMATCH,
    pysam.CEQUAL,
    pysam.CDIFF,
]
exon_consuming = [
    pysam.CINS,
    pysam.CDEL,
    pysam.CMATCH,
    pysam.CEQUAL,
    pysam.CDIFF,
]
intron_consuming = [
    pysam.CINS,
    pysam.CDEL,
    pysam.CMATCH,
    pysam.CEQUAL,
    pysam.CDIFF,
]
target_and_query_consuming = [
    pysam.CMATCH,
    pysam.CEQUAL,
    pysam.CDIFF,
]
target_skipping = [
    pysam.CDEL,
    pysam.CREF_SKIP,
]
cop_to_str = [
    "M",
    "I",
    "D",
    "N",
    "S",
    "H",
    "P",
    "=",
    "X",
    "B",
]

TranscriptionalIntervals = NamedTuple(
    "TranscriptionalIntervals",
    [
        ("intervals", list[tuple[int, int]]),
        ("rids", list[int]),
    ],
)


def fix_intervals(intervals):
    for ts, te, qs, qe, cigar in intervals:
        if len(cigar) == 0:
            continue
        (t, c) = cigar[0]
        if t == pysam.CDEL:
            ts += c
            cigar = cigar[1:]
        if len(cigar) == 0:
            continue
        (t, c) = cigar[-1]
        if t == pysam.CDEL:
            te -= c
            cigar = cigar[:-1]
        if ts < te:
            yield (ts, te, qs, qe, cigar)


# Both query and target intervals are 0-based, start inclusive, and end exlusive
# E.g. the interval 0-10 is 10bp long,
# and includes the base at index 0 but not the base at index 10
def get_intervals(aln, max_del_size=20) -> list[tuple[int, int, int, int]]:
    cigar: list[tuple[int, int]] = aln.cigartuples
    qstart = 0
    if cigar[0][0] == pysam.CSOFT_CLIP:
        qstart += cigar[0][1]
    qlen = 0
    for t, c in cigar:
        if t in query_consuming:
            qlen += c
    assert qlen == len(aln.query_sequence)
    qend = qlen
    if cigar[-1][0] == pysam.CSOFT_CLIP:
        qend -= cigar[-1][1]
    assert qend > qstart

    # aln.reference_start is 0-indexed
    tstart: int = aln.reference_start

    intervals: list[
        tuple[int, int, int, int]
    ] = list()  # list of exonic intervals of the alignment
    qstart_c: int = qstart  # current interval's start on query
    qend_c: int = qstart  # current interval's end on query
    tstart_c: int = tstart  # current interval's start on target
    tend_c: int = tstart  # current interval's end on target
    interval_cigar = list()  # current interval's list of cigar operations
    for t, c in cigar:
        assert 0 <= t < 10, t
        # Treat any deletion (cigar D) longer than max_del_size as a target skip (cigar N)
        if t == pysam.CDEL and c > max_del_size:
            t = pysam.CREF_SKIP
        if t in exon_consuming:
            interval_cigar.append((t, c))
        if t == pysam.CDEL:
            tend_c += c
        elif t == pysam.CINS:
            qend_c += c
        elif t in target_and_query_consuming:
            tend_c += c
            qend_c += c
        # End of the current interval
        if t == pysam.CREF_SKIP:
            intervals.append(
                (
                    tstart_c,
                    tend_c,
                    qstart_c,
                    qend_c,
                )
            )
            assert (
                sum(c for t, c in interval_cigar if t in query_consuming)
                == qend_c - qstart_c
            )
            assert (
                sum(c for t, c in interval_cigar if t in target_consuming)
                == tend_c - tstart_c
            )
            interval_cigar = list()
            tend_c += c
            tstart_c = tend_c
            qstart_c = qend_c
    if tstart_c < tend_c:
        intervals.append(
            (
                tstart_c,
                tend_c,
                qstart_c,
                qend_c,
            )
        )
        assert (
            sum(c for t, c in interval_cigar if t in query_consuming)
            == qend_c - qstart_c
        )
        assert (
            sum(c for t, c in interval_cigar if t in target_consuming)
            == tend_c - tstart_c
        )
    intervals = [
        (st, et, sr, er) for (st, et, sr, er) in intervals if st != et and sr != er
    ]
    return intervals


def find_longest_polyA(
    seq: str,
    start: int,
    end: int,
    match_s: int = 1,
    mismatch_s: int = -2,
):
    result = (0, 0, 0, 0)
    if end - start <= 0:
        return result
    max_length = 0
    for char in "AT":
        if seq[start] == char:
            scores = [match_s]
        else:
            scores = [0]
        for m in (match_s if c == char else mismatch_s for c in seq[start + 1 : end]):
            scores.append(max(0, scores[-1] + m))

        for is_positive, g in groupby(enumerate(scores), lambda x: x[1] > 0):
            if not is_positive:
                continue
            idxs, cur_scores = list(zip(*g))
            _, last_idx = max(zip(cur_scores, idxs))
            last_idx += 1
            first_idx = idxs[0]
            length = last_idx - first_idx
            if length > max_length:
                max_length = length
                result = (
                    0,  # target start, target being polyA
                    length,  # target end
                    first_idx,  # query start
                    last_idx,  # query end
                )
    return result


class Read:
    def __init__(
        self,
        idx: int,
        strand: str,
        intervals: list[tuple[int, int, int, int]],
    ):
        self.idx = idx
        self.strand = strand
        self.intervals = intervals

    def compute_polyA_intervals(self, seq: str):
        self.left_polyA = find_longest_polyA(seq, 0, self.query_start())
        self.right_polyA = find_longest_polyA(seq, self.query_end(), len(seq))

    def target_start(self):
        return self.intervals[0][0]

    def target_end(self):
        return self.intervals[-1][1]

    def query_start(self):
        return self.intervals[0][2]

    def query_end(self):
        return self.intervals[-1][3]


def overlapping_reads(
    sam_path: str, contig: str, names_outpath: str
) -> Generator[dict[int, Read], None, None]:
    sam = pysam.AlignmentFile(sam_path, "rb")
    reads: dict[int, Read] = dict()
    start: int = -1
    end: int = -1
    ridx = 0
    names_outfile = gzip.open(names_outpath, "tw+")
    for aln in sam.fetch(contig=contig):
        if (
            aln.is_unmapped
            or aln.is_supplementary
            or aln.is_secondary
            or aln.reference_name == None
        ):
            continue
        read = Read(
            idx=ridx,
            strand="-" if aln.is_reverse else "+",
            intervals=get_intervals(aln),
        )
        ridx += 1
        print(aln.query_name, file=names_outfile)  # type: ignore
        read.compute_polyA_intervals(aln.query_sequence)  # type: ignore
        s = read.target_start()
        e = read.target_end()
        if (start, end) == (-1, -1):
            start, end = s, e
        if s > end:
            yield reads
            reads = dict()
            end = e
        end = max(end, e)
        reads[read.idx] = read
    if len(reads) > 0:
        yield reads
    names_outfile.close()


def break_tint(tint, reads):
    rids = tint.rids
    intervals = tint.intervals
    start = intervals[0][0]
    end = intervals[-1][1]
    pos_to_intrv = np.zeros(end - start, dtype=int)
    pos_to_intrv[:] = len(intervals)
    intrv_to_rids = [set() for _ in intervals]
    rid_to_intrvs = {rid: set() for rid in rids}
    for idx, (s, e) in enumerate(intervals):
        pos_to_intrv[s - start : e - start] = idx
    edges = dict()
    for rid in rids:
        read = reads[rid]
        alns = read["intervals"]
        for aln in alns:
            s = aln[0]
            v1 = pos_to_intrv[s - start]
            intrv_to_rids[v1].add(rid)
            rid_to_intrvs[rid].add(v1)
        for a1, a2 in zip(alns[:-1], alns[1:]):
            junc_start = a1[1]
            junc_end = a2[0]
            v1 = pos_to_intrv[junc_start - start - 1]
            v2 = pos_to_intrv[junc_end - start]
            assert v1 <= v2 < len(intervals), (
                junc_start,
                junc_end,
                v1,
                v2,
            )
            edges[(v1, v2)] = edges.get((v1, v2), 0) + 1

    graph_edges = [(u, v) for (u, v), w in edges.items() if w >= 2]
    G = nx.Graph()
    G.add_nodes_from(range(len(intervals)))
    G.add_edges_from(graph_edges)
    for c in nx.connected_components(G):
        c_rids = set()
        for i in c:
            c_rids.update(intrv_to_rids[i])
        if len(c_rids) > 2:
            rid_intrvs = set()
            for rid in c_rids:
                rid_intrvs.update(rid_to_intrvs[rid])
            rid_intrvs = sorted(rid_intrvs)
            yield dict(
                intervals=[intervals[i] for i in rid_intrvs],
                rids=sorted(c_rids),
            )


def get_transcriptional_intervals(reads: dict[int, Read]) -> list[TranscriptionalIntervals]:
    BasicInterval = NamedTuple(
        "BasicInterval",
        [
            ("start", int),
            ("end", int),
            ("rids", list[int]),
        ],
    )

    bintervals: list[BasicInterval] = list()
    start, end = -1, -1
    bint_rids: list[int] = list()
    rid_to_bints: dict[int, list[int]] = {read.idx: list() for read in reads.values()}
    for s, e, rid in sorted(
        (ts, te, read.idx) for read in reads.values() for (ts, te, _, _) in read.intervals
    ):
        if (start, end) == (-1, -1):
            start, end = s, e
        if s > end:
            bintervals.append(
                BasicInterval(
                    start=start,
                    end=end,
                    rids=bint_rids,
                )
            )
            start = s
            end = e
            bint_rids = list()
        assert start <= s
        end = max(end, e)
        bint_rids.append(rid)
        rid_to_bints[rid].append(len(bintervals))
    if (start, end) == (-1, -1):
        return list()
    bintervals.append(
        BasicInterval(
            start=start,
            end=end,
            rids=bint_rids,
        )
    )

    enqueued = [False for _ in bintervals]
    tints: list[TranscriptionalIntervals] = list()
    bint_idx: int
    # Breadth-first search
    for bint_idx in range(len(bintervals)):
        if enqueued[bint_idx]:
            continue
        group: list[int] = list()
        queue: deque[int] = deque()
        queue.append(bint_idx)
        enqueued[bint_idx] = True
        while len(queue) > 0:
            bint_idx = queue.pop()
            group.append(bint_idx)
            for rid in bintervals[bint_idx].rids:
                for bint_idx in rid_to_bints[rid]:
                    if not enqueued[bint_idx]:
                        enqueued[bint_idx] = True
                        queue.append(bint_idx)
        tint_rids: set[int] = set()
        group_intervals: list[tuple[int, int]] = list()
        for bint_idx in group:
            tint_rids.update(bintervals[bint_idx].rids)
            group_intervals.append(
                (bintervals[bint_idx].start, bintervals[bint_idx].end)
            )
        # if len(tint_rids) < 3:
        #     continue
        tints.append(
            TranscriptionalIntervals(
                intervals=sorted(group_intervals),
                rids=sorted(tint_rids),
            )
        )
    assert all(enqueued)
    return tints


def run_split(split_args: tuple[str, str, str]):
    sam_path, contig, outdir = split_args
    os.makedirs(outdir, exist_ok=True)
    names_outpath = f"{outdir}/{contig}.read_names.txt.gz"
    tints_outfile = gzip.open(f"{outdir}/{contig}.split.tsv.gz", "tw+")
    tint_idx = 0
    for reads in overlapping_reads(
        sam_path=sam_path,
        contig=contig,
        names_outpath=names_outpath,
    ):
        tints = get_transcriptional_intervals(reads=reads)
        for tint in tints:
            write_tint(
                tint,
                tint_idx,
                reads,
                contig,
                tints_outfile,  # type: ignore
            )
            tint_idx += 1
    return contig


def write_tint(
    tint: TranscriptionalIntervals,
    tint_idx: int,
    reads: dict[int, Read],
    contig: str,
    outfile,
):
    record = list()
    record.append(f"#{contig}")
    record.append(f"{tint_idx}")
    record.append(f"{len(tint.rids)}")
    record.append(",".join(f"{s}-{e}" for s, e in tint.intervals))
    print(
        "\t".join(record),
        file=outfile,  # type: ignore
    )
    for rid in tint.rids:
        read = reads[rid]
        record = list()
        record.append(f"{read.idx}")
        record.append(f"{read.strand}")
        for ts, te, qs, qe in [read.left_polyA, read.right_polyA] + read.intervals:
            record.append(f"{ts}-{te}:{qs}-{qe}")
        print(
            "\t".join(record),
            file=outfile,  # type: ignore
        )


def main():
    args = parse_args()

    args.outdir = args.outdir.rstrip("/")
    os.makedirs(args.outdir, exist_ok=True)
    print("[freddie_split.py] Running split with args:", args)

    contigs = {
        x["SN"]
        for x in pysam.AlignmentFile(args.bam, "rb").header["SQ"]  # type: ignore
        if x["LN"] > args.contig_min_size
    }
    assert (
        len(contigs) > 0
    ), f"No contigs are left! Try checking BAM header or --contig-min-size parameter"
    args.threads = min(args.threads, len(contigs))
    split_args = list()
    for contig in sorted(contigs, reverse=True):
        split_args.append(
            (
                args.bam,
                contig,
                args.outdir,
            )
        )
    threads_pool = Pool(args.threads)
    if args.threads > 1:
        mapper = functools.partial(threads_pool.imap_unordered, chunksize=1)
    else:
        mapper = map
        threads_pool.close()
    for contig in tqdm(
        mapper(run_split, split_args),
        total=len(contigs),
        desc="Spliting contigs into transcriptional intervals",
    ):
        pass
    threads_pool.close()


if __name__ == "__main__":
    main()
