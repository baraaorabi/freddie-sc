#!/usr/bin/env python3
import argparse
import enum
from itertools import groupby
import os
import functools
from collections import Counter, deque
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


class CIGAR_OPS_SIMPLE(enum.IntEnum):
    both = 0
    target = 1
    query = 2


paired_interval = NamedTuple(
    "paired_interval",
    [
        ("qs", int),
        ("qe", int),
        ("ts", int),
        ("te", int),
    ],
)

op_simply: dict[int, CIGAR_OPS_SIMPLE] = {
    pysam.CSOFT_CLIP: CIGAR_OPS_SIMPLE.query,
    pysam.CINS: CIGAR_OPS_SIMPLE.query,
    pysam.CDEL: CIGAR_OPS_SIMPLE.target,
    pysam.CREF_SKIP: CIGAR_OPS_SIMPLE.target,
    pysam.CMATCH: CIGAR_OPS_SIMPLE.both,
    pysam.CDIFF: CIGAR_OPS_SIMPLE.both,
    pysam.CEQUAL: CIGAR_OPS_SIMPLE.both,
}


TranscriptionalIntervals = NamedTuple(
    "TranscriptionalIntervals",
    [
        ("intervals", list[tuple[int, int]]),
        ("rids", list[int]),
    ],
)


def canonize_cigar(
    cigartuples: list[tuple[int, int]]
) -> list[tuple[CIGAR_OPS_SIMPLE, int]]:
    """
    Canonizes CIGAR tuples by combining operations of the same type and sorting them.
    - all operations are simplified to target-consuming, query-consuming,
      or both-consuming operations.
    - operations between any group (of one or more) both-consuming operations are sorted by their type
    - operations of the same type within each group are sum-combined into a single operation of the same type
    For example, the CIGAR (10M, 20I, 5D, 5I, 5D, 10M) is canonized to (10M, 30I, 10D, 10M).
    These un-canonical CIGARS are rare but they are produced sometimes by Minimap2:
    (https://github.com/lh3/minimap2/issues/1118).

    Parameters
    ----------
    cigartuples : list[tuple[int, int]]
        List of CIGAR tuples (operation, length) as produced by pysam AlignedSegment.cigartuples
    """
    simple_cigartuples = [(op_simply[op], l) for op, l in cigartuples]
    canonized_cigar: list[tuple[CIGAR_OPS_SIMPLE, int]] = list()
    for _, g in groupby(
        simple_cigartuples, key=lambda x: x[0] == CIGAR_OPS_SIMPLE.both
    ):
        C: Counter[CIGAR_OPS_SIMPLE] = Counter()
        for op, l in g:
            C[op] += l
        for op, l in sorted(C.items()):
            if l > 0:
                canonized_cigar.append((op, l))
    return canonized_cigar


def get_intervals(aln, max_del_size=20) -> list[paired_interval]:
    """
    Returns a list of intervals of the alignment.
    Each interval is a tuple of (target_start, target_end, query_start, query_end)
    Both target and query intervals are 0-based, start inclusive, and end exlusive
    E.g. the interval 0-10 is 10bp long and includes the base at index 0 but not the base at index 10.
    Note: the CIGARs are first canonized using canonize_cigar() function.

    Parameters
    ----------
    aln : pysam.AlignedSegment
        pysam AlignedSegment object
    max_del_size : int, optional
        Maximum deletion, by default 20bp. Target skipping cigars (N, D) longer than this will
        trigger a split of the alignment into multiple intervals.
    """
    cigartuples: list[tuple[int, int]] = list(aln.cigartuples)
    cigar = canonize_cigar(cigartuples)
    qstart = 0
    qlen = 0
    for op, l in cigar:
        if op in [CIGAR_OPS_SIMPLE.query, CIGAR_OPS_SIMPLE.both]:
            qlen += l
    assert qlen == len(aln.query_sequence)
    p_interval_wc = NamedTuple(
        "paired_interval_with_cigar",
        [
            ("qs", int),
            ("qe", int),
            ("ts", int),
            ("te", int),
            ("cigar", list[tuple[CIGAR_OPS_SIMPLE, int]]),
        ],
    )
    intervals: list[p_interval_wc] = list()  # list of exonic intervals of the alignment

    qstart: int = 0  # current interval's start on query
    qend: int = 0  # current interval's end on query
    tstart: int = aln.reference_start  # aln.reference_start is 0-indexed
    tend: int = tstart  # current interval's end on target
    for is_splice, g in groupby(
        cigar,
        key=lambda x: x[0] == CIGAR_OPS_SIMPLE.target and x[1] > max_del_size,
    ):
        cur_cigar = list(g)
        for op, l in cur_cigar:
            if op == CIGAR_OPS_SIMPLE.query:
                qend += l
            elif op == CIGAR_OPS_SIMPLE.target:
                tend += l
            elif op == CIGAR_OPS_SIMPLE.both:
                qend += l
                tend += l
        if not is_splice:
            intervals.append(
                p_interval_wc(
                    ts=tstart,
                    te=tend,
                    qs=qstart,
                    qe=qend,
                    cigar=cur_cigar,
                )
            )
            print(qend - qstart, tend - tstart)
        qstart = qend
        tstart = tend
    final_intervals: list[paired_interval] = list()
    for interval in intervals:
        qs = interval.qs
        qe = interval.qe
        ts = interval.ts
        te = interval.te
        cigar = interval.cigar
        assert qe - qs == (
            S := sum(
                l
                for op, l in cigar
                if op in [CIGAR_OPS_SIMPLE.query, CIGAR_OPS_SIMPLE.both]
            )
        ), (qe - qs, S)
        assert te - ts == (
            S := sum(
                l
                for op, l in cigar
                if op in [CIGAR_OPS_SIMPLE.target, CIGAR_OPS_SIMPLE.both]
            )
        ), (qe - qs, S)
        print(interval)
        for op, l in cigar:
            if op == CIGAR_OPS_SIMPLE.both:
                break
            if op == CIGAR_OPS_SIMPLE.query:
                qs += l
            elif op == CIGAR_OPS_SIMPLE.target:
                ts += l
        for op, l in cigar[::-1]:
            if op == CIGAR_OPS_SIMPLE.both:
                break
            if op == CIGAR_OPS_SIMPLE.query:
                qe -= l
            elif op == CIGAR_OPS_SIMPLE.target:
                te -= l
        final_intervals.append(
            paired_interval(
                qs=qs,
                qe=qe,
                ts=ts,
                te=te,
            )
        )
    return final_intervals


def find_longest_polyA(
    seq: str,
    start: int,
    end: int,
    match_s: int = 1,
    mismatch_s: int = -2,
):
    """
    Finds the longest polyA in the sequence.

    Parameters
    ----------
    seq : str
        Sequence
    start : int
        Search start index (inclusive)
    end : int
        Search end index (exclusive)
    match_s : int, optional
        Score for matching base, by default 1
    mismatch_s : int, optional
        Score for mismatching base, by default -2
    """
    result = paired_interval(0, 0, 0, 0)
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
                result = paired_interval(
                    ts=0,
                    te=length,
                    qs=first_idx,
                    qe=last_idx,
                )
    return result


class Read:
    """
    Read object

    Attributes
    ----------
    idx : int
        Read index
    name : str
        Read name
    strand : str
        Read strand
    intervals : list[paired_interval]
        List of intervals of the read
        Each interval is a paired_interval namedtuple of (target_start, target_end, query_start, query_end)
        Both target and query intervals are 0-based, start inclusive, and end exlusive
        E.g. the interval 0-10 is 10bp long, and includes the base at index 0 but not the base at index 10.
    left_polyA : tuple[int, int, int, int]
        Left polyA interval
    right_polyA : tuple[int, int, int, int]
        Right polyA interval
    """

    def __init__(
        self,
        idx: int,
        name: str,
        strand: str,
        intervals: list[paired_interval],
        seq: str,
    ):
        self.idx = idx
        self.name = name
        self.strand = strand
        self.intervals = intervals
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
    """
    Generator of reads overlapping with the contig

    Parameters
    ----------
    sam_path : str
        Path to SAM/BAM file
    contig : str
        Contig name
    names_outpath : str
        Path to output file of read names

    Yields
    ------
    Generator[dict[int, Read], None, None]
        Each generation is a dictionary of reads where any two reads that
        overlap will be in the same dictionary
    """
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
            name=aln.query_name,  # type: ignore
            strand="-" if aln.is_reverse else "+",
            intervals=get_intervals(aln),
            seq=aln.query_sequence,  # type: ignore
        )
        ridx += 1
        print(aln.query_name, file=names_outfile)  # type: ignore
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


def get_transcriptional_intervals(
    reads: dict[int, Read]
) -> list[TranscriptionalIntervals]:
    """
    Returns a list of connected transcriptional intervals from a list of reads

    Parameters
    ----------
    reads : dict[int, Read]
        Dictionary of reads

    Returns
    -------
    list[TranscriptionalIntervals]
    """
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
        (I.ts, I.te, read.idx) for read in reads.values() for I in read.intervals
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


def run_split(split_args: tuple[str, str, str]) -> str:
    """
    Run the splitting stage on a given contig

    Parameters
    ----------
    split_args: tuple[str, str, str]
        A tuple denoting (sam_path, contig, outdir)

    Returns
    -------
    str:
        The contig given as a parameter
    """
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
    """
    Write a transcriptional interval to file

    Parameters
    ----------
    tint : TranscriptionalIntervals
        Transcriptional interval
    tint_idx : int

    reads : dict[int, Read]
        Dictionary of reads
    contig : str
        Contig name
    outfile : file
        Output file
    """
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
        record.append(f"{read.name}")
        record.append(f"{read.strand}")
        for I in [read.left_polyA, read.right_polyA] + read.intervals:
            record.append(f"{I.ts}-{I.te}:{I.qs}-{I.qe}")
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
