#!/usr/bin/env python3
import argparse
import functools
from multiprocessing import Pool
import sys
from typing import Generator

import pulp
from tqdm import tqdm

from freddie_ilp import FredILP
from freddie_split import FredSplit, Read, Tint
from freddie_segment import canonInts


def parse_args():
    parser = argparse.ArgumentParser(
        description="scFreddie: Detecting isoforms from single-cell long-read RNA-seq data",
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
        help="Number of threads.",
    )
    parser.add_argument(
        "--contig-min-len",
        default=1_000_000,
        type=int,
        help="Minimum contig size. Any contig with less size will not be processes.",
    )
    parser.add_argument(
        "--cigar-max-del",
        default=20,
        type=int,
        help="Maximum deletion size in CIGAR. Deletions (or N operators) longer than this will trigger a splice split.",
    )
    parser.add_argument(
        "--polyA-min-len",
        default=10,
        type=int,
        help="Minimum polyA length. Any polyA shorter than this will be ignored.",
    )
    parser.add_argument(
        "--polyA-scores",
        default="1,-2",
        type=str,
        help="PolyA scores. Comma-separated scores for matching and mismatching bases.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="",
        help="Path to output file. Default: stdout",
    )
    parser.add_argument(
        "--ilp-time-limit",
        type=int,
        default=5 * 60,
        help="Time limit for ILP solver in seconds.",
    )
    parser.add_argument(
        "--max-correction-len",
        type=int,
        default=20,
        help="Maximum length of canonical intervals correction in each read.",
    )
    parser.add_argument(
        "--max-correction-count",
        type=int,
        default=3,
        help="Maximum number of canonical intervals correction in each read.",
    )
    parser.add_argument(
        "--ilp-solver",
        type=str,
        default="COIN_CMD",
        choices=pulp.listSolvers(onlyAvailable=True),
        help="ILP solver.",
    )

    args = parser.parse_args()
    args.polyA_m_score, args.polyA_x_score = list(
        map(int, args.polyA_scores.split(","))
    )

    assert 0 <= args.cigar_max_del
    assert 0 <= args.polyA_min_len
    assert 0 < args.contig_min_len
    assert 0 < args.threads
    assert 0 < args.ilp_time_limit
    assert 0 < args.max_correction_len
    assert 0 < args.max_correction_count

    return args


def get_isoforms(
    tint: Tint,
    ilp_time_limit: int,
    max_correction_len: int,
    max_correction_count: int,
    ilp_solver: str,
) -> Generator[tuple[list[Read], str], None, None]:
    reads = tint.reads
    while len(reads) > 0:
        intervals = canonInts(reads)
        for i in range(10):
            intervals.pop(i)
        ilp = FredILP(intervals)
        ilp.build_model(max_corrections=max_correction_count, slack=max_correction_len)
        status, cost, bins = ilp.solve(solver=ilp_solver, timeLimit=ilp_time_limit)
        if status != pulp.LpStatusOptimal:
            break
        isoform_reads = [tint.reads[i] for i in bins[1]]
        yield isoform_reads, str(isoform_reads)
        reads = [tint.reads[i] for i in bins[0]]


def main():
    args = parse_args()

    print(f"[freddie.py] Args:\n{args}", file=sys.stderr)

    if args.output == "":
        outfile = sys.stdout
    else:
        outfile = open(args.output, "w+")

    split = FredSplit(
        cigar_max_del=args.cigar_max_del,
        polyA_m_score=args.polyA_m_score,
        polyA_x_score=args.polyA_x_score,
        polyA_min_len=args.polyA_min_len,
        contig_min_len=args.contig_min_len,
    )
    get_isoforms_f = functools.partial(
        get_isoforms,
        ilp_time_limit=args.ilp_time_limit,
        max_correction_len=args.max_correction_len,
        max_correction_count=args.max_correction_count,
        ilp_solver=args.ilp_solver,
    )
    generate_all_tints_f = functools.partial(
        split.generate_all_tints,
        sam_path=args.bam,
    )

    with Pool(args.threads) as pool:
        for isoform_gen in tqdm(
            pool.imap_unordered(get_isoforms_f, generate_all_tints_f()),
            desc="All tint counter",
        ):
            for reads, isoform in isoform_gen:
                print(reads, file=outfile)
                print(isoform, file=outfile)
    outfile.close()


if __name__ == "__main__":
    main()
