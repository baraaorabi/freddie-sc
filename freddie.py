import argparse
import functools
from multiprocessing import Pool
import sys

from freddie.isoforms import get_isoforms, Isoform, ClusteringParams
from freddie.split import FredSplit

import pulp
from tqdm import tqdm


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
        "--rname-to-celltypes",
        type=str,
        default=None,
        help="Path to TSV file with two columns: 1st  is read namd, 2nd is its cell type(s)."
        + "Reads omitted are assumed belong to no cell type."
        + "Cell types are comma-separated and can be strings (with no commas!).",
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
        "--max-isoform-count",
        default=20,
        type=int,
        help="Maximum number of isoforms to output per transcriptional interval (i.e. ~gene).",
    )
    parser.add_argument(
        "--min-read-support",
        default=3,
        type=int,
        help="Minimum number of reads supporting an isoform.",
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
    parser.add_argument(
        "--sort-output",
        action="store_true",
        help="Sort GTF output by genomic coordinate.",
    )
    parser.add_argument(
        "--readnames-output",
        default=None,
        type=str,
        help="Output path for TSV with isoform_id and read_name columns.",
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


def main():
    args = parse_args()

    print(f"[freddie.py] Args:\n{args}", file=sys.stderr)

    split = FredSplit(
        cigar_max_del=args.cigar_max_del,
        polyA_m_score=args.polyA_m_score,
        polyA_x_score=args.polyA_x_score,
        polyA_min_len=args.polyA_min_len,
        contig_min_len=args.contig_min_len,
        rname_to_celltypes=args.rname_to_celltypes,
    )
    generate_all_tints_f = functools.partial(
        split.generate_all_tints,
        sam_path=args.bam,
    )
    get_isoforms_f = functools.partial(
        get_isoforms,
        params=ClusteringParams(
            ilp_time_limit=args.ilp_time_limit,
            max_correction_len=args.max_correction_len,
            max_correction_count=args.max_correction_count,
            ilp_solver=args.ilp_solver,
            max_isoform_count=args.max_isoform_count,
            min_read_support=args.min_read_support,
        ),
    )

    all_isoforms: list[Isoform] = list()
    if args.output == "":
        outfile = sys.stdout
    else:
        outfile = open(args.output, "w+")
    if args.readnames_output is not None:
        args.readnames_output = open(args.readnames_output, "w+")
    with Pool(args.threads) as pool:
        for isoform_gen in tqdm(
            pool.imap_unordered(get_isoforms_f, generate_all_tints_f()),
            desc="All tint counter",
        ):
            for isoform in isoform_gen:
                if args.sort_output:
                    all_isoforms.append(isoform)
                else:
                    print(isoform, file=outfile)
                if args.readnames_output is not None:
                    for read in isoform.reads:
                        print(f"{isoform.iid}\t{read.name}", file=args.readnames_output)
    if args.sort_output:
        all_isoforms.sort()
        for isoform in all_isoforms:
            print(isoform, file=outfile)
    outfile.close()


if __name__ == "__main__":
    main()
