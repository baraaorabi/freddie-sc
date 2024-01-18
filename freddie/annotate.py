#!/usr/bin/env python3
from collections import defaultdict
import os
import glob
from multiprocessing import Pool

import argparse
from itertools import groupby


def parse_args():
    parser = argparse.ArgumentParser(
        description="Annotate isoforms in GTF file with gene/transcript information from annotation GTF file using matching intronic intervals.")
    parser.add_argument("-i",
                        "--input-gtf",
                        type=str,
                        required=True,
                        help="Path to the input GTF file")
    parser.add_argument("-a",
                        "--annotation-gtf",
                        type=str,
                        required=True,
                        help="Path to the annotation GTF file")
    parser.add_argument("-o",
                        "--output",
                        type=str,
                        default='annotated_freddie_isoforms.gtf',
                        help="Path to output annotated GTF file. Default: annotated_freddie_isoforms.gtf")
    args = parser.parse_args()
    return args

def get_isoforms(in_gtf):
    isoforms = defaultdict(list)
    for line in open(in_gtf):
        if line.startswith('#'):
            continue
        line = line.strip().split('\t')
        if line[2] != 'exon':
            continue

        info = line[8]
        info = [x.strip().split(' ') for x in info.strip(';').split(';')]
        info = {x[0]:x[1].strip('"') for x in info}

        tid = info['transcript_id']
        isoforms[tid].append((line[0], int(line[3]), int(line[4])))
    return isoforms

def main():
    args = parse_args()

    # Read input GTF file
    in_isoforms = get_isoforms(args.input_gtf)

if __name__ == "__main__":
    main()
