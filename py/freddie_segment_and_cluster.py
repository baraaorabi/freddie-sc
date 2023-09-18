#!/usr/bin/env python3
from multiprocessing import Pool
import os
import glob
import gzip

from itertools import groupby
import argparse
import re
from typing import Generator

import numpy as np
from tqdm import tqdm


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
    def __init__(self, split_header):
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
        self.reads: list[Read] = list()
        # read_reps is a list of items (key,val)
        # key is a tuple of intervals tuples (start, end) of the read
        # val is a list of read indices that have the same intervals
        self.read_reps: list[
            tuple[
                tuple[tuple[int, int], ...], list[int]  # Read intervals  # (start, end)
            ]  # list of read indices
        ] = list()
        # final_positions is a list of positions
        self.final_positions: list[int] = list()
        # segs is a list of tuples (start, end) of segments
        self.segs: list[tuple[int, int]] = list()  # (start, end)
        assert all(
            a[1] < b[0] for a, b in zip(self.intervals[:-1], self.intervals[1:])
        ), self.intervals
        assert all(s < e for s, e in self.intervals)

    # def compute_read_reps(self):
    #     D = dict()
    #     for ridx, read in enumerate(self.reads):
    #         k = tuple(((ts, te) for ts, te, _, _ in read.intervals))
    #         if not k in D:
    #             D[k] = list()
    #         D[k].append(ridx)
    #     self.read_reps = list(D.items())
    #     assert self.read_count == sum(len(x) for _, x in self.read_reps)

    # def get_as_record(self, sep: str = "\t"):
    #     record = list()
    #     record.append("#{}".format(self.contig))
    #     record.append(str(self.id))
    #     record.append(",".join(map(str, self.final_positions)))
    #     return sep.join(record)


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
        # record.append("".join("{},".format(g) for g in self.gaps))
        return sep.join(record)


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


def generate_tint(split_lines: list[str]) -> Tint:
    tint = Tint(split_lines[0])
    for read_line in split_lines[1:]:
        tint.reads.append(Read(read_line))
    return tint


def segment_and_cluster(tint_lines: list[str]):
    tint = generate_tint(tint_lines)
    del tint_lines


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
