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
import typing
from matplotlib import pyplot as plt

import numpy as np
import numpy.typing as npt
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
        self.rid = int(re_dict["tint_id"])
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
        self.rid = int(re_dict["rid"])
        self.name = re_dict["name"]
        self.strand = re_dict["strand"]
        all_intervals = [
            (int(ts), int(te), int(qs), int(qe))
            for (ts, te, qs, qe) in SplitRegex.read_interval_prog.findall(
                re_dict["intervals"]
            )
        ]
        self.intervals = all_intervals[2:]
        self.polyA_start = all_intervals[0][1] - all_intervals[0][0] > 10
        self.polyA_end = all_intervals[1][1] - all_intervals[1][0] > 10
        self.seq: str = ""
        self.length: int = 0
        self.data: list = list()
        assert all(
            aet <= bst and aer <= bsr
            for (_, aet, _, aer), (bst, _, bsr, _) in zip(
                self.intervals[:-1], self.intervals[1:]
            )
        )
        assert all(st < et and sr < er for (st, et, sr, er) in self.intervals[3:])

    def get_as_record(self, sep: str = "\t"):
        record = list()
        record.append(str(self.rid))
        record.append(self.name)
        record.append(self.strand)
        record.append("".join(map(str, self.data)))
        return sep.join(record)


class canonInts:
    class intType(enum.IntEnum):
        unaln = 0
        intron = 1
        exon = 3
        polyA = 2

    class cinterval:
        def __init__(self, start: int, end: int) -> None:
            self.start = start
            self.end = end
            self._type_to_rids: dict[canonInts.intType, set[int]] = {
                t: set() for t in canonInts.intType
            }
            self._rid_to_type: dict[int, canonInts.intType] = dict()

        def add_rid(self, rid: int, t: "canonInts.intType"):
            self._type_to_rids[t].add(rid)
            self._rid_to_type[rid] = t

        def add_rids(self, rids: typing.Iterable[int], t: "canonInts.intType"):
            for rid in rids:
                self.add_rid(rid, t)

        def intronic_rids(self) -> set[int]:
            return self._type_to_rids[canonInts.intType.intron]

        def exonic_rids(self) -> set[int]:
            return self._type_to_rids[canonInts.intType.exon]

        def rids(self) -> set[int]:
            return set(self._rid_to_type.keys())

        def change_rid_type(self, rid: int, t: "canonInts.intType") -> None:
            old_t = self._rid_to_type.get(rid, canonInts.intType.unaln)
            self._type_to_rids[old_t].discard(rid)
            self._type_to_rids[t].add(rid)
            self._rid_to_type[rid] = t

        def change_rids_type(self, rids: set[int], t: "canonInts.intType") -> None:
            for rid in rids:
                self.change_rid_type(rid, t)

    def __init__(self, reads: list[Read], rids: typing.Union[set[int], None] = None):
        all_read_rids = {read.rid for read in reads}
        if rids is None:
            rids = all_read_rids
        assert rids.issubset(all_read_rids)
        self._all_reads = reads
        self.rids = rids
        self.rid_to_ridx = {
            read.rid: idx for idx, read in enumerate(reads) if read.rid in rids
        }
        self.intervals = self.make_cintervals(
            [read for read in reads if read.rid in self.rids]
        )
        self.matrix: npt.NDArray[np.uint8] = np.ndarray((0, 0), dtype=np.uint8)

    @staticmethod
    def reverse_dict(d: dict) -> dict:
        result = defaultdict(set)
        for k, v in d.items():
            result[v].add(k)
        return result

    @staticmethod
    def make_cintervals(reads: list[Read]) -> list["canonInts.cinterval"]:
        result: list[canonInts.cinterval] = list()
        breakpoints_set: set[int] = set()
        g = cgranges.cgranges()
        for ridx, read in enumerate(reads, 1):
            for start, end, _, _ in read.intervals:
                g.add("", start, end, ridx)
                breakpoints_set.add(start)
                breakpoints_set.add(end)
            for (_, start, _, _), (end, _, _, _) in zip(
                read.intervals[:-1], read.intervals[1:]
            ):
                g.add("", start, end, -ridx)
        g.index()
        breakpoints: list[int] = sorted(breakpoints_set)
        rid_to_f: dict[int, int] = {read.rid: len(breakpoints) for read in reads}
        rid_to_l: dict[int, int] = {read.rid: -1 for read in reads}
        for start, end in zip(breakpoints[:-1], breakpoints[1:]):
            cint = canonInts.cinterval(
                start=start,
                end=end,
            )
            cint_idx = len(result)
            for _, _, ridx_label in g.overlap("", start, end):
                rid = reads[abs(ridx_label) - 1].rid
                if ridx_label > 0:
                    cint.add_rid(rid, canonInts.intType.exon)
                    rid_to_f[rid] = min(rid_to_f[rid], cint_idx)
                    rid_to_l[rid] = max(rid_to_l[rid], cint_idx)
                else:
                    cint.add_rid(rid, canonInts.intType.intron)
            result.append(cint)
        for read in reads:
            if read.polyA_start:
                for cint in result[: rid_to_f[read.rid]]:
                    cint.change_rid_type(read.rid, canonInts.intType.intron)
            if read.polyA_end:
                for cint in result[rid_to_l[read.rid] + 1 :]:
                    cint.change_rid_type(read.rid, canonInts.intType.intron)
        return result

    # extend the first/last exons of the reads
    # as far as no intron of another read is crossed
    def extend(self, recompute_matrix=True):
        def do_extend(intervals: list[canonInts.cinterval]):
            for idx, curr_cint in enumerate(self.intervals[:-1]):
                next_cint = intervals[idx + 1]
                # no exons
                if len(curr_cint.exonic_rids()) == 0:
                    continue
                # check if any read starts a new intron
                curr_intronic_rids = curr_cint.intronic_rids()
                next_intronic_rids = next_cint.intronic_rids()
                if any(rid not in curr_intronic_rids for rid in next_intronic_rids):
                    continue
                # expands the reads ending at the end of the current interval
                for rid in curr_cint.exonic_rids():
                    next_cint.change_rid_type(rid, canonInts.intType.exon)

        do_extend(self.intervals)
        self.intervals.reverse()
        do_extend(self.intervals)
        self.intervals.reverse()
        if recompute_matrix:
            self.compute_matrix()

    # merge adjacent intervals with the same sets of reads
    def compress(self, recompute_matrix=True):
        result: list[canonInts.cinterval] = list()
        last = self.intervals[0]
        for curr in self.intervals[1:]:
            if (
                curr.exonic_rids() != last.exonic_rids()
                or curr.intronic_rids() != last.intronic_rids()
            ):
                result.append(last)
                last = curr
            else:
                last = canonInts.cinterval(
                    start=last.start,
                    end=curr.end,
                )
                last.add_rids(curr.intronic_rids(), canonInts.intType.intron)
                last.add_rids(curr.exonic_rids(), canonInts.intType.exon)
        result.append(last)
        self.intervals = result
        if recompute_matrix:
            self.compute_matrix()

    def substring(
        self,
        i: typing.Union[int, None],
        j: typing.Union[int, None],
    ) -> "canonInts":
        rids = set()
        for interval in self.intervals[i:j]:
            rids.update(interval.rids())
        return canonInts(self._all_reads, rids)

    def split_on_freq(self, min_freq=0.50) -> tuple["canonInts", "canonInts"]:
        drop_rids = set()
        for interval in self.intervals:
            interval_read_count = len(interval.rids())
            if interval_read_count == 1:
                continue
            if len(S := interval.exonic_rids()) / interval_read_count <= min_freq:
                drop_rids.update(S)
            if len(S := interval.intronic_rids()) / interval_read_count <= min_freq:
                drop_rids.update(S)
        drop_cints = canonInts(self._all_reads, drop_rids)
        keep_cints = canonInts(self._all_reads, self.rids - drop_rids)
        return keep_cints, drop_cints

    def get_matrix(self) -> npt.NDArray[np.uint8]:
        if self.matrix.shape[0] == 0:
            self.compute_matrix()
        return self.matrix

    # convert the intervals to a matrix
    def compute_matrix(self):
        self.matrix = np.full(
            (
                len(self.rids),
                len(self.intervals) + 2,
            ),
            canonInts.intType.unaln,
            dtype=np.uint8,
        )
        rid_to_idx = {rid: idx for idx, rid in enumerate(self.rids)}
        for j, interval in enumerate(self.intervals, start=1):
            for rid, val in interval._rid_to_type.items():
                i = rid_to_idx[rid]
                self.matrix[i, j] = val
        for rid in self.rids:
            i = rid_to_idx[rid]
            read = self._all_reads[self.rid_to_ridx[rid]]
            if read.polyA_start:
                self.matrix[i, 0] = canonInts.intType.polyA
            if read.polyA_end:
                self.matrix[i, -1] = canonInts.intType.polyA

    # remove intervals shorter than min_len
    # if extend run extend() at the end
    # if compress run compress() at the end
    # if recompute_matrix recompute the matrix at the end
    def pop(self, min_len, extend=True, compress=True, recompute_matrix=True):
        def enum_cint_len(args: tuple[int, canonInts.cinterval]):
            _, cint = args
            return cint.end - cint.start

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
                curr_interval = self.intervals[idx]
                next_interval = self.intervals[idx + 1]
                prev_interval = self.intervals[idx - 1]
                if idx == 0:
                    next_interval.start = curr_interval.start
                elif idx == len(self.intervals) - 1:
                    prev_interval.end = curr_interval.end
                else:
                    cost_prev = len(
                        prev_interval.intronic_rids() - curr_interval.intronic_rids()
                    ) + len(prev_interval.exonic_rids() - curr_interval.exonic_rids())
                    cost_next = len(
                        next_interval.intronic_rids() - curr_interval.intronic_rids()
                    ) + len(next_interval.exonic_rids() - curr_interval.exonic_rids())
                    if cost_prev < cost_next:
                        prev_interval.end = curr_interval.end
                    else:
                        next_interval.start = curr_interval.start
                drop_idxs.add(idx)
            else:
                counter = defaultdict(Counter)
                start = self.intervals[idxs[0]].start
                end = self.intervals[idxs[-1]].end
                exonic_rids: set[int] = set()
                intronic_rids: set[int] = set()
                for interval in self.intervals[idxs[0] : idxs[-1] + 1]:
                    length = interval.end - interval.start
                    for rid in interval.exonic_rids():
                        counter[rid][canonInts.intType.exon] += length
                    for rid in interval.intronic_rids():
                        counter[rid][canonInts.intType.intron] += length
                for rid in counter:
                    if (
                        counter[rid][canonInts.intType.exon]
                        > counter[rid][canonInts.intType.intron]
                    ):
                        exonic_rids.add(rid)
                    else:
                        intronic_rids.add(rid)
                self.intervals[idxs[0]] = canonInts.cinterval(
                    start=start,
                    end=end,
                )
                self.intervals[idxs[0]].add_rids(exonic_rids, canonInts.intType.exon)
                self.intervals[idxs[0]].add_rids(
                    intronic_rids, canonInts.intType.intron
                )
                drop_idxs.update(idxs[1:])
        self.intervals = [x for i, x in enumerate(self.intervals) if i not in drop_idxs]
        if extend:
            self.extend(recompute_matrix=False)
        if compress:
            self.compress(recompute_matrix=False)
        if recompute_matrix:
            self.compute_matrix()

    def plot(
        self,
        unique: bool = True,
        min_height: int = 5,
        out_prefix: typing.Union[str, None] = None,
    ):
        fig, axes = plt.subplots(
            2,
            2,
            figsize=(
                15,
                10,
            ),
            sharex=True,
            gridspec_kw={
                "height_ratios": [1, 5],
                "width_ratios": [10, 1],
            },
            squeeze=False,
        )
        plt.subplots_adjust(wspace=0, hspace=0)
        heights_ax = axes[0, 0]
        imshow_ax = axes[1, 0]
        fig.subplots_adjust(hspace=0)
        heights = (
            [0] + [interval.end - interval.start for interval in self.intervals] + [0]
        )
        heights_ax.bar(
            np.arange(0, len(heights), 1),
            heights,
            width=1,
            color=["red" if h < min_height else "blue" for h in heights],
        )
        heights_ax.set_ylabel("Interval length", size=10)
        heights_ax.set_ylim(0, 50)
        yticks = np.arange(5, 50 + 1, 5)
        heights_ax.set_yticks(yticks)
        heights_ax.set_yticklabels(yticks, size=8)
        heights_ax.grid()

        matrix = self.get_matrix()
        if unique:
            matrix = np.unique(matrix, axis=0)
        unique_read_count = matrix.shape[0]
        imshow_ax.imshow(matrix, cmap="binary", aspect="auto", interpolation="none")

        consensus_cols = [
            len(interval.exonic_rids()) * len(interval.intronic_rids()) == 0
            for interval in self.intervals
        ]
        for i, flag in enumerate(consensus_cols):
            if flag:
                imshow_ax.axvline(i, color="green", linewidth=1)

        imshow_ax.set_ylabel(
            f"Read index (n={len(self.rids)}, u={unique_read_count})", size=10
        )
        imshow_ax.set_xlabel("Interval index", size=10)
        starts = (
            [0]
            + [interval.start for interval in self.intervals]
            + [self.intervals[-1].end]
        )
        xticks = np.arange(1, len(starts), max(1, len(starts) // 30))
        if xticks[-1] != len(starts) - 1:
            xticks = np.append(xticks, len(starts) - 1)
        imshow_ax.set_xticks(xticks - 0.5)
        imshow_ax.set_xticklabels(
            [f"{i}) {starts[i]:,}" for i in xticks],
            size=8,
            rotation=90,
        )
        yticks = np.arange(0, unique_read_count, max(1, unique_read_count // 30))
        imshow_ax.set_yticks(yticks - 0.5)
        imshow_ax.set_yticklabels(yticks.astype(int), size=8)
        if out_prefix is not None:
            plt.savefig(f"{out_prefix}.png", dpi=500, bbox_inches="tight")
            plt.savefig(f"{out_prefix}.pdf", bbox_inches="tight")
        imshow_ax.grid(which="major", axis="both")

        for ax in axes[:, 1]:
            ax.tick_params(
                axis="both",
                which="both",
                left=False,
                bottom=False,
                labelleft=False,
                labelbottom=False,
            )
            for _, spine in ax.spines.items():
                spine.set_visible(False)
        plt.tight_layout()
        plt.show()


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
    cints = canonInts(tint.reads)
    for i in range(10):
        cints.pop(i)

    M = cints.get_matrix()
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
