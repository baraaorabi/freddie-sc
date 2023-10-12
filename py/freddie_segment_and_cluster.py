#!/usr/bin/env python3
from collections import Counter, defaultdict
import enum
from itertools import groupby
from multiprocessing import Pool
import os
import glob
import gzip

import argparse
import re
import typing
from matplotlib import pyplot as plt

import numpy as np
import numpy.typing as npt
from tqdm import tqdm
import cgranges


def parse_args():
    parser = argparse.ArgumentParser(
        description="Cluster aligned reads into isoforms",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
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
        help="Path to output directory.",
    )
    parser.add_argument(
        "-t",
        "--threads",
        type=int,
        default=1,
        help="Number of threads for multiprocessing.",
    )
    args = parser.parse_args()
    args.split_dir.rstrip("/")
    assert args.threads > 0
    return args


class SplitRegex:
    index_re = "(0|[1-9][0-9]*)"
    interval_re = f"{index_re}-{index_re}"
    rinterval_re = f"{interval_re}:{interval_re}"
    chr_re = "[0-9A-Za-z!#$%&+./:;?@^_|~-][0-9A-Za-z!#$%&*+./:;=?@^_|~-]*"
    tint_interval_prog = re.compile(interval_re)
    read_interval_prog = re.compile(rinterval_re)
    tint_prog = re.compile(
        r"#(?P<contig>%s)\t" % chr_re
        + r"(?P<tint_id>%s)\t" % index_re
        + r"(?P<read_count>%s)\t" % index_re
        + r"(?P<intervals>%(i)s(,%(i)s)*)\n" % {"i": interval_re}
    )
    read_prog = re.compile(
        r"(?P<rid>%s)\t" % index_re
        + r"(?P<name>[!-?A-~]{1,254})\t"
        + r"(?P<strand>[+-])\t"
        + r"(?P<length>%s)\t" % index_re
        + r"(?P<lpA_a>%(i)s):(?P<lpA_b>%(i)s):(?P<lpA_c>%(i)s)\t" % {"i": index_re}
        + r"(?P<rpA_a>%(i)s):(?P<rpA_b>%(i)s):(?P<rpA_c>%(i)s)\t" % {"i": index_re}
        + r"(?P<intervals>%(i)s(\t%(i)s)*)\n$" % {"i": rinterval_re}
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
    polyA_t = typing.NamedTuple(
        "polyA_t",
        [
            ("overhang", int),
            ("length", int),
            ("slack", int),
        ],
    )

    def __init__(self, split_line: str):
        re_match = SplitRegex.read_prog.match(split_line)
        assert re_match != None
        re_dict = re_match.groupdict()
        self.rid = int(re_dict["rid"])
        self.name = str(re_dict["name"])
        self.strand = str(re_dict["strand"])
        self.intervals = [
            (int(ts), int(te), int(qs), int(qe))
            for (ts, te, qs, qe) in SplitRegex.read_interval_prog.findall(
                re_dict["intervals"]
            )
        ]
        self.polyAs = (
            Read.polyA_t(
                overhang=int(re_dict["lpA_a"]),
                length=int(re_dict["lpA_b"]),
                slack=int(re_dict["lpA_c"]),
            ),
            Read.polyA_t(
                slack=int(re_dict["rpA_a"]),
                length=int(re_dict["rpA_b"]),
                overhang=int(re_dict["rpA_c"]),
            ),
        )
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
    """
    A class to represent a set of canoninal intervals of a set of read alignments

    Attributes
    ----------
    intervals : list[canonInts.cinterval]
        list of intervals
    matrix : npt.NDArray[np.uint8]
        matrix representation of the intervals

    Methods
    -------
    make_cintervals(reads: list[Read]) -> list[canonInts.cinterval]
        make the intervals from a list of reads
    extend(recompute_matrix=True)
        extend the first/last exons of the reads as far as no intron of another read is crossed
    compress(recompute_matrix=True)
        merge adjacent intervals with the same sets of reads
    substring(i: typing.Union[int, None], j: typing.Union[int, None]) -> canonInts
        return a new canonInts object with the intervals i to j
    split_on_freq(min_freq=0.50) -> tuple[canonInts, canonInts]
        split the intervals on the frequency of the reads
    get_matrix() -> npt.NDArray[np.uint8]
        return the matrix representation of the intervals
    compute_matrix()
        builds the matrix representation of the intervals
    pop(min_len, extend=True, compress=True, recompute_matrix=True)
        remove intervals shorter than min_len
    """

    class aln_type(enum.IntEnum):
        """
        An enum to represent the type of an interval
        """

        unaln = 0
        intron = 1
        exon = 3
        polyA = 2

    class cinterval:
        """
        A class to represent a single canonical interval

        Attributes
        ----------
        start : int
            start position of the interval
        end : int
            end position of the interval
        _type_to_rids : dict[canonInts.aln_type, set[int]]
            dictionary mapping the type alignment to the set of reads
            with that type of alignment on the interval
        _rid_to_type : dict[int, canonInts.aln_type]
            dictionary mapping the read id to the type of alignment it
            has on the interval

        Methods
        -------
        add_rid(rid: int, t: canonInts.aln_type)
            add a read with a type of alignment to the interval
        add_rids(rids: typing.Iterable[int], t: canonInts.aln_type)
            add a set of reads with a type of alignment to the interval
        intronic_rids() -> set[int]
            return the set of reads with an intronic alignment on the interval
        exonic_rids() -> set[int]
            return the set of reads with an exonic alignment on the interval
        rids() -> set[int]
            return the set of reads with an alignment on the interval
        change_rid_type(rid: int, t: canonInts.aln_type)
            change the type of alignment of a read on the interval
        change_rids_type(rids: set[int], t: canonInts.aln_type)
            change the type of alignment of a set of reads on the interval
        """

        def __init__(self, start: int, end: int) -> None:
            self.start = start
            self.end = end
            self._type_to_rids: dict[canonInts.aln_type, set[int]] = {
                t: set() for t in canonInts.aln_type
            }
            self._rid_to_type: dict[int, canonInts.aln_type] = dict()

        def add_rid(self, rid: int, t: "canonInts.aln_type"):
            """
            Add a read with a type of alignment to the interval

            Parameters
            ----------
            rid : int
                read id
            t : canonInts.aln_type
                type of alignment
            """
            self._type_to_rids[t].add(rid)
            self._rid_to_type[rid] = t

        def add_rids(self, rids: typing.Iterable[int], t: "canonInts.aln_type"):
            """
            Add a set of reads with a type of alignment to the interval

            Parameters
            ----------
            rids : set[int]
                read ids
            t : canonInts.aln_type
                type of alignment
            """
            for rid in rids:
                self.add_rid(rid, t)

        def intronic_rids(self) -> set[int]:
            """
            Return the set of reads with an intronic alignment on the interval

            Returns
            -------
            set[int]
            """
            return self._type_to_rids[canonInts.aln_type.intron]

        def exonic_rids(self) -> set[int]:
            """
            Return the set of reads with an exonic alignment on the interval

            Returns
            -------
            set[int]
            """
            return self._type_to_rids[canonInts.aln_type.exon]

        def rids(self) -> set[int]:
            """
            Return the set of reads with an alignment on the interval

            Returns
            -------
            set[int]
            """
            return set(self._rid_to_type.keys())

        def change_rid_type(self, rid: int, t: "canonInts.aln_type") -> None:
            """
            Change the type of alignment of a read on the interval

            Parameters
            ----------
            rid : int
                read id
            t : canonInts.aln_type
                type of target alignment
            """
            old_t = self._rid_to_type.get(rid, canonInts.aln_type.unaln)
            self._type_to_rids[old_t].discard(rid)
            self._type_to_rids[t].add(rid)
            self._rid_to_type[rid] = t

        def change_rids_type(self, rids: set[int], t: "canonInts.aln_type") -> None:
            """
            Change the type of alignment of a set of reads on the interval

            Parameters
            ----------
            rids : set[int]
                read ids
            t : canonInts.aln_type
                type of target alignment
            """
            for rid in rids:
                self.change_rid_type(rid, t)

    def __init__(self, reads: list[Read], rids: typing.Union[set[int], None] = None):
        all_read_rids = {read.rid for read in reads}
        if rids is None:
            rids = all_read_rids
        assert rids.issubset(all_read_rids)
        self._all_reads = reads
        self.ridx_to_polyA_slacks = [
            [
                read.polyAs[0].slack,
                read.polyAs[1].slack,
            ]
            for read in reads
        ]
        self.rids = rids
        self.rid_to_ridx = {
            read.rid: idx for idx, read in enumerate(reads) if read.rid in rids
        }
        self.intervals = self.make_cintervals(
            [read for read in reads if read.rid in self.rids]
        )
        self.matrix: npt.NDArray[np.uint8] = np.ndarray((0, 0), dtype=np.uint8)

    @staticmethod
    def make_cintervals(reads: list[Read]) -> list["canonInts.cinterval"]:
        """
        Make the canonical intervals from a list of reads

        Parameters
        ----------
        reads : list[Read]
            list of reads

        Returns
        -------
        list[canonInts.cinterval]:
            list of canonical intervals of the reads
        """
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
        for start, end in zip(breakpoints[:-1], breakpoints[1:]):
            cint = canonInts.cinterval(
                start=start,
                end=end,
            )
            for _, _, ridx_label in g.overlap("", start, end):
                rid = reads[abs(ridx_label) - 1].rid
                if ridx_label > 0:
                    cint.add_rid(rid, canonInts.aln_type.exon)
                else:
                    cint.add_rid(rid, canonInts.aln_type.intron)
            result.append(cint)
        return result

    def extend(self, recompute_matrix=False):
        """
        Extend the first/last exons of the reads as far as no intron of another read is crossed

        Parameters
        ----------
        recompute_matrix : bool, optional
            if True recompute the matrix representation of the intervals
        """

        def do_extend(reverse: bool):
            if reverse:
                start = len(self.intervals) - 1
                end = 0
                step = -1
            else:
                start = 0
                end = len(self.intervals) - 1
                step = 1
            for idx in range(start, end, step):
                curr_cint = self.intervals[idx]
                next_cint = self.intervals[idx + step]
                # no exons
                if len(curr_cint.exonic_rids()) == 0 or len(next_cint.exonic_rids()) == 0:
                    continue
                # if any read starts a new intron on the next interval, skip
                if any(
                    rid not in curr_cint.intronic_rids()
                    for rid in next_cint.intronic_rids()
                ):
                    continue
                # if any reads starts a new exon on the next interval, skip
                if any(
                    rid not in curr_cint.exonic_rids()
                    for rid in next_cint.exonic_rids()
                ):
                    continue
                # expands the reads ending at the end of the current interval
                next_length = next_cint.end - next_cint.start
                for rid in curr_cint.exonic_rids() - next_cint.exonic_rids():
                    # Read should have no alignment on the next interval
                    assert rid not in next_cint.intronic_rids(), (
                        step,
                        idx,
                        rid,
                        curr_cint.exonic_rids(),
                    )
                    ridx = self.rid_to_ridx[rid]
                    read = self._all_reads[ridx]
                    polyA_idx = int(not reverse)
                    polyA = read.polyAs[polyA_idx]
                    if polyA.length > 0:
                        if self.ridx_to_polyA_slacks[ridx][polyA_idx] < next_length:
                            continue
                        self.ridx_to_polyA_slacks[ridx][polyA_idx] -= next_length
                    next_cint.add_rid(rid, canonInts.aln_type.exon)

        do_extend(reverse=False)
        do_extend(reverse=True)
        if recompute_matrix:
            self.compute_matrix()

    def compress(self, recompute_matrix=False):
        """
        Merge adjacent intervals with the same sets of reads

        Parameters
        ----------
        recompute_matrix : bool, optional
            if True recompute the matrix representation of the intervals
        """
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
                last.add_rids(curr.intronic_rids(), canonInts.aln_type.intron)
                last.add_rids(curr.exonic_rids(), canonInts.aln_type.exon)
        result.append(last)
        self.intervals = result
        if recompute_matrix:
            self.compute_matrix()

    def substring(
        self,
        i: typing.Union[int, None],
        j: typing.Union[int, None],
    ) -> "canonInts":
        """
        Return a new canonInts object with the intervals i to j

        Parameters
        ----------
        i : int
            start index of the interval
        j : int
            end index of the interval

        Returns
        -------
        canonInts
        """

        rids = set()
        for interval in self.intervals[i:j]:
            rids.update(interval.rids())
        return canonInts(self._all_reads, rids)

    def split_on_freq(self, min_freq=0.50) -> tuple["canonInts", "canonInts"]:
        """
        Split the intervals on the frequency of the reads alignment type per interval.
        Any read with a frequency of alignment type below min_freq on any interval is dropped.

        Parameters
        ----------
        min_freq : float, optional
            minimum frequency of alignment type

        Returns
        -------
        tuple[canonInts, canonInts]
            tuple of two canonInts objects, the first with the reads above the threshold
            on every interval and the second with the reads below the threshold on any interval
        """
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
        """
        Return the matrix representation of the intervals.
        If the matrix is was never computed, it will be computed before being returned.

        Returns
        -------
        npt.NDArray[np.uint8]
            matrix representation of the intervals
        """
        if self.matrix.shape[0] == 0:
            self.compute_matrix()
        return self.matrix

    def compute_matrix(self):
        """
        Builds the matrix representation of the intervals. The matrix is stored in self.matrix.
        The integer representation is defined by canonInts.aln_type.
        The matrix includes two additional columns for the polyA start and end at the beginning and end of the matrix.
        """
        self.matrix = np.full(
            (
                len(self.rids),
                len(self.intervals) + 2,
            ),
            canonInts.aln_type.unaln,
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
            if read.polyAs[0].length > 0:
                self.matrix[i, 0] = canonInts.aln_type.polyA
            if read.polyAs[1].length > 0:
                self.matrix[i, -1] = canonInts.aln_type.polyA

    def pop(self, min_len, extend=True, compress=True, recompute_matrix=False) -> None:
        """
        Remove intervals shorter than min_len.
        The method first finds neighbouhoods of intervals shorter than min_len.
        If the neighbourhood is a single interval, it is extended to the left or right
        depending on which side has the lowest cost (number of reads with a different alignment type).
        If the neighbourhood is multiple intervals, the intervals of the neighbourhood are merged into a single new interval.
        The reads of the new interval are the union of the reads of the neighbourhood and the type of alignment of each read
        is determined by the alignment type with the highest total length on the interval for that read.

        Parameters
        ----------
        min_len : int
            minimum length of an interval below which the interval is removed/merged
        extend : bool, optional
            if True run extend(recompute_matrix=False) at the end of the method
        compress : bool, optional
            if True run compress(recompute_matrix=False) at the end of the method
        recompute_matrix : bool, optional
            if True recompute the matrix representation of the intervals at the end of the method
        """
        if len(self.intervals) <= 1:
            return

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
                if idx == 0:
                    next_interval = self.intervals[idx + 1]
                    next_interval.start = curr_interval.start
                elif idx == len(self.intervals) - 1:
                    prev_interval = self.intervals[idx - 1]
                    prev_interval.end = curr_interval.end
                else:
                    prev_interval = self.intervals[idx - 1]
                    next_interval = self.intervals[idx + 1]
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
                        counter[rid][canonInts.aln_type.exon] += length
                    for rid in interval.intronic_rids():
                        counter[rid][canonInts.aln_type.intron] += length
                for rid in counter:
                    if (
                        counter[rid][canonInts.aln_type.exon]
                        > counter[rid][canonInts.aln_type.intron]
                    ):
                        exonic_rids.add(rid)
                    else:
                        intronic_rids.add(rid)
                self.intervals[idxs[0]] = canonInts.cinterval(
                    start=start,
                    end=end,
                )
                self.intervals[idxs[0]].add_rids(exonic_rids, canonInts.aln_type.exon)
                self.intervals[idxs[0]].add_rids(
                    intronic_rids, canonInts.aln_type.intron
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
        figsize: tuple[int, int] = (15, 10),
        out_prefix: typing.Union[str, None] = None,
    ):
        """
        Plot the intervals and the matrix representation of the intervals using matplotlib's imshow

        Parameters
        ----------
        unique : bool, optional
            if True plot only unique rows of the matrix
        min_height : int, optional
            intervals shorter than min_height are plotted in red, otherwise in blue
        out_prefix : str, optional
            if not None save the plot to out_prefix.png and out_prefix.pdf
        """
        fig, axes = plt.subplots(
            2,
            2,
            figsize=figsize,
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
                imshow_ax.axvline(i + 1, color="green", linewidth=1)

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


def generate_tint_lines(split_dir: str) -> typing.Generator[list[str], None, None]:
    """
    Generator of the lines of the split tsv files

    Parameters
    ----------
    split_dir : str
        linux path (with wildcards) to the split tsv files

    Yields
    ------
    list[str]
        list of lines for each transcriptional intervals (tint) in the split tsv files
    """
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
    """
    Segment and cluster a set of transcriptional intervals (tint) lines

    Parameters
    ----------
    tint_lines : list[str]
        list of lines for each transcriptional intervals (tint)
    """
    tint = Tint(tint_lines)
    del tint_lines
    cints = canonInts(tint.reads)
    for i in range(10):
        cints.pop(i)

    M = cints.get_matrix()
    print(*M.shape)


def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    tqdm_desc = "Segmenting and clustering transcriptional intervals"
    if args.threads == 1:
        for _ in tqdm(
            map(
                segment_and_cluster,
                generate_tint_lines(args.split_dir),
            ),
            desc=tqdm_desc,
        ):
            pass
    else:
        threads_pool = Pool(args.threads)
        with Pool(args.threads) as threads_pool:
            for _ in tqdm(
                threads_pool.imap_unordered(
                    segment_and_cluster,
                    generate_tint_lines(args.split_dir),
                ),
                desc=tqdm_desc,
            ):
                pass


if __name__ == "__main__":
    main()
