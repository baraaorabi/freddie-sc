from collections import Counter, defaultdict
import enum
from itertools import groupby
from typing import Union, Iterable

from freddie.split import Read, PairedInterval, Interval

import numpy as np
import numpy.typing as npt
import cgranges


class aln_t(enum.IntEnum):
    """
    An enum to represent the type of an interval
    """

    unaln = 0
    intron = 1
    polyA = 2
    exon = 3


class CanonIntervals:
    """
    A class to represent a set of canoninal intervals of a set of read alignments

    Attributes
    ----------
    intervals : list[CanonIntervals.CanonInterval]
        list of intervals
    matrix : npt.NDArray[np.uint8]
        matrix representation of the intervals

    Methods
    -------
    make_cintervals(reads: list[Read]) -> list[CanonIntervals.CanonInterval]
        make the intervals from a list of reads
    extend(recompute_matrix=True)
        extend the first/last exons of the reads as far as no intron of another read is crossed
    compress(recompute_matrix=True)
        merge adjacent intervals with the same sets of reads
    substring(i: Union[int, None], j: Union[int, None]) -> CanonIntervals
        return a new CanonIntervals object with the intervals i to j
    split_on_freq(min_freq=0.50) -> tuple[CanonIntervals, CanonIntervals]
        split the intervals on the frequency of the reads
    get_matrix() -> npt.NDArray[np.uint8]
        return the matrix representation of the intervals
    compute_matrix()
        builds the matrix representation of the intervals
    pop(min_len, extend=True, compress=True, recompute_matrix=True)
        remove intervals shorter than min_len
    """

    class CanonInterval(Interval):
        """
        A class to represent a single canonical interval

        Attributes
        ----------

        _type_to_ridxs : dict[aln_t, set[int]]
            dictionary mapping the type alignment to the set of reads
            with that type of alignment on the interval
        _ridx_to_type : dict[int, aln_t]
            dictionary mapping the read id to the type of alignment it
            has on the interval

        Methods
        -------
        add_ridx(ridx: int, t: aln_t)
            add a read with a type of alignment to the interval
        add_ridxs(ridxs: Iterable[int], t: aln_t)
            add a set of reads with a type of alignment to the interval
        intronic_ridxs() -> set[int]
            return the set of reads with an intronic alignment on the interval
        exonic_ridxs() -> set[int]
            return the set of reads with an exonic alignment on the interval
        ridxs() -> set[int]
            return the set of reads with an alignment on the interval
        change_ridx_type(ridx: int, t: aln_t)
            change the type of alignment of a read on the interval
        change_ridxs_type(ridxs: set[int], t: aln_t)
            change the type of alignment of a set of reads on the interval
        """

        def __init__(self, start: int, end: int) -> None:
            super().__init__(start=start, end=end)
            self._type_to_ridxs: dict[aln_t, set[int]] = {t: set() for t in aln_t}
            self._ridx_to_type: dict[int, aln_t] = dict()

        def add_ridx(self, ridx: int, t: "aln_t"):
            """
            Add a read index with a type of alignment to the interval

            Parameters
            ----------
            ridx : int
                read index in reads of the CanonIntervals object
            t : aln_t
                type of alignment
            """
            self._type_to_ridxs[t].add(ridx)
            self._ridx_to_type[ridx] = t

        def add_ridxs(self, ridxs: Iterable[int], t: "aln_t"):
            """
            Add a set of read indices with a type of alignment to the interval

            Parameters
            ----------
            ridxs : set[int]
                read indices in reads of the CanonIntervals object
            t : aln_t
                type of alignment
            """
            for ridx in ridxs:
                self.add_ridx(ridx, t)

        def intronic_ridxs(self) -> set[int]:
            """
            Return the set of read indices with an intronic alignment on the interval

            Returns
            -------
            set[int]
            """
            return self._type_to_ridxs[aln_t.intron]

        def exonic_ridxs(self) -> set[int]:
            """
            Return the set of read indices with an exonic alignment on the interval

            Returns
            -------
            set[int]
            """
            return self._type_to_ridxs[aln_t.exon]

        def ridxs(self) -> set[int]:
            """
            Return the set of read indices with an alignment on the interval

            Returns
            -------
            set[int]
            """
            return set(self._ridx_to_type.keys())

        def change_ridx_type(self, ridx: int, aln: "aln_t") -> None:
            """
            Change the type of alignment of a read on the interval

            Parameters
            ----------
            ridx : int
                read index in reads list of the CanonIntervals object
            t : aln_t
                type of target alignment
            """
            old_aln = self._ridx_to_type.get(ridx, aln_t.unaln)
            self._type_to_ridxs[old_aln].discard(ridx)
            self._type_to_ridxs[aln].add(ridx)
            self._ridx_to_type[ridx] = aln

        def change_ridxs_type(self, ridxs: set[int], t: "aln_t") -> None:
            """
            Change the type of alignment of a set of reads on the interval

            Parameters
            ----------
            ridxs : set[int]
                read indices in reads list of the CanonIntervals object
            t : aln_t
                type of target alignment
            """
            for ridx in ridxs:
                self.change_ridx_type(ridx, t)

    def __init__(self, reads: list[Read]):
        self.reads = reads
        self.polyA_slacks = [
            [
                read.polyAs[0].slack,
                read.polyAs[1].slack,
            ]
            for read in self.reads
        ]
        self.intervals = self.make_cintervals((read.intervals for read in self.reads))
        self.validate_intervals()
        self.matrix: npt.NDArray[np.uint8] = np.ndarray((0, 0), dtype=np.uint8)

    def validate_intervals(self):
        for i in self.intervals:
            assert i.intronic_ridxs() & i.exonic_ridxs() == set()
            assert i.ridxs() == i.intronic_ridxs() | i.exonic_ridxs(), (
                i.ridxs(),
                i.intronic_ridxs(),
                i.exonic_ridxs(),
            )
            assert i.start < i.end
        for i1, i2 in zip(self.intervals[:-1], self.intervals[1:]):
            assert i1.end == i2.start, (
                i1.start,
                i1.end,
                i2.start,
                i2.end,
            )

    @staticmethod
    def make_cintervals(
        intervals_iter: Iterable[list[PairedInterval]],
    ) -> list["CanonIntervals.CanonInterval"]:
        """
        Make the canonical intervals from a list of reads

        Parameters
        ----------
        intervals_list : list[Read]
            list of reads

        Returns
        -------
        list[CanonIntervals.CanonInterval]:
            list of canonical intervals of the reads
        """
        result: list[CanonIntervals.CanonInterval] = list()
        breakpoints_set: set[int] = set()
        g = cgranges.cgranges()
        for base1_idx, intervals in enumerate(intervals_iter, 1):
            for interval in intervals:
                g.add("", interval.target.start, interval.target.end, base1_idx)
                breakpoints_set.add(interval.target.start)
                breakpoints_set.add(interval.target.end)
            for interval1, interval2 in zip(intervals[:-1], intervals[1:]):
                g.add("", interval1.target.end, interval2.target.start, -base1_idx)
        g.index()
        breakpoints: list[int] = sorted(breakpoints_set)
        for start, end in zip(breakpoints[:-1], breakpoints[1:]):
            cint = CanonIntervals.CanonInterval(
                start=start,
                end=end,
            )
            for _, _, base1_signed_idx in g.overlap("", start, end):
                idx = abs(base1_signed_idx) - 1
                if base1_signed_idx > 0:
                    cint.add_ridx(idx, aln_t.exon)
                else:
                    cint.add_ridx(idx, aln_t.intron)
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
                if (
                    len(curr_cint.exonic_ridxs()) == 0
                    or len(next_cint.exonic_ridxs()) == 0
                ):
                    continue
                # if any read starts a new intron on the next interval, skip
                if any(
                    ridx not in curr_cint.intronic_ridxs()
                    for ridx in next_cint.intronic_ridxs()
                ):
                    continue
                # if any reads starts a new exon on the next interval, skip
                if any(
                    ridx not in curr_cint.exonic_ridxs()
                    for ridx in next_cint.exonic_ridxs()
                ):
                    continue
                # expands the reads ending at the end of the current interval
                next_length = next_cint.end - next_cint.start
                for ridx in curr_cint.exonic_ridxs() - next_cint.exonic_ridxs():
                    # Read should have no alignment on the next interval
                    assert ridx not in next_cint.intronic_ridxs(), (
                        step,
                        idx,
                        ridx,
                        curr_cint.exonic_ridxs(),
                    )
                    read = self.reads[ridx]
                    polyA_idx = int(not reverse)
                    polyA = read.polyAs[polyA_idx]
                    if polyA.length > 0:
                        if self.polyA_slacks[ridx][polyA_idx] < next_length:
                            continue
                        self.polyA_slacks[ridx][polyA_idx] -= next_length
                    next_cint.add_ridx(ridx, aln_t.exon)

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
        result: list[CanonIntervals.CanonInterval] = list()
        last = self.intervals[0]
        for curr in self.intervals[1:]:
            if (
                curr.exonic_ridxs() != last.exonic_ridxs()
                or curr.intronic_ridxs() != last.intronic_ridxs()
            ):
                result.append(last)
                last = curr
            else:
                last = CanonIntervals.CanonInterval(
                    start=last.start,
                    end=curr.end,
                )
                last.add_ridxs(curr.intronic_ridxs(), aln_t.intron)
                last.add_ridxs(curr.exonic_ridxs(), aln_t.exon)
        result.append(last)
        self.intervals = result
        if recompute_matrix:
            self.compute_matrix()

    def substring(
        self,
        i: Union[int, None],
        j: Union[int, None],
    ) -> "CanonIntervals":
        """
        Return a new CanonIntervals object with the intervals i to j

        Parameters
        ----------
        i : int
            start index of the interval
        j : int
            end index of the interval

        Returns
        -------
        CanonIntervals
        """

        ridxs = set()
        for interval in self.intervals[i:j]:
            ridxs.update(interval.ridxs())
        return CanonIntervals([self.reads[ridx] for ridx in ridxs])

    def split_on_freq(self, min_freq=0.50) -> tuple["CanonIntervals", "CanonIntervals"]:
        """
        Split the intervals on the frequency of the reads alignment type per interval.
        Any read with a frequency of alignment type below min_freq on any interval is dropped.

        Parameters
        ----------
        min_freq : float, optional
            minimum frequency of alignment type

        Returns
        -------
        tuple[CanonIntervals, CanonIntervals]
            tuple of two CanonIntervals objects, the first with the reads above the threshold
            on every interval and the second with the reads below the threshold on any interval
        """
        drop_ridxs = set()
        for interval in self.intervals:
            interval_read_count = len(interval.ridxs())
            if interval_read_count == 1:
                continue
            if len(S := interval.exonic_ridxs()) / interval_read_count <= min_freq:
                drop_ridxs.update(S)
            if len(S := interval.intronic_ridxs()) / interval_read_count <= min_freq:
                drop_ridxs.update(S)
        keep_ridxs = set(range(len(self.reads))) - drop_ridxs
        drop_cints = CanonIntervals([self.reads[ridx] for ridx in sorted(drop_ridxs)])
        keep_cints = CanonIntervals([self.reads[ridx] for ridx in sorted(keep_ridxs)])
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
        The integer representation is defined by aln_t.
        The matrix includes two additional columns for the polyA start and end at the beginning and end of the matrix.
        """
        self.matrix = np.full(
            (
                len(self.reads),
                len(self.intervals) + 2,
            ),
            aln_t.unaln,
            dtype=np.uint8,
        )
        for j, interval in enumerate(self.intervals, start=1):
            for i, val in interval._ridx_to_type.items():
                self.matrix[i, j] = val
        for i, read in enumerate(self.reads):
            if read.polyAs[0].length > 0:
                self.matrix[i, 0] = aln_t.polyA
            if read.polyAs[1].length > 0:
                self.matrix[i, -1] = aln_t.polyA

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

        def enum_cint_len(args: tuple[int, CanonIntervals.CanonInterval]):
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
                        prev_interval.intronic_ridxs() - curr_interval.intronic_ridxs()
                    ) + len(prev_interval.exonic_ridxs() - curr_interval.exonic_ridxs())
                    cost_next = len(
                        next_interval.intronic_ridxs() - curr_interval.intronic_ridxs()
                    ) + len(next_interval.exonic_ridxs() - curr_interval.exonic_ridxs())
                    if cost_prev < cost_next:
                        prev_interval.end = curr_interval.end
                    else:
                        next_interval.start = curr_interval.start
                drop_idxs.add(idx)
            else:
                counter = defaultdict(Counter)
                start = self.intervals[idxs[0]].start
                end = self.intervals[idxs[-1]].end
                exonic_ridxs: set[int] = set()
                intronic_ridxs: set[int] = set()
                for interval in self.intervals[idxs[0] : idxs[-1] + 1]:
                    length = interval.end - interval.start
                    for ridx in interval.exonic_ridxs():
                        counter[ridx][aln_t.exon] += length
                    for ridx in interval.intronic_ridxs():
                        counter[ridx][aln_t.intron] += length
                for ridx in counter:
                    if counter[ridx][aln_t.exon] > counter[ridx][aln_t.intron]:
                        exonic_ridxs.add(ridx)
                    else:
                        intronic_ridxs.add(ridx)
                self.intervals[idxs[0]] = CanonIntervals.CanonInterval(
                    start=start,
                    end=end,
                )
                self.intervals[idxs[0]].add_ridxs(
                    exonic_ridxs,
                    aln_t.exon,
                )
                self.intervals[idxs[0]].add_ridxs(
                    intronic_ridxs,
                    aln_t.intron,
                )
                drop_idxs.update(idxs[1:])
        self.intervals = [x for i, x in enumerate(self.intervals) if i not in drop_idxs]
        if extend:
            self.extend(recompute_matrix=False)
        if compress:
            self.compress(recompute_matrix=False)
        if recompute_matrix:
            self.compute_matrix()
