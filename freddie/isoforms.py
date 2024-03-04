import enum
import functools
from dataclasses import dataclass
import pickle

from freddie.ilp import FredILP, IlpParams, UnsolvableILP, TimeoutILP
from freddie.segment import CanonIntervals, PairedInterval, aln_t
from freddie.split import Interval, Read, Tint

import numpy as np
import pulp


class timeoutStrat(enum.IntEnum):
    stop = 0
    subsample = 1


@dataclass
class IsoformsParams:
    max_isoform_count: int = 20
    min_read_support: int = 3
    timeout_stategy: timeoutStrat = timeoutStrat.stop
    ilp_params: IlpParams = IlpParams()

    def __post_init__(self):
        assert 1 <= self.max_isoform_count
        assert 1 <= self.min_read_support


@dataclass
class IntervalSupport(Interval):
    support: float = 0.0


@functools.total_ordering
class Isoform:
    """
    Isoform class.

    Attributes:
        tid: Tint ID
        reads: List of reads comprising the isoform
        iid: Isoform index
        contig: Contig
        strand: Strand of the isoform if it can be determined from the read polyA tails (i.e. + or -). Otherwise, "."
        exons: List of genomic intervals (i.e. exons) comprising the isoform with support values.
                The support value is computed by adding the number of bases covered by
                each read in the interval and dividing by the interval length.
        cell_types: Tuple of cell types

    Methods:
        __eq__: Equality operator
        __lt__: Less than operator
        __repr__: String representation of the isoform in GTF format
    """

    def __init__(
        self,
        tid: int,
        contig: str,
        reads: list[Read],
        isoform_index: int,
    ) -> None:
        self.tid = tid
        self.reads = reads
        self.iid = isoform_index
        self.contig = contig
        self.exons: list[IntervalSupport] = list()
        cell_types_set = set()
        for read in self.reads:
            if len(cell_types_set) == 0:
                cell_types_set.add("NA")
            for ct in read.cell_types:
                cell_types_set.add(ct)
        self.cell_types = tuple(sorted(cell_types_set))

        canon_ints = CanonIntervals(self.reads)
        for i in range(10):
            canon_ints.pop(i)
        intervals: list[IntervalSupport] = list()
        for i in canon_ints.intervals:
            if (e_cnt := len(i.exonic_ridxs())) > len(i.intronic_ridxs()):
                intervals.append(IntervalSupport(i.start, i.end, e_cnt))
        intervals.sort()
        for i in intervals:
            # Add first exon
            if len(self.exons) == 0:
                self.exons.append(i)
                continue
            # Current interval is not adjacent to previous interval: add new exon
            if self.exons[-1].end < i.start:
                self.exons.append(i)
                continue
            # Current interval is adjacent to previous interval: merge exons and update support
            e = self.exons[-1]

            self.exons[-1] = IntervalSupport(
                e.start,
                i.end,
                (len(e) * e.support + len(i) * i.support) / (len(e) + len(i)),
            )

        self.strand = "."
        for read in self.reads:
            if read.polyAs[0].length > 0:
                self.strand = "-"
                break
            if read.polyAs[1].length > 0:
                self.strand = "+"
                break

    def __eq__(self, __value: object) -> bool:
        if not isinstance(__value, Isoform):
            return NotImplemented
        return self.contig == __value.contig and self.exons == __value.exons

    def __lt__(self, __value: object) -> bool:
        if not isinstance(__value, Isoform):
            return NotImplemented
        if self.contig != __value.contig:
            # If contig is a number, sort by number
            if self.contig.isnumeric() and __value.contig.isnumeric():
                return int(self.contig) < int(__value.contig)
            return self.contig < __value.contig
        return self.exons < __value.exons

    def __repr__(self) -> str:
        gtf_records = list()
        gene_id = f"{self.contig}_{self.tid}"
        isoform_id = f"{gene_id}_{self.iid}"
        gtf_records.append(
            "\t".join(
                [
                    self.contig,
                    "freddie",
                    "transcript",
                    f"{self.exons[0].start + 1}",
                    f"{self.exons[-1].end}",
                    ".",
                    self.strand,
                    ".",
                    " ".join(
                        [
                            f'gene_id "{gene_id}";',
                            f'transcript_id "{isoform_id}";',
                            f'read_support "{len(self.reads)}";',
                            f'cell_types "{",".join(self.cell_types)}";',
                        ]
                    ),
                ]
            )
        )
        for idx, exon in enumerate(self.exons, start=1):
            gtf_records.append(
                "\t".join(
                    [
                        self.contig,
                        "freddie",
                        "exon",
                        f"{exon.start + 1}",
                        f"{exon.end}",
                        ".",
                        self.strand,
                        ".",
                        " ".join(
                            [
                                f'gene_id "{gene_id}";',
                                f'transcript_id "{isoform_id}";',
                                f'read_support "{exon.support:.2f}";',
                                f'exon_number "{idx}";',
                            ]
                        ),
                    ]
                )
            )
        return "\n".join(gtf_records)


def get_isoforms(
    tint: Tint,
    params: IsoformsParams = IsoformsParams(),
) -> tuple[Tint, list[Isoform]]:
    """
    Returns isoforms for the given Tint.

    Args:
        tint: Tint
        params: Isoform params dataclass

    Returns:
        isoforms: list[Isoform]
    """
    assert params.min_read_support > 0
    reads: list[Read] = tint.reads
    isoforms = list()
    for isoform_index in range(params.max_isoform_count):
        try:
            canon_ints, recycling_bin, isoform_bin, unsampled_bin = run_ilp_loop(
                reads, params
            )
        except (UnsolvableILP, TimeoutILP) as e:
            pickle.dump(
                reads,
                open(
                    f"tints/{str(e).replace(' ', '')}.contig_{tint.contig}.tint_{tint.tid}.pickle",
                    "wb+",
                ),
            )
            break
        if len(isoform_bin) < params.min_read_support:
            break
        isoform_reads = [reads[ridx] for ridx in isoform_bin]
        recycling_reads = [reads[ridx] for ridx in recycling_bin]
        if len(unsampled_bin) > 0:
            unsampled_isoform_reads, unsampled_recycing_reads = get_compatible_reads_bins(
                [reads[ridx] for ridx in unsampled_bin],
                isoform_reads,
                canon_ints,
                [aln_t.exon] * (len(canon_ints.intervals) - 1),
                params.ilp_params.max_correction_len,
            )
            isoform_reads.extend(unsampled_isoform_reads)
            recycling_reads.extend(unsampled_recycing_reads)

        isoform = Isoform(
            tid=tint.tid,
            contig=tint.contig,
            reads=isoform_reads,
            isoform_index=isoform_index,
        )
        isoforms.append(isoform)
        # Remove the reads that were used to construct the isoform
        reads = recycling_reads
        if len(reads) < params.min_read_support:
            break
    return tint, isoforms


def run_ilp(
    canon_ints: CanonIntervals, params: IsoformsParams
) -> tuple[int, list[int], list[int]]:
    """
    Args:
        canon_ints: CanonIntervals
        params: Isoforms params dataclass

    Returns:
        recycling_bin: List of ridxs belonging to the recycling bin
        isoform_bin: List of ridxs belonging to the isoform bin
    """
    ilp: FredILP = FredILP(canon_ints, params.ilp_params)
    ilp.build_model(K=2)
    status, (recycling_bin, isoform_bin) = ilp.solve()
    return status, recycling_bin, isoform_bin


def canonize_reads(reads):
    canon_ints = CanonIntervals(reads)
    for i in range(10):
        canon_ints.pop(i)
    return canon_ints


def run_ilp_loop(
    reads: list[Read],
    params: IsoformsParams,
) -> tuple[CanonIntervals, list[int], list[int], list[int]]:
    """
    Run ILP with the given reads and return the recycling reads and isoform reads.
    If the ILP fails to find an optimal solution, iteratively keep halving the
    number of reads until an optimal solution is found or the number of reads
    drops below the minimum read support.

    Args:
        reads: List of reads
        params: IsoformsParams object

    Returns:
        canon_ints: Canonical intervals of the recycling + isoform reads
        recycling_ridxs: List of recycling read indices
        isoform_ridxs: List of isoform read indices
        unsampled_ridxs: List of unsampled read indices
    """
    N = len(reads)
    reads_idxs = list(range(N))
    canon_ints = canonize_reads(reads)
    if len(reads) < params.min_read_support:
        return canon_ints, reads_idxs, list(), list()
    try:
        status, recycling_bin, isoform_bin = run_ilp(canon_ints, params)
        assert status == pulp.LpStatusOptimal
        return canon_ints, recycling_bin, isoform_bin, list()
    except TimeoutILP:
        pass

    if params.timeout_stategy == timeoutStrat.stop:
        raise TimeoutILP
    elif params.timeout_stategy == timeoutStrat.subsample:
        N2 = min(
            int(params.ilp_params.timeLimit * 30),  # Each 30 reads take ~1sec
            N // 2,
        )
    else:
        raise ValueError(f"Invalid timeout strategy: {params.timeout_stategy}")

    sample_ridxs: list[int] = list()
    unsampled_ridxs: list[int] = list()
    S: set[int] = set(np.random.choice(reads_idxs, size=N2, replace=False))
    for ridx in reads_idxs:
        if ridx in S:
            sample_ridxs.append(ridx)
        else:
            unsampled_ridxs.append(ridx)

    _, sub_recycling_bin, sub_isoform_bin, sub_unsampled_bin = run_ilp_loop(
        [reads[ridx] for ridx in sample_ridxs],
        params,
    )
    recycling_bin = [sample_ridxs[ridx] for ridx in sub_recycling_bin]
    isoform_bin = [sample_ridxs[ridx] for ridx in sub_isoform_bin]
    unsampled_ridxs.extend([sample_ridxs[ridx] for ridx in sub_unsampled_bin])

    return canon_ints, recycling_bin, isoform_bin, unsampled_ridxs


def get_compatible_reads_bins(
    unsampled_reads: list[Read],
    isoform_reads: list[Read],
    canon_ints: CanonIntervals,
    i_vect: list[aln_t],
    slack: int,
):
    """
    Split unsampled reads into in/compatible lists of reads. The method is used when the ILP
    was run with less than all reads. A read is compatible if it shares an exon with the isoform
    and it does not add any exons to the isoform. Additionally, the number

    Args:
        unsampled_reads: List of unsampled reads
        isoform_reads: List of isoform reads
        intervals: Canonical intervals
        isoform_vect: Isoform vector
        slack: Slack

    Returns:
        incompatible_reads: List of incompatible reads
        compatible_reads: List of compatible reads
    """
    isoform_intervals = [
        PairedInterval(
            target=Interval(canon_ints.intervals[j].start, canon_ints.intervals[j].end)
        )
        for j, e in enumerate(i_vect[1:-1])
        if e == aln_t.exon
    ]
    isoform_read = Read(
        idx=-1,
        name="",
        strand="",
        intervals=isoform_intervals,
        qlen=0,
        polyAs=(
            Read.PolyA(overhang=0, length=i_vect[0] == aln_t.exon, slack=0),
            Read.PolyA(overhang=0, length=i_vect[-1] == aln_t.exon, slack=0),
        ),
        cell_types=tuple({ct for read in isoform_reads for ct in read.cell_types}),
    )

    compatible_reads: list[Read] = list()
    incompatible_reads: list[Read] = list()
    for read in unsampled_reads:
        cints = CanonIntervals([isoform_read, read])
        for i in range(10):
            cints.pop(i)
        M = cints.get_matrix()
        i_row = M[0, :]
        r_row = M[1, :]
        is_compat = True
        for j in range(M.shape[1]):
            if i_row[j] == aln_t.exon and r_row[j] != aln_t.exon:
                is_compat = False
                break
            if r_row[j] == aln_t.exon and i_row[j] != aln_t.intron:
                L = cints.intervals[j - 1].end - cints.intervals[j - 1].start
                if L > slack:
                    is_compat = False
                    break
        if is_compat:
            compatible_reads.append(read)
        else:
            incompatible_reads.append(read)
    return incompatible_reads, compatible_reads
