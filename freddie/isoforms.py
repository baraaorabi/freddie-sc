import functools
from typing import Generator
from dataclasses import dataclass

from freddie.ilp import FredILP, IlpParams
from freddie.segment import CanonIntervals, PairedInterval, aln_t
from freddie.split import Interval, Read, Tint

import numpy as np
import pulp


@dataclass
class IsoformsParams:
    max_isoform_count: int = 20
    min_read_support: int = 3
    ilp_params: IlpParams = IlpParams()


@dataclass
class IntervalSupport(Interval):
    support: float = 0.0


@functools.total_ordering
class Isoform:
    """
    Isoform class.

    Attributes:
        tid: Tint ID
        contig: Contig
        reads: List of reads comprising the isoform
        exons: List of genomic intervals (i.e. exons) comprising the isoform with support values.
                The support value is computed by adding the number of bases covered by
                each read in the interval and dividing by the interval length.
        strand: Strand of the isoform if it can be determined from the read polyA tails (i.e. + or -). Otherwise, "."

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
    params: IsoformsParams = IsoformsParams()
) -> list[Isoform]:
    """
    Get isoforms for the given Tint.

    Args:
        tint: Tint
        params: Isoform params dataclass

    Yields:
        Isoform
    """
    assert params.min_read_support > 0
    reads: list[Read] = tint.reads
    isoforms = list()
    for isoform_index in range(params.max_isoform_count):
        recycling_reads, isoform_reads = run_ilp(reads, params)
        if len(isoform_reads) < params.min_read_support:
            break
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
    return isoforms


def run_ilp(reads: list[Read], params: IsoformsParams) -> tuple[list[Read], list[Read]]:
    """
    Run ILP with the given reads and return the recycling reads and isoform reads.
    If the ILP fails to find an optimal solution, iteratively keep halving the
    number of reads until an optimal solution is found or the number of reads
    drops below the minimum read support.

    Args:
        reads: List of reads
        params: Isoforms params dataclass

    Returns:
        recycling_reads: List of recycling reads
        isoform_reads: List of isoform reads
    """
    isoform_reads: list[Read] = list()
    recycling_reads: list[Read] = reads

    sample_reads: list[Read] = reads
    unsampled_reads: list[Read] = list()
    while len(sample_reads) >= params.min_read_support:
        canon_ints = CanonIntervals(sample_reads)
        for i in range(10):
            canon_ints.pop(i)
        ilp: FredILP = FredILP(canon_ints, params.ilp_params)
        ilp.build_model(K=2)
        status, vects, bins = ilp.solve()
        assert len(vects) == len(bins) == 2, f"Expected 2 bins, got {len(bins)}"
        recycling_bin, isoform_bin = bins
        isoform_vect = vects[1]
        # If ILP fails to find optimal solution, retry with half the reads
        if status == pulp.LpStatusOptimal:
            recycling_reads = [sample_reads[idx] for idx in recycling_bin]
            isoform_reads = [sample_reads[idx] for idx in isoform_bin]
            # If the ILP was run with less than all reads, check if there are
            # any reads that are compatible with the isoform
            if len(unsampled_reads) > 0:
                incompatible_reads, compatible_reads = get_compatible_reads_bins(
                    unsampled_reads=unsampled_reads,
                    isoform_reads=isoform_reads,
                    slack=params.ilp_params.max_correction_len,
                    canon_ints=canon_ints,
                    i_vect=isoform_vect,
                )
                isoform_reads.extend(compatible_reads)
                recycling_reads.extend(incompatible_reads)
            reads = recycling_reads
            break
        else:
            sample_ridxs = set(
                np.random.choice(
                    list(range(len(reads))),
                    size=len(sample_reads) // 2,
                    replace=False,
                )
            )
            sample_reads = list()
            unsampled_reads = list()
            for ridx in range(len(reads)):
                if ridx in sample_ridxs:
                    sample_reads.append(reads[ridx])
                else:
                    unsampled_reads.append(reads[ridx])
    return recycling_reads, isoform_reads


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
    print(f"compatible reads: {len(compatible_reads)}")
    return incompatible_reads, compatible_reads
