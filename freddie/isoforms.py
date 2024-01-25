import functools
from typing import Generator, Iterable, NamedTuple

from freddie.ilp import FredILP
from freddie.segment import canonInts, paired_interval_t
from freddie.split import Read, Tint

import numpy as np
import pulp

clustering_settings_t = NamedTuple(
    "clustering_settings_t",
    [
        ("ilp_time_limit", int),
        ("max_correction_len", int),
        ("max_correction_count", int),
        ("ilp_solver", str),
        ("max_isoform_count", int),
        ("min_read_support", int),
    ],
)
aln_t = canonInts.aln_t


@functools.total_ordering
class Isoform:
    """
    Isoform class.

    Attributes:
        tid: Tint ID
        contig: Contig
        reads: List of reads comprising the isoform
        intervals: List of genomic intervals (i.e. exons) comprising the isoform
        supports: List of support values for each interval.
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

        self.intervals: list[tuple[int, int]] = list()
        self.supports: list[float] = list()
        cints = canonInts(reads)
        for i in range(10):
            cints.pop(i)
        intervals: list[tuple[int, int]] = list()
        supports: list[float] = list()
        for cur_interval in cints.intervals:
            e_cnt = len(cur_interval.exonic_ridxs())
            i_cnt = len(cur_interval.intronic_ridxs())
            if e_cnt > i_cnt:
                intervals.append((cur_interval.start, cur_interval.end))
                supports.append(e_cnt)
        intervals.sort()
        for cur_interval, cur_support in zip(intervals, supports):
            if len(self.intervals) == 0:
                self.intervals.append(cur_interval)
                self.supports.append(cur_support)
                continue
            if self.intervals[-1][1] < cur_interval[0]:
                self.intervals.append(cur_interval)
                self.supports.append(cur_support)
                continue
            pre_interval = self.intervals[-1]
            pre_support = self.supports[-1]
            self.intervals[-1] = (pre_interval[0], cur_interval[1])
            l1 = pre_interval[1] - pre_interval[0]
            l2 = cur_interval[1] - cur_interval[0]
            self.supports[-1] = float((l1 * pre_support + l2 * cur_support) / (l1 + l2))

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
        return self.contig == __value.contig and self.intervals == __value.intervals

    def __lt__(self, __value: object) -> bool:
        if not isinstance(__value, Isoform):
            return NotImplemented
        if self.contig != __value.contig:
            # If contig is a number, sort by number
            if self.contig.isnumeric() and __value.contig.isnumeric():
                return int(self.contig) < int(__value.contig)
            return self.contig < __value.contig
        return self.intervals < __value.intervals

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
                    f"{self.intervals[0][0] + 1}",
                    f"{self.intervals[-1][1]}",
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
        for idx, (interval, support) in enumerate(
            zip(self.intervals, self.supports), start=1
        ):
            gtf_records.append(
                "\t".join(
                    [
                        self.contig,
                        "freddie",
                        "exon",
                        f"{interval[0] + 1}",
                        f"{interval[1]}",
                        ".",
                        self.strand,
                        ".",
                        " ".join(
                            [
                                f'gene_id "{gene_id}";',
                                f'transcript_id "{isoform_id}";',
                                f'read_support "{support:.2f}";',
                                f'exon_number "{idx}";',
                            ]
                        ),
                    ]
                )
            )
        return "\n".join(gtf_records)


def get_isoforms(
    tint: Tint,
    ilp_settings: clustering_settings_t,
) -> Generator[Isoform, None, None]:
    """
    Get isoforms for the given Tint.

    Args:
        tint: Tint
        ilp_settings: ILP settings namedtuple

    Yields:
        Isoform
    """
    assert ilp_settings.min_read_support > 0
    reads: list[Read] = tint.reads
    for isoform_index in range(ilp_settings.max_isoform_count):
        recycling_reads, isoform_reads = run_ilp(reads, ilp_settings)
        if len(isoform_reads) < ilp_settings.min_read_support:
            break
        isoform = Isoform(
            tid=tint.tid,
            contig=tint.contig,
            reads=isoform_reads,
            isoform_index=isoform_index,
        )
        yield isoform
        # Remove the reads that were used to construct the isoform
        reads = recycling_reads
        if len(reads) < ilp_settings.min_read_support:
            break


def run_ilp(
    reads: list[Read],
    ilp_settings: clustering_settings_t,
) -> tuple[list[Read], list[Read]]:
    """
    Run ILP with the given reads and return the recycling reads and isoform reads.
    If the ILP fails to find an optimal solution, iteratively keep halving the
    number of reads until an optimal solution is found or the number of reads
    drops below the minimum read support.

    Args:
        reads: List of reads
        ilp_settings: ILP settings namedtuple

    Returns:
        recycling_reads: List of recycling reads
        isoform_reads: List of isoform reads
    """
    isoform_reads: list[Read] = list()
    recycling_reads: list[Read] = reads

    sample_reads: list[Read] = reads
    unsampled_reads: list[Read] = list()
    while len(sample_reads) >= ilp_settings.min_read_support:
        intervals = canonInts(sample_reads)
        for i in range(10):
            intervals.pop(i)
        ilp: FredILP = FredILP(intervals)
        ilp.build_model(
            K=2,
            max_corrections=ilp_settings.max_correction_count,
            slack=ilp_settings.max_correction_len,
        )
        status, vects, bins = ilp.solve(
            solver=ilp_settings.ilp_solver,
            timeLimit=ilp_settings.ilp_time_limit,
        )
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
                    slack=ilp_settings.max_correction_len,
                    intervals=intervals,
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
    intervals: canonInts,
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
        paired_interval_t(
            qs=0,
            qe=0,
            ts=intervals.intervals[j].start,
            te=intervals.intervals[j].end,
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
        cints = canonInts([isoform_read, read])
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
