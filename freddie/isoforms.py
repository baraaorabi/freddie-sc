import functools
from typing import Generator

from freddie.ilp import FredILP
from freddie.segment import canonInts, paired_interval_t
from freddie.split import Read, Tint

import numpy as np
import pulp


@functools.total_ordering
class Isoform:
    def __init__(self, tint: Tint, read_idxs: list[int], isoform_index: int) -> None:
        self.tid = tint.tid
        self.ridxs = read_idxs.copy()
        self.iid = isoform_index
        self.contig = tint.contig
        self.read_count = len(self.ridxs)

        self.intervals: list[tuple[int, int]] = list()
        self.supports: list[float] = list()
        cints = canonInts([tint.reads[ridx] for ridx in self.ridxs])
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
        for ridx in self.ridxs:
            read = tint.reads[ridx]
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
                            f'read_support "{self.read_count}";',
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


def get_compatible_reads(
    ridxs: list[int],
    reads: list[Read],
    isoform: Read,
    slack: int,
) -> list[int]:
    compatible_reads = list()
    for ridx in ridxs:
        cints = canonInts([isoform, reads[ridx]])
        for i in range(10):
            cints.pop(i)
        M = cints.get_matrix()
        is_compat = True
        for j in range(M.shape[1]):
            if M[1, j] == canonInts.aln_t.exon and M[0, j] != canonInts.aln_t.exon:
                is_compat = False
                break
            if M[0, j] == canonInts.aln_t.exon and M[1, j] != canonInts.aln_t.intron:
                L = cints.intervals[j - 1].end - cints.intervals[j - 1].start
                if L > slack:
                    is_compat = False
                    break
        if is_compat:
            compatible_reads.append(ridx)
    return compatible_reads


def get_isoforms(
    tint: Tint,
    ilp_time_limit: int,
    max_correction_len: int,
    max_correction_count: int,
    ilp_solver: str,
    max_isoform_count: int = 20,
) -> Generator[tuple[Isoform, list[str]], None, None]:
    recycling_ridxs: list[int] = list(range(len(tint.reads)))
    sample_ridxs: list[int]
    for isoform_index in range(max_isoform_count):
        sample_size = len(recycling_ridxs)
        while True:
            if sample_size == len(recycling_ridxs):
                sample_ridxs = recycling_ridxs
                unsampled_ridxs = list()
            else:
                sample_ridxs = list(
                    np.random.choice(recycling_ridxs, size=sample_size, replace=False)
                )
                unsampled_ridxs = list(set(recycling_ridxs) - set(sample_ridxs))
            intervals = canonInts([tint.reads[ridx] for ridx in sample_ridxs])
            for i in range(10):
                intervals.pop(i)
            ilp = FredILP(intervals)
            ilp.build_model(
                K=2,
                max_corrections=max_correction_count,
                slack=max_correction_len,
            )
            status, isoforms, bins = ilp.solve(
                solver=ilp_solver,
                timeLimit=ilp_time_limit,
            )
            if status != pulp.LpStatusOptimal:
                sample_size //= 2
                continue
            isoform = isoforms[0]
            isoform_intervals = [
                paired_interval_t(
                    qs=0,
                    qe=0,
                    ts=intervals.intervals[j].start,
                    te=intervals.intervals[j].end,
                )
                for j, e in enumerate(isoform[1:-1])
                if e == canonInts.aln_t.exon
            ]
            isoform_reads = [recycling_ridxs[i] for i in bins[1]]
            isoform_read = Read(
                idx=-1,
                name="",
                strand="",
                intervals=isoform_intervals,
                qlen=0,
                polyAs=(
                    Read.PolyA(
                        overhang=0,
                        length=isoform[0] == canonInts.aln_t.exon,
                        slack=0,
                    ),
                    Read.PolyA(
                        overhang=0,
                        length=isoform[-1] == canonInts.aln_t.exon,
                        slack=0,
                    ),
                ),
                cell_types=tuple(
                    {ct for ridx in isoform_reads for ct in tint.reads[ridx].cell_types}
                ),
            )
            isoform_reads.extend(
                get_compatible_reads(
                    ridxs=unsampled_ridxs,
                    reads=tint.reads,
                    isoform=isoform_read,
                    slack=max_correction_len,
                )
            )
            break
        isoform = Isoform(tint, isoform_reads, isoform_index)
        read_names = [tint.reads[ridx].name for ridx in isoform_reads]
        yield isoform, read_names
        recycling_ridxs = list(set(recycling_ridxs) - set(isoform_reads))
        if len(recycling_ridxs) == 0:
            break
