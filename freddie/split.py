import enum
from functools import total_ordering
from itertools import groupby
from collections import Counter, defaultdict, deque
from typing import Generator, Iterable
from dataclasses import dataclass, field

import pysam
from tqdm import tqdm


class CIGAR_OPS_SIMPLE(enum.IntEnum):
    both = 0
    target = 1
    query = 2


@total_ordering
@dataclass
class Interval:
    start: int = 0
    end: int = 0

    def __post_init__(self):
        assert 0 <= self.start <= self.end

    def __eq__(self, other):
        return (self.start, self.end) == (other.start, other.end)

    def __lt__(self, other):
        return (self.start, self.end) < (other.start, other.end)

    def __le__(self, other):
        return (self.start, self.end) <= (other.start, other.end)

    def __gt__(self, other):
        return (self.start, self.end) > (other.start, other.end)

    def __ge__(self, other):
        return (self.start, self.end) >= (other.start, other.end)

    def __len__(self):
        return self.end - self.start


@dataclass
class PairedInterval:
    query: Interval = field(default_factory=Interval)
    target: Interval = field(default_factory=Interval)


@dataclass
class PairedIntervalCigar(PairedInterval):
    cigar: list[tuple[CIGAR_OPS_SIMPLE, int]] = field(default_factory=list)


@dataclass
class IntervalRIDs(Interval):
    rids: list[int] = field(default_factory=list)


op_simply: dict[int, CIGAR_OPS_SIMPLE] = {
    pysam.CSOFT_CLIP: CIGAR_OPS_SIMPLE.query,
    pysam.CINS: CIGAR_OPS_SIMPLE.query,
    pysam.CDEL: CIGAR_OPS_SIMPLE.target,
    pysam.CREF_SKIP: CIGAR_OPS_SIMPLE.target,
    pysam.CMATCH: CIGAR_OPS_SIMPLE.both,
    pysam.CDIFF: CIGAR_OPS_SIMPLE.both,
    pysam.CEQUAL: CIGAR_OPS_SIMPLE.both,
}


@dataclass
class Read:
    idx: int
    name: str
    strand: str
    intervals: list[PairedInterval]
    qlen: int
    polyAs: tuple["Read.PolyA", "Read.PolyA"]
    cell_types: tuple[str, ...]

    @dataclass
    class PolyA:
        overhang: int = 0
        length: int = 0
        slack: int = 0

        def __post_init__(self):
            assert self.overhang >= 0
            assert self.length >= 0
            assert self.slack >= 0


@dataclass
class Tint:
    contig: str
    tid: int
    reads: list[Read] = field(default_factory=list)


@dataclass
class FredSplitParams:
    cigar_max_del: int = 20
    polyA_m_score: int = 1
    polyA_x_score: int = -2
    polyA_min_len: int = 10
    contig_min_len: int = 1_000_000


class FredSplit:
    """
    Class to split a SAM/BAM file into transcriptional intervals (tint's).

    Parameters
    ----------
    bam_path : str
        Path to BAM file. Must be sorted and in compressed BAM format.
    contigs : Iterable[str] | None
        Iterable of contigs to process. If None, all contigs will be processed.
        This parameter does not override the contig_min_len parameter in the FredSplitParams object.
    params : FredSplitParams
        Parameters for FredSplit
    rname_to_celltypes : str | None
        Path to a file that maps read names to cell types. Each line should be in the format:
        read_name\tcell_types
        where cell_types is a comma-separated list of 0 or more cell types.
        If rname_to_celltypes=None, all reads will be assigned to an empty tuple of cell types.
    """

    def __init__(
        self,
        bam_path: str,
        contigs: None | Iterable[str] = None,
        params: FredSplitParams = FredSplitParams(),
        rname_to_celltypes: None | str = None,
    ) -> None:
        self.sam = pysam.AlignmentFile(bam_path, "rb")
        self.contigs: list[FredSplit.contig] = list()
        self.params = params
        self.qname_to_celltypes: defaultdict[str, tuple[str, ...]] = defaultdict(tuple)
        # Check if SAM is sorted and indexed
        for k, v in self.sam.header.to_dict()["HD"].items():
            if k == "SO":
                field = v.split(",")
                assert (
                    field[0] == "coordinate"
                ), f"{bam_path} SAM file must be sorted by coordinate"
        assert self.sam.check_index(), f"{bam_path} SAM file must be indexed"
        # Build list of contigs
        if contigs is None:
            contigs = set(self.sam.references)
        else:
            contigs = set(contigs)
        for x in self.sam.header.to_dict()["SQ"]:
            if x["SN"] not in contigs:
                continue
            if x["LN"] < self.params.contig_min_len:
                continue
            self.contigs.append(FredSplit.contig(name=x["SN"], length=x["LN"]))
        # Build dictionary of read names to cell types
        if rname_to_celltypes is not None:
            for line in open(rname_to_celltypes):
                read_name, cell_types = line.rstrip("\n").split("\t")
                ct_set = set()
                for ct in cell_types.split(","):
                    if ct == "":
                        continue
                    ct_set.add(ct)
                self.qname_to_celltypes[read_name] = tuple(ct_set)

    @dataclass
    class contig:
        name: str
        length: int

        def __post_init__(self):
            assert self.length > 0

    @staticmethod
    def get_tints(reads: list[Read]) -> Generator[list[Read], None, None]:
        """
        Yields connected lists of reads that are connected by their exonic intervals.
        The union of the output lists is equal to the input list.
        No read is repeated in the output lists.

        Parameters
        ----------
        reads : list[Read]
            List of reads

        Returns
        -------
        Generator[list[Read], None, None]
            Generator of transcriptional intervals
        """

        intervals: list[IntervalRIDs] = list()
        start: int = -1
        end: int = -1
        cur_rids: list[int] = list()
        rid_to_int_idx: dict[int, list[int]] = {read.idx: list() for read in reads}
        for s, e, rid in sorted(
            (I.target.start, I.target.end, read.idx)
            for read in reads
            for I in read.intervals
        ):
            if (start, end) == (-1, -1):
                start, end = s, e
            if s > end:
                intervals.append(IntervalRIDs(start, end, cur_rids))
                start = s
                end = e
                cur_rids = list()
            assert start <= s
            end = max(end, e)
            cur_rids.append(rid)
            rid_to_int_idx[rid].append(len(intervals))
        if (start, end) == (-1, -1):
            return
        intervals.append(IntervalRIDs(start, end, cur_rids))

        enqueued = [False for _ in intervals]
        # Breadth-first search
        for int_idx in range(len(intervals)):
            if enqueued[int_idx]:
                continue
            group: list[int] = list()
            queue: deque[int] = deque()
            queue.append(int_idx)
            enqueued[int_idx] = True
            while len(queue) > 0:
                int_idx = queue.pop()
                group.append(int_idx)
                for rid in intervals[int_idx].rids:
                    for int_idx in rid_to_int_idx[rid]:
                        if not enqueued[int_idx]:
                            enqueued[int_idx] = True
                            queue.append(int_idx)
            tint_rids: set[int] = set()
            group_intervals: list[tuple[int, int]] = list()
            for int_idx in group:
                tint_rids.update(intervals[int_idx].rids)
                group_intervals.append(
                    (
                        intervals[int_idx].start,
                        intervals[int_idx].end,
                    )
                )
            tint_reads = [read for read in reads if read.idx in tint_rids]
            yield tint_reads
        assert all(enqueued)

    @staticmethod
    def find_longest_polyA(
        seq: str,
        m_score: int,
        x_score: int,
        min_len: int,
    ) -> tuple[int, int, int]:
        """
        Finds the longest polyA in the sequence.

        Parameters
        ----------
        seq : str
            Sequence
        m_score : int
            Match score
        x_score : int
            Mismatch score
        min_len : int
            Minimum length of polyA (reports 0 length if the polyA is shorter than this value)
        Returns
        -------
        tuple[int, int, int]
            Tuple of (before length, polyA length, after length) that sums up to
            the length of the sequence
        """
        result = (len(seq), 0, 0)
        if len(seq) == 0:
            return result
        max_length = 0
        for char in "AT":
            if seq[0] == char:
                scores = [m_score]
            else:
                scores = [0]
            for m in (m_score if c == char else x_score for c in seq[1:]):
                scores.append(max(0, scores[-1] + m))

            for is_positive, g in groupby(enumerate(scores), lambda x: x[1] > 0):
                if not is_positive:
                    continue
                idxs, cur_scores = list(zip(*g))
                _, last_idx = max(zip(cur_scores, idxs))
                last_idx += 1
                first_idx = idxs[0]
                length = last_idx - first_idx
                if length > max_length and length >= min_len:
                    max_length = length
                    result = (first_idx, length, len(seq) - last_idx)
        return result

    @staticmethod
    def canonize_cigar(
        cigartuples: list[tuple[int, int]]
    ) -> list[tuple[CIGAR_OPS_SIMPLE, int]]:
        """
        Canonizes CIGAR tuples by combining operations of the same type and sorting them.
        - all operations are simplified to target-consuming, query-consuming,
        or both-consuming operations.
        - operations between any group (of one or more) both-consuming operations are sorted by their type
        - operations of the same type within each group are sum-combined into a single operation of the same type
        For example, the CIGAR (10M, 20I, 5D, 5I, 5D, 10M) is canonized to (10M, 30I, 10D, 10M).
        These un-canonical CIGARS are rare but they are produced sometimes by Minimap2:
        (https://github.com/lh3/minimap2/issues/1118).

        Parameters
        ----------
        cigartuples : list[tuple[int, int]]
            List of CIGAR tuples (operation, length) as produced by pysam AlignedSegment.cigartuples
        """
        simple_cigartuples = [(op_simply[op], l) for op, l in cigartuples]
        canonized_cigar: list[tuple[CIGAR_OPS_SIMPLE, int]] = list()
        for _, g in groupby(
            simple_cigartuples, key=lambda x: x[0] == CIGAR_OPS_SIMPLE.both
        ):
            C: Counter[CIGAR_OPS_SIMPLE] = Counter()
            for op, l in g:
                C[op] += l
            for op, l in sorted(C.items()):
                if l > 0:
                    canonized_cigar.append((op, l))
        return canonized_cigar

    @staticmethod
    def get_intervals(
        aln: pysam.AlignedSegment,
        cigar_max_del: int,
    ) -> list[PairedInterval]:
        """
        Returns a list of intervals of the alignment.
        Each interval is a tuple of (target_start, target_end, query_start, query_end)
        Both target and query intervals are 0-based, start inclusive, and end exlusive
        E.g. the interval 0-10 is 10bp long and includes the base at index 0 but not the base at index 10.
        Note: the CIGARs are first canonized using canonize_cigar() function.

        Parameters
        ----------
        aln : pysam.AlignedSegment
            pysam AlignedSegment object
        cigar_max_del : int
            Maximum deletion (op='D') length to not be considered as a splice junction.
            Any deletion longer than this value will be considered as a splice junction.

        Returns
        -------
        list[PairedInterval]
            List of exonic intervals of the alignment
        """
        assert (
            aln.cigartuples is not None
        ), f"CIGAR tuples are None in {aln.query_name} SAM record."
        assert (
            aln.query_sequence is not None
        ), f"Query sequence is None in {aln.query_name} SAM record."
        cigar = FredSplit.canonize_cigar(aln.cigartuples)
        qstart = 0
        qlen = 0
        for op, l in cigar:
            if op in [CIGAR_OPS_SIMPLE.query, CIGAR_OPS_SIMPLE.both]:
                qlen += l
        assert qlen == len(aln.query_sequence)

        # list of exonic intervals of the alignment
        intervals: list[PairedIntervalCigar] = list()

        qstart: int = 0  # current interval's start on query
        qend: int = 0  # current interval's end on query
        tstart: int = aln.reference_start  # aln.reference_start is 0-indexed
        tend: int = tstart  # current interval's end on target
        for is_splice, g in groupby(
            cigar,
            key=lambda x: x[0] == CIGAR_OPS_SIMPLE.target and x[1] > cigar_max_del,
        ):
            cur_cigar = list(g)
            for op, l in cur_cigar:
                if op == CIGAR_OPS_SIMPLE.query:
                    qend += l
                elif op == CIGAR_OPS_SIMPLE.target:
                    tend += l
                elif op == CIGAR_OPS_SIMPLE.both:
                    qend += l
                    tend += l
            if not is_splice:
                intervals.append(
                    PairedIntervalCigar(
                        query=Interval(qstart, qend),
                        target=Interval(tstart, tend),
                        cigar=cur_cigar,
                    )
                )
            qstart = qend
            tstart = tend
        final_intervals: list[PairedInterval] = list()
        for interval in intervals:
            qs = interval.query.start
            qe = interval.query.end
            ts = interval.target.start
            te = interval.target.end
            cigar = interval.cigar
            assert qe - qs == (
                S := sum(
                    l
                    for op, l in cigar
                    if op in [CIGAR_OPS_SIMPLE.query, CIGAR_OPS_SIMPLE.both]
                )
            ), (qe - qs, S)
            assert te - ts == (
                S := sum(
                    l
                    for op, l in cigar
                    if op in [CIGAR_OPS_SIMPLE.target, CIGAR_OPS_SIMPLE.both]
                )
            ), (qe - qs, S)
            for op, l in cigar:
                if op == CIGAR_OPS_SIMPLE.both:
                    break
                if op == CIGAR_OPS_SIMPLE.query:
                    qs += l
                elif op == CIGAR_OPS_SIMPLE.target:
                    ts += l
            for op, l in cigar[::-1]:
                if op == CIGAR_OPS_SIMPLE.both:
                    break
                if op == CIGAR_OPS_SIMPLE.query:
                    qe -= l
                elif op == CIGAR_OPS_SIMPLE.target:
                    te -= l
            final_intervals.append(
                PairedInterval(query=Interval(qs, qe), target=Interval(ts, te))
            )
        return final_intervals

    def overlapping_reads(
        self,
        sam: pysam.AlignmentFile,
        contig: str,
        read_starting_index: int = 0,
    ) -> Generator[list[Read], None, None]:
        """
        Generates lists of reads with overlapping alignment (exonically
        or intronically) on the contig

        Parameters
        ----------
        sam : pysam.AlignmentFile
            pysam AlignmentFile object
        contig : str
            Contig name to extract reads from
        read_starting_index : int
            Starting index of the reads used for giving incremental IDs to the reads
        Yields
        ------
        Generator[list[Read], None, None]
            Each generation is a dictionary of reads where any two reads that
            overlap will be in the same dictionary
        """
        reads: list[Read] = list()
        start: int = -1
        end: int = -1
        for aln in sam.fetch(contig=contig):
            if (
                aln.is_unmapped
                or aln.is_supplementary
                or aln.is_secondary
                or aln.reference_name == None
            ):
                continue
            qname: str = str(aln.query_name)
            seq = str(aln.query_sequence)
            intervals = self.get_intervals(aln, self.params.cigar_max_del)
            lpA_a, lpA_b, lpA_c = self.find_longest_polyA(
                seq=seq[: intervals[0].query.start],
                m_score=self.params.polyA_m_score,
                x_score=self.params.polyA_x_score,
                min_len=self.params.polyA_min_len,
            )
            rpA_a, rpA_b, rpA_c = self.find_longest_polyA(
                seq=seq[intervals[-1].query.end :],
                m_score=self.params.polyA_m_score,
                x_score=self.params.polyA_x_score,
                min_len=self.params.polyA_min_len,
            )
            if lpA_b > 0 and rpA_b > 0:
                polyAs = (Read.PolyA(), Read.PolyA())
            else:
                polyAs = (
                    Read.PolyA(
                        overhang=lpA_a,
                        length=lpA_b,
                        slack=lpA_c,
                    ),
                    Read.PolyA(
                        slack=rpA_a,
                        length=rpA_b,
                        overhang=rpA_c,
                    ),
                )
            read = Read(
                idx=len(reads) + read_starting_index,
                name=qname,
                strand="-" if aln.is_reverse else "+",
                intervals=intervals,
                qlen=len(seq),
                polyAs=polyAs,
                cell_types=self.qname_to_celltypes[qname],
            )
            s = intervals[0].target.start
            e = intervals[-1].target.end
            if (start, end) == (-1, -1):
                start, end = s, e
            if s > end:
                yield reads
                reads = list()
                end = e
            end = max(end, e)
            reads.append(read)
        if len(reads) > 0:
            yield reads

    def generate_all_tints(
        self,
        pbar_tint: tqdm | None = None,
        pbar_reads: tqdm | None = None,
    ) -> Generator[Tint, None, None]:
        """
        Generates all transcriptional intervals from a SAM/BAM file

        Parameters
        ----------
        pbar_tint : tqdm | None
            Progress bar for transcriptional intervals.
            The total will be updated by this method while
            the progress value should be updated by the caller.
        pbar_reads : tqdm | None
            Progress bar for reads.
            The total will be updated by this method while
            the progress value should be updated by the caller.
        """
        pbar_genome = tqdm(
            desc="[freddie] Genome-wide progress",
            total=sum(contig.length for contig in self.contigs),
            unit="bp",
            unit_scale=True,
            leave=True,
        )
        tint_count: int = 0
        read_index: int = 0
        for contig in self.contigs:
            pbar_genome.set_description(
                f"[freddie] Genome-wide progress (congig: {contig})"
            )
            last_pos = 0
            for reads in self.overlapping_reads(self.sam, contig.name, read_index):
                read_index += len(reads)
                cur_pos = reads[-1].intervals[-1].target.end
                pbar_genome.update(cur_pos - last_pos)
                last_pos = cur_pos
                for tint_reads in self.get_tints(reads):
                    tint = Tint(
                        contig=contig.name,
                        reads=tint_reads,
                        tid=tint_count,
                    )
                    tint_count += 1
                    if pbar_reads is not None:
                        pbar_reads.total += len(tint_reads)
                        pbar_reads.refresh()
                    if pbar_tint is not None:
                        pbar_tint.total += 1
                        pbar_tint.refresh()
                    yield tint
        pbar_genome.close()
