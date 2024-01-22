import enum
from itertools import groupby
from collections import Counter, defaultdict, deque
from typing import Generator, NamedTuple
import typing

import pysam
from tqdm import tqdm


class CIGAR_OPS_SIMPLE(enum.IntEnum):
    both = 0
    target = 1
    query = 2


paired_interval_t = NamedTuple(
    "paired_interval_t",
    [
        ("qs", int),
        ("qe", int),
        ("ts", int),
        ("te", int),
    ],
)

op_simply: dict[int, CIGAR_OPS_SIMPLE] = {
    pysam.CSOFT_CLIP: CIGAR_OPS_SIMPLE.query,
    pysam.CINS: CIGAR_OPS_SIMPLE.query,
    pysam.CDEL: CIGAR_OPS_SIMPLE.target,
    pysam.CREF_SKIP: CIGAR_OPS_SIMPLE.target,
    pysam.CMATCH: CIGAR_OPS_SIMPLE.both,
    pysam.CDIFF: CIGAR_OPS_SIMPLE.both,
    pysam.CEQUAL: CIGAR_OPS_SIMPLE.both,
}


class Read:
    """
    Read alignment object

    Attributes
    ----------
    idx : int
        Read index (0-based, unique for each read in FredSplit object)
    name : str
        Read name
    strand : str
        Read strand
    qlen : int
        Read length
    intervals : list[paired_interval_t]
        List of intervals of the read
        Each interval is a paired_interval_t namedtuple of (target_start, target_end, query_start, query_end)
        Both target and query intervals are 0-based, start inclusive, and end exlusive
        E.g. the interval 0-10 is 10bp long, and includes the base at index 0 but not the base at index 10.
    polyAs : tuple[polyA_t, polyA_t]
        Tuple of polyA_t namedtuples of (overhang, length, slack) for the left and right polyA tails
        overhang is the number of bases that are after polyA tail
        length is the length of the polyA tail
        slack is the number of bases that are before polyA tail (between the read alignment and the polyA tail)
    cell_types : tuple[str, ...]
        Tuple of cell types that the read belongs to
    """

    PolyA = typing.NamedTuple(
        "PolyA",
        [
            ("overhang", int),
            ("length", int),
            ("slack", int),
        ],
    )

    def __init__(
        self,
        idx: int,
        name: str,
        strand: str,
        intervals: list[paired_interval_t],
        qlen: int,
        polyAs: tuple[PolyA, PolyA],
        cell_types: tuple[str, ...],
    ):
        self.idx = idx
        self.name = name
        self.strand = strand
        self.qlen = qlen
        self.intervals = intervals
        self.polyAs = polyAs
        self.cell_types = cell_types


Tint = typing.NamedTuple(
    "Tint",
    [
        ("contig", str),
        ("tid", int),
        ("reads", list[Read]),
    ],
)


class FredSplit:
    def __init__(
        self,
        cigar_max_del: int,
        polyA_m_score: int,
        polyA_x_score: int,
        polyA_min_len: int,
        contig_min_len: int,
        rname_to_celltypes: None | str = None,
    ) -> None:
        self.cigar_max_del = cigar_max_del
        self.polyA_m_score = polyA_m_score
        self.polyA_x_score = polyA_x_score
        self.polyA_min_len = polyA_min_len
        self.contig_min_len = contig_min_len
        self.read_count = 0
        self.tint_count = 0
        self.qname_to_celltypes: defaultdict[str, tuple[str, ...]] = defaultdict(tuple)
        if rname_to_celltypes is not None:
            for line in open(rname_to_celltypes):
                read_name, cell_types = line.rstrip("\n").split("\t")
                ct_set = set()
                for ct in cell_types.split(","):
                    if ct == "":
                        continue
                    ct_set.add(ct)
                self.qname_to_celltypes[read_name] = tuple(ct_set)

    def get_transcriptional_intervals(
        self,
        reads: list[Read],
        contig: str,
        min_read_support: int,
    ) -> Generator[Tint, None, None]:
        """
        Yields connected transcriptional intervals from a list of reads

        Parameters
        ----------
        reads : list[Read]
            List of reads

        Returns
        -------
        Generator[Tint, None, None]
            Generator of transcriptional intervals
        """
        BasicInterval = NamedTuple(
            "BasicInterval",
            [
                ("start", int),
                ("end", int),
                ("rids", list[int]),
            ],
        )

        bintervals: list[BasicInterval] = list()
        start: int = -1
        end: int = -1
        bint_rids: list[int] = list()
        rid_to_bints: dict[int, list[int]] = {read.idx: list() for read in reads}
        for s, e, rid in sorted(
            (I.ts, I.te, read.idx) for read in reads for I in read.intervals
        ):
            if (start, end) == (-1, -1):
                start, end = s, e
            if s > end:
                bintervals.append(
                    BasicInterval(
                        start=start,
                        end=end,
                        rids=bint_rids,
                    )
                )
                start = s
                end = e
                bint_rids = list()
            assert start <= s
            end = max(end, e)
            bint_rids.append(rid)
            rid_to_bints[rid].append(len(bintervals))
        if (start, end) == (-1, -1):
            return
        bintervals.append(
            BasicInterval(
                start=start,
                end=end,
                rids=bint_rids,
            )
        )

        enqueued = [False for _ in bintervals]
        bint_idx: int
        # Breadth-first search
        for bint_idx in range(len(bintervals)):
            if enqueued[bint_idx]:
                continue
            group: list[int] = list()
            queue: deque[int] = deque()
            queue.append(bint_idx)
            enqueued[bint_idx] = True
            while len(queue) > 0:
                bint_idx = queue.pop()
                group.append(bint_idx)
                for rid in bintervals[bint_idx].rids:
                    for bint_idx in rid_to_bints[rid]:
                        if not enqueued[bint_idx]:
                            enqueued[bint_idx] = True
                            queue.append(bint_idx)
            tint_rids: set[int] = set()
            group_intervals: list[tuple[int, int]] = list()
            for bint_idx in group:
                tint_rids.update(bintervals[bint_idx].rids)
                group_intervals.append(
                    (bintervals[bint_idx].start, bintervals[bint_idx].end)
                )
            tint = Tint(
                contig=contig,
                tid=self.tint_count,
                reads=[read for read in reads if read.idx in tint_rids],
            )
            assert len(tint.reads) == len(tint_rids)
            if len(tint.reads) < min_read_support:
                continue
            yield tint
            self.tint_count += 1
        assert all(enqueued)

    def find_longest_polyA(self, seq: str) -> tuple[int, int, int]:
        """
        Finds the longest polyA in the sequence.

        Parameters
        ----------
        seq : str
            Sequence

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
                scores = [self.polyA_m_score]
            else:
                scores = [0]
            for m in (
                self.polyA_m_score if c == char else self.polyA_x_score for c in seq[1:]
            ):
                scores.append(max(0, scores[-1] + m))

            for is_positive, g in groupby(enumerate(scores), lambda x: x[1] > 0):
                if not is_positive:
                    continue
                idxs, cur_scores = list(zip(*g))
                _, last_idx = max(zip(cur_scores, idxs))
                last_idx += 1
                first_idx = idxs[0]
                length = last_idx - first_idx
                if length > max_length and length >= self.polyA_min_len:
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

    def get_intervals(self, aln) -> list[paired_interval_t]:
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
        """
        cigartuples: list[tuple[int, int]] = list(aln.cigartuples)
        cigar = FredSplit.canonize_cigar(cigartuples)
        qstart = 0
        qlen = 0
        for op, l in cigar:
            if op in [CIGAR_OPS_SIMPLE.query, CIGAR_OPS_SIMPLE.both]:
                qlen += l
        assert qlen == len(aln.query_sequence)
        p_interval_wc = NamedTuple(
            "paired_interval_with_cigar",
            [
                ("qs", int),
                ("qe", int),
                ("ts", int),
                ("te", int),
                ("cigar", list[tuple[CIGAR_OPS_SIMPLE, int]]),
            ],
        )
        intervals: list[
            p_interval_wc
        ] = list()  # list of exonic intervals of the alignment

        qstart: int = 0  # current interval's start on query
        qend: int = 0  # current interval's end on query
        tstart: int = aln.reference_start  # aln.reference_start is 0-indexed
        tend: int = tstart  # current interval's end on target
        for is_splice, g in groupby(
            cigar,
            key=lambda x: x[0] == CIGAR_OPS_SIMPLE.target and x[1] > self.cigar_max_del,
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
                    p_interval_wc(
                        ts=tstart,
                        te=tend,
                        qs=qstart,
                        qe=qend,
                        cigar=cur_cigar,
                    )
                )
            qstart = qend
            tstart = tend
        final_intervals: list[paired_interval_t] = list()
        for interval in intervals:
            qs = interval.qs
            qe = interval.qe
            ts = interval.ts
            te = interval.te
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
                paired_interval_t(
                    qs=qs,
                    qe=qe,
                    ts=ts,
                    te=te,
                )
            )
        return final_intervals

    def overlapping_reads(self, sam, contig: str) -> Generator[list[Read], None, None]:
        """
        Generates lists of reads with overlapping alignment (exonically
        or intronically) on the contig

        Parameters
        ----------
        sam_path : str
            Path to SAM/BAM file
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
            intervals = self.get_intervals(aln)
            lpA_a, lpA_b, lpA_c = self.find_longest_polyA(seq[: intervals[0].qs])
            rpA_a, rpA_b, rpA_c = self.find_longest_polyA(seq[intervals[-1].qe :])
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
                idx=self.read_count,
                name=qname,
                strand="-" if aln.is_reverse else "+",
                intervals=intervals,
                qlen=len(seq),
                polyAs=polyAs,
                cell_types=self.qname_to_celltypes[qname],
            )
            self.read_count += 1
            s = intervals[0].ts
            e = intervals[-1].te
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
        sam_path: str,
        min_read_support: int = 3,
    ) -> Generator[Tint, None, None]:
        sam = pysam.AlignmentFile(sam_path, "rb")
        contigs: list[str] = [
            x["SN"] for x in sam.header["SQ"] if x["LN"] > self.contig_min_len
        ]
        for contig in contigs:
            for reads in self.overlapping_reads(sam, contig):
                for tint in tqdm(
                    self.get_transcriptional_intervals(reads, contig, min_read_support),
                    desc=f"Generating tints of contig {contig}",
                ):
                    yield tint
