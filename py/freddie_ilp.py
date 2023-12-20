from collections import defaultdict
from itertools import groupby
import typing

import pulp
from freddie_segment import canonInts

aln_t = canonInts.aln_t

ALN_T_MAP = {
    aln_t.exon: aln_t.exon,
    aln_t.polyA: aln_t.exon,
    aln_t.intron: aln_t.intron,
    aln_t.unaln: aln_t.intron,
}


class FredILP:
    def __init__(self, cints: canonInts):
        data_to_ridxs: defaultdict[
            tuple[tuple[aln_t, ...], tuple[str, ...]],
            list[int],
        ] = defaultdict(list)
        self.interval_lengths = (
            (10,)
            + tuple(interval.end - interval.start for interval in cints.intervals)
            + (10,)
        )
        for idx, row in enumerate(cints.get_matrix()):
            cell_types = cints.reads[idx].cell_types
            first = len(row) - 1
            last = 0
            for j, aln_type in enumerate(row):
                if aln_type in [
                    aln_t.exon,
                    aln_t.polyA,
                ]:
                    first = min(first, j)
                    last = max(last, j)
            assert first <= last
            key = (
                tuple(aln_t.unaln for _ in range(first))
                + tuple(ALN_T_MAP[i] for i in row[first : last + 1])
                + tuple(aln_t.unaln for _ in range(last + 1, len(row)))
            ), cell_types
            data_to_ridxs[key].append(idx)
        keys: tuple[tuple[tuple[aln_t, ...], tuple[str, ...]], ...]
        vals: tuple[list[int], ...]
        keys, vals = zip(*data_to_ridxs.items())
        self.rows = tuple(r for r, _ in keys)
        self.cell_types = tuple(cts for _, cts in keys)
        self.ridxs = tuple(tuple(v) for v in vals)
        for row in self.rows:
            assert len(row) == len(self.interval_lengths)
        del data_to_ridxs, keys, vals

    def get_introns(self, i) -> typing.Generator[tuple[int, int], None, None]:
        for key, group in groupby(enumerate(self.rows[i]), key=lambda x: x[1]):
            if key == aln_t.intron:
                j1 = next(group)[0]
                j2 = j1
                for j2, _ in group:
                    pass
                yield j1, j2

    def build_model(
        self,
        K: int = 2,
        slack: int = 20,
        max_corrections: int = 3,
    ) -> None:
        assert K >= 2
        self.K: int = K
        M: int = len(self.interval_lengths)
        N: int = len(self.rows)
        MAX_ISOFORM_LG: int = sum(self.interval_lengths)

        # ILP model ------------------------------------------------------
        self.model = pulp.LpProblem("isoforms_v9_20231014", pulp.LpMinimize)
        # Decision variables
        # R2I[i, k] = 1 if read i assigned to isoform k
        self.R2I: dict[tuple[int, int], pulp.LpVariable] = dict()
        for i in range(N):
            for k in range(K):
                self.R2I[i, k] = pulp.LpVariable(
                    name=f"R2I_{i},{k}",
                    cat=pulp.LpBinary,
                )
            # Constraint enforcing that each read is assigned to exactly one isoform
            self.model += pulp.lpSum(self.R2I[i, k] for k in range(0, K)) == 1

        # Implied variable: canonical exons presence in isoforms
        # E2I[j, k]     = 1 if canonical exon j is in isoform k,
        #                 i.e., over all reads i, E2I >= E2IR[j, k, i]
        # E2IR[j, k, i] = 1 if read i assigned to isoform k AND exon j covered by read i,
        #                 i.e., R2I[i,k] AND (rows[i][j] == exon)
        self.E2I: dict[tuple[int, int], pulp.LpVariable] = dict()
        E2IR: dict[tuple[int, int, int], pulp.LpVariable] = dict()
        for j in range(M):
            # No exon is assignd to the garbage isoform
            self.E2I[j, 0] = pulp.LpVariable(
                name=f"E2I_{j},0",
                cat=pulp.LpBinary,
            )
            self.model += self.E2I[(j, 0)] == 0
            # We start assigning exons from the first isoform
            for k in range(1, K):
                self.E2I[j, k] = pulp.LpVariable(
                    name=f"E2I_{j},{k}",
                    cat=pulp.LpBinary,
                )
                for i in range(N):
                    E2IR[j, k, i] = pulp.LpVariable(
                        name=f"E2IR_{j},{k},{i}",
                        cat=pulp.LpBinary,
                    )
                    self.model += E2IR[j, k, i] == self.R2I[i, k] * (
                        self.rows[i][j] == aln_t.exon
                    )
                # E2I[j, k] = max over  all reads i of E2IR[j, k, i]
                for i in range(N):
                    self.model += self.E2I[j, k] >= E2IR[j, k, i]
                self.model += self.E2I[j, k] <= pulp.lpSum(
                    E2IR[j, k, i] for i in range(N)
                )

        # Implied variable: interval is covered (intronically or exonically) by isoform
        # C2IR[j, k, i]  = 1 if read i assigned to isoform k AND interval j covered by read i,
        #                  i.e., R2I[i,k] AND (rows[i][j] != unaln)
        # C2I[j, k]      = 1 if interval j is covered by isoform k,
        #                  i.e., over all reads i, C2I >= C2IR[j, k, i]
        # CHANGE2I[j, k] = C2I[j, k] XOR C2I[j + 1, k]
        #                  Per isoform, the sum over C2I vals should be exactly 2
        C2I: dict[tuple[int, int], pulp.LpVariable] = dict()
        C2IR: dict[tuple[int, int, int], pulp.LpVariable] = dict()
        CHANGE2I: dict[tuple[int, int], pulp.LpVariable] = dict()
        # We start assigning intervals from the first isoform
        for k in range(1, K):
            for j in [-1, M]:
                C2I[j, k] = pulp.LpVariable(
                    name=f"C2I_{'n' if j < 0 else ''}{abs(j)},{k}",
                    cat=pulp.LpBinary,
                )
                self.model += C2I[j, k] == 0
            for j in range(M):
                C2I[j, k] = pulp.LpVariable(
                    name=f"C2I_{j},{k}",
                    cat=pulp.LpBinary,
                )
                for i in range(N):
                    C2IR[j, k, i] = pulp.LpVariable(
                        name=f"C2IR_{j},{k},{i}",
                        cat=pulp.LpBinary,
                    )
                    self.model += C2IR[j, k, i] == self.R2I[i, k] * (
                        self.rows[i][j] != aln_t.unaln
                    )
                # C2I[j, k] = max over of C2IR[j, k, i] for each read i
                for i in range(N):
                    self.model += C2I[j, k] >= C2IR[j, k, i]
                self.model += C2I[j, k] <= pulp.lpSum(C2IR[j, k, i] for i in range(N))
            for j in range(-1, M):
                CHANGE2I[j, k] = pulp.LpVariable(
                    name=f"CHANGE2I_{'n' if j < 0 else ''}{abs(j)},{k}",
                    cat=pulp.LpBinary,
                )
                x = C2I[j, k]
                y = C2I[j + 1, k]
                self.model += CHANGE2I[j, k] <= x + y
                self.model += CHANGE2I[j, k] >= x - y
                self.model += CHANGE2I[j, k] >= y - x
                self.model += CHANGE2I[j, k] <= 2 - x - y
            self.model += pulp.lpSum(CHANGE2I[j, k] for j in range(-1, M)) == 2

        # There can be only one (or zero) polyA interval per isoform
        for k in range(1, K):
            self.model += self.E2I[0, k] + self.E2I[M - 1, k] <= 1

        # Adding constraints for unaligned gaps
        # If read i is assigned to isoform k, and read i contains intron j1 <-> j2, and
        # the sum of the lengths of exons in isoform k between exons j1 and j2 is L
        # then L <= slack
        # GAPI[j1, j2, k] = sum of the length of the exons between exons j1 and j2 (inclusively) in isoform k
        GAPI: dict[tuple[int, int, int], pulp.LpVariable] = dict()
        for i in range(N):
            for j1, j2 in self.get_introns(i):
                # Start from k=1; no constraint on the garbage isoform
                for k in range(1, K):
                    if (key := (j1, j2, k)) not in GAPI:
                        GAPI[key] = pulp.LpVariable(
                            name=f"GAPI_{j1},{j2},{k}",
                            cat=pulp.LpInteger,
                        )
                        # Constraint fixing the value of GAPI
                        self.model += GAPI[key] == pulp.lpSum(
                            self.E2I[j, k] * self.interval_lengths[j]
                            for j in range(j1, j2 + 1)
                        )
                    self.model += (
                        GAPI[key] - MAX_ISOFORM_LG * (1 - self.R2I[i, k]) <= slack
                    )
        OBJ: dict[tuple[int, int, int], pulp.LpVariable] = dict()
        OBJ_SUM = pulp.lpSum(0.0)
        for i in range(N):
            for j in range(M):
                # Only introns can be corrected to exons
                if self.rows[i][j] != aln_t.intron:
                    continue
                for k in range(1, K):
                    OBJ[i, j, k] = pulp.LpVariable(
                        name=f"OBJ_{i},{j},{k}",
                        cat=pulp.LpBinary,
                    )
                    # Set OBJ[i, j, k] to 1, without using multiplication, if:
                    # - read i is assigned to isoform k
                    # - exon j is in isoform k
                    # - exon j is an intron in read i
                    self.model += OBJ[i, j, k] >= self.E2I[j, k] + self.R2I[i, k] - 1
                    OBJ_SUM += OBJ[i, j, k]
        # We add the chosen cost for each isoform assigned to the garbage isoform if any
        for i in range(N):
            OBJ_SUM += len(self.ridxs[i]) * self.R2I[i, 0] * (max_corrections + 1)
        self.model.setObjective(obj=OBJ_SUM)
        self.model.sense = pulp.LpMinimize

    def solve(
        self, solver: str = "COIN_CMD", threads: int = 1, timeLimit: int = 5 * 60
    ) -> tuple[int, tuple[list[aln_t], ...], tuple[list[int], ...]]:
        solver = pulp.getSolver(
            solver,
            timeLimit=timeLimit,
            threads=threads,
            msg=0,
        )
        self.model.solve(solver=solver)
        status = self.model.status
        bins: tuple[list[int], ...] = tuple(list() for _ in range(self.K))
        isoforms: tuple[list[aln_t], ...] = tuple(list() for _ in range(self.K))

        if status == pulp.LpStatusOptimal:
            for k in range(self.K):
                for i, ridxs in enumerate(self.ridxs):
                    val = self.R2I[i, k].varValue
                    assert val != None, f"Unsolved variable, {self.R2I[i, k]}"
                    if val > 0.5:
                        bins[k].extend(ridxs)
                for j in range(len(self.interval_lengths)):
                    val = self.E2I[j, k].varValue
                    assert val != None, f"Unsolved variable, {self.E2I[j, k]}"
                    isoforms[k].append(aln_t.exon if val > 0.5 else aln_t.intron)
        return self.model.status, isoforms, bins
