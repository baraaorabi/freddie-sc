from collections import defaultdict
from itertools import groupby
import typing

import pulp
import freddie_segment_and_cluster as fsac


class FredILP:
    def __init__(self, cints: fsac.canonInts):
        row_to_ridxs: defaultdict[
            tuple[fsac.canonInts.aln_type, ...],
            list[int],
        ] = defaultdict(list)
        useful_intervals = [
            len(interval.exonic_ridxs()) * len(interval.intronic_ridxs()) > 0
            for interval in cints.intervals
        ]
        self.interval_lengths = (
            (10,)
            + tuple(
                interval.end - interval.start
                for idx, interval in enumerate(cints.intervals)
                if useful_intervals[idx]
            )
            + (10,)
        )
        useful_cols = [True] + useful_intervals + [True]
        for idx, row in enumerate(cints.get_matrix()):
            row_to_ridxs[tuple(row[useful_cols])].append(idx)
        self.rows = tuple(row_to_ridxs.keys())
        self.ridxs = tuple(tuple(row_to_ridxs[k]) for k in self.rows)
        for row in self.rows:
            assert len(row) == len(self.interval_lengths)
        del row_to_ridxs, useful_intervals, useful_cols

    def get_introns(self, i) -> typing.Generator[tuple[int, int], None, None]:
        for key, group in groupby(enumerate(self.rows[i]), key=lambda x: x[1]):
            if key == fsac.canonInts.aln_type.intron:
                j1 = next(group)[0]
                j2 = j1
                for j2, _ in group:
                    pass
                yield j1, j2

    def build_model(self, K: int = 2, slack: int = 10) -> None:
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
                    name=f"R2I[{i}, {k}]",
                    cat=pulp.LpBinary,
                )
            # Constraint enforcing that each read is assigned to exactly one isoform
            self.model += pulp.lpSum(self.R2I[i, k] for k in range(0, K)) == 1

        # Implied variable: canonical exons presence in isoforms
        # E2I[j, k]     = 1 if canonical exon j is in isoform k
        # E2I_min[j, k] = 1 if canonical exon j is in isoform k and is shared by all reads of that isoform
        # E2IR[j, k, i] = 1 if read i assigned to isoform k AND exon j covered by read i
        # Auxiliary variable
        # E2IR[j, k, i] = R2I[i,k] AND I[i, j]
        # E2I[j, k]     = max over  all reads i of E2IR[j, k, i]
        # E2I_min[j, k] = min over  all reads i of E2IR[j, k, i]
        E2I: dict[tuple[int, int], pulp.LpVariable] = dict()
        E2I_min: dict[tuple[int, int], pulp.LpVariable] = dict()
        E2IR: dict[tuple[int, int, int], pulp.LpVariable] = dict()
        for j in range(M):
            # No exon is assignd to the garbage isoform
            E2I[j, 0] = pulp.LpVariable(
                name=f"E2I[{j}, 0]",
                cat=pulp.LpBinary,
            )
            self.model += E2I[(j, 0)] == 0
            # We start assigning exons from the first isoform
            for k in range(1, K):
                E2I[j, k] = pulp.LpVariable(
                    name=f"E2I[{j}, {k}]",
                    cat=pulp.LpBinary,
                )
                E2I_min[j, k] = pulp.LpVariable(
                    name=f"E2I_min[{j}, {k}]",
                    cat=pulp.LpBinary,
                )
                for i in range(N):
                    E2IR[j, k, i] = pulp.LpVariable(
                        name=f"E2IR[{j}, {k}, {i}]",
                        cat=pulp.LpBinary,
                    )
                    self.model += E2IR[j, k, i] == self.R2I[i, k] * (
                        self.rows[i][j] == fsac.canonInts.aln_type.exon
                    )
                self.model += E2I[j, k] <= pulp.lpSum(E2IR[j, k, i] for i in range(N))
                self.model += E2I_min[j, k] >= pulp.lpSum(
                    E2IR[j, k, i] for i in range(N)
                )

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
                    if (key := (j1, j2, k)) in GAPI:
                        GAPI[key] = pulp.LpVariable(
                            name=f"GAPI[{j1}, {j2}, {k}]",
                            cat=pulp.LpInteger,
                        )
                        # Constraint fixing the value of GAPI
                        self.model += GAPI[key] == pulp.lpSum(
                            E2I[j, k] * self.interval_lengths[j]
                            for j in range(j1, j2 + 1)
                        )
                    self.model += (
                        GAPI[key] - MAX_ISOFORM_LG * (1 - self.R2I[i, k]) <= slack
                    )
        OBJ: dict[tuple[int, int, int], pulp.LpVariable] = dict()
        OBJ_SUM = pulp.lpSum(0.0)
        for i in range(N):
            for j in range(M):
                if self.rows[i][j] != fsac.canonInts.aln_type.intron:
                    continue
                for k in range(1, K):
                    OBJ[i, j, k] = pulp.LpVariable(
                        name=f"OBJ[{i}, {j}, {k}]",
                        cat=pulp.LpBinary,
                    )
                    self.model += OBJ[i, j, k] == self.R2I[i, k] * E2I[j, k]
                    OBJ_SUM += OBJ[i, j, k]
        # We add the chosen cost for each isoform assigned to the garbage isoform if any
        for i in range(N):
            OBJ_SUM += len(self.ridxs[i]) * self.R2I[i, 0]
        self.model.setObjective(obj=OBJ_SUM)
        self.model.sense = pulp.LpMinimize

    def solve(
        self,
        solver: str = "cbc",
        threads: int = 1,
        timeLimit: int = 5 * 60,
    ) -> tuple[int, list[list[int]]]:
        solver = pulp.getSolver(solver, timeLimit=timeLimit, threads=threads)
        self.model.solve(solver=solver)
        status = self.model.status
        if status != pulp.LpStatusOptimal:
            return status, []
        bins: list[list[int]] = [list() for _ in range(self.K)]
        for i, ridxs in enumerate(self.ridxs):
            for k in range(self.K):
                val = self.R2I[i, k].varValue
                if val == None:
                    raise ValueError("Unsolved variable")
                elif val > 0.5:
                    bins[k].extend(ridxs)
                    break
        return self.model.status, bins
