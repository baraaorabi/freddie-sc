from collections import defaultdict
from dataclasses import dataclass
from itertools import groupby
from typing import Generator

from freddie.segment import CanonIntervals, aln_t

import pulp

ALN_T_MAP = {
    aln_t.exon: aln_t.exon,
    aln_t.polyA: aln_t.exon,
    aln_t.intron: aln_t.intron,
    aln_t.unaln: aln_t.intron,
}


@dataclass
class IlpParams:
    timeLimit: int = 5 * 60
    max_correction_len: int = 20
    max_correction_count: int = 3
    ilp_solver: str = "COIN_CMD"
    ilp_threads: int = 1


class FredILP:
    def __init__(self, cints: CanonIntervals, params: IlpParams = IlpParams()):
        self.params = params
        data_to_ridxs: defaultdict[
            tuple[tuple[aln_t, ...], tuple[str, ...]],
            list[int],
        ] = defaultdict(list)
        self.interval_lengths = (10,) + tuple(map(len, cints.intervals)) + (10,)
        for idx, row in enumerate(cints.get_matrix()):
            cell_types = cints.reads[idx].cell_types
            first = len(row) - 1
            last = 0
            for j, aln_type in enumerate(row):
                if aln_type in [aln_t.exon, aln_t.polyA]:
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

    def get_introns(self, i) -> Generator[tuple[int, int], None, None]:
        for key, group in groupby(enumerate(self.rows[i]), key=lambda x: x[1]):
            if key == aln_t.intron:
                j1 = next(group)[0]
                j2 = j1
                for j2, _ in group:
                    pass
                yield j1, j2

    def ilp_max_binary(
        self,
        lhs_var: pulp.LpVariable,
        rhs_vars: list[pulp.LpVariable],
    ):
        # if any rhs_var is 1, then lhs_var is 1
        for rhs_var in rhs_vars:
            self.model += lhs_var >= rhs_var
        # if all rhs_vars are 0, then lhs_var is 0
        self.model += lhs_var <= pulp.lpSum(rhs_vars)

    def ilp_xor_binary(
        self,
        lhs_var: pulp.LpVariable,
        x: pulp.LpVariable,
        y: pulp.LpVariable,
    ):
        # If both x and y are 0, then lhs_var is 0
        self.model += lhs_var <= x + y
        # If x is 1 and y is 0, then lhs_var is 1
        self.model += lhs_var >= x - y
        # If y is 1 and x is 0, then lhs_var is 1
        self.model += lhs_var >= y - x
        # If both x and y are 1, then lhs_var is 0
        self.model += lhs_var <= 2 - x - y

    def ilp_and_binary(
        self,
        lhs_var: pulp.LpVariable,
        x: pulp.LpVariable,
        y: pulp.LpVariable,
    ):
        # If x is 0, then lhs_var is 0
        self.model += lhs_var <= x
        # If y is 0, then lhs_var is 0
        self.model += lhs_var <= y
        # If x and y are both 1, then lhs_var is 1
        self.model += lhs_var >= x + y - 1

    def build_model(
        self,
        K: int = 2,
    ) -> None:
        ## Some constants ##
        ct_to_ctidx: dict[str, int] = dict()
        ctidx_to_ct: dict[int, str] = dict()
        for cell_types in self.cell_types:
            for cell_type in cell_types:
                if cell_type not in ct_to_ctidx:
                    ctidx = len(ct_to_ctidx)
                    ct_to_ctidx[cell_type] = ctidx
                    ctidx_to_ct[ctidx] = cell_type
        assert K >= 2
        self.K: int = K
        J: int = len(ct_to_ctidx)
        M: int = len(self.interval_lengths)
        N: int = len(self.rows)
        MAX_ISOFORM_LG: int = sum(self.interval_lengths)

        # --------------------------- ILP model --------------------------
        self.model = pulp.LpProblem("freddie_v20240125", pulp.LpMinimize)
        # ---------------------- Decision variables ----------------------
        # R2I[i, k] = 1 if read i assigned to isoform k
        self.R2I: dict[tuple[int, int], pulp.LpVariable] = dict()
        for i in range(N):
            for k in range(K):
                self.R2I[i, k] = pulp.LpVariable(name=f"R2I_{i},{k}", cat=pulp.LpBinary)
            # Each read must be assigned to exactly one bin
            self.model += pulp.lpSum(self.R2I[i, k] for k in range(0, K)) == 1
        # ---------------------- Helping variables -----------------------
        ## Vars for canonical exons presence in isoforms
        # E2I[j, k]     = 1 if canonical exon j is in isoform k,
        #                 i.e., over all reads i, E2I >= E2IR[j, k, i]
        # E2IR[j, k, i] = 1 if read i assigned to isoform k AND exon j covered by read i,
        #                 i.e., R2I[i,k] AND (rows[i][j] == exon)
        ## Vars for exon contiguity in reads and isoforms
        # EXON_CONTIG2IR[j, k, i] = 1 if exons j and j + 1 are both expressed in read i if it's assigned to isoform k
        #                           i.e., R2I[i, k] AND E2IR[j, k, i] AND E2IR[j + 1, k, i]
        # EXON_CONTIG2I[j, k] = 1 if exon j and exon j + 1 are both expressed in isoform k
        #                      i.e., E2I[j, k] AND E2I[j + 1, k]
        ## Vars for interval being covered (intronically or exonically) by isoform
        # C2IR[j, k, i]  = 1 if read i assigned to isoform k AND interval j covered by read i,
        #                  i.e., R2I[i,k] AND (rows[i][j] != unaln)
        # C2I[j, k]      = 1 if interval j is covered by isoform k,
        #                  i.e., over all reads i, C2I >= C2IR[j, k, i]
        ## Vars for interval changes coverage state from its previous interval
        # CHANGE2I[j, k] = C2I[j, k] XOR C2I[j + 1, k]
        #                  Per isoform, the sum over CHANGE2I vals should be exactly 2
        ## Vars for cell type being expressed by reads in isoform k
        # I2T[k, l]      = 1 if isoform k has a read with cell type l
        #                  i.e., over all reads i with cell type l,
        #                        I2T[k, l] = max R2I[i, k],
        ## Vars for interval being covered by isoform in specific cell type
        # C2IRT[j, k, i, l] = 1 if read i has cell type l,
        #                     assigned to isoform k,
        #                     AND interval j covered by read i,
        #                     i.e., R2I[i,k] AND (rows[i][j] != unaln)
        # C2IT[j, k, j]     = 1 if interval j is covered by isoform k,
        #                     i.e., over all reads i with cell type l,
        #                     C2IT[j, k, l] = max(C2IRT[j, k, i, l])
        ## Vars for unaligned gaps
        # GAPI[j1, j2, k] = sum of the length of the intervals
        #                   between intervals j1 and j2 (inclusively) in isoform k
        self.E2I: dict[tuple[int, int], pulp.LpVariable] = dict()
        E2IR: dict[tuple[int, int, int], pulp.LpVariable] = dict()
        EXON_CONTIG2IR: dict[tuple[int, int, int], pulp.LpVariable] = dict()
        EXON_CONTIG2I: dict[tuple[int, int], pulp.LpVariable] = dict()
        C2I: dict[tuple[int, int], pulp.LpVariable] = dict()
        C2IR: dict[tuple[int, int, int], pulp.LpVariable] = dict()
        CHANGE2I: dict[tuple[int, int], pulp.LpVariable] = dict()
        I2T: dict[tuple[int, int], pulp.LpVariable] = dict()
        C2IRT: dict[tuple[int, int, int, int], pulp.LpVariable] = dict()
        C2IT: dict[tuple[int, int, int], pulp.LpVariable] = dict()
        GAPI: dict[tuple[int, int, int], pulp.LpVariable] = dict()
        # Setting up E2IR and E2I vars
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
                self.ilp_max_binary(
                    lhs_var=self.E2I[j, k],
                    rhs_vars=[E2IR[j, k, i] for i in range(N)],
                )
        # Setting up EXON_CONTIG2IR and EXON_CONTIG2I vars
        for k in range(1, K):
            for i in range(N):
                for j in range(M - 1):
                    EXON_CONTIG2IR[j, k, i] = pulp.LpVariable(
                        name=f"EXON_CONTIG2IR_{j},{k},{i}",
                        cat=pulp.LpBinary,
                    )
                    row = self.rows[i]
                    is_contig = int(row[j] == aln_t.exon and row[j + 1] == aln_t.exon)
                    self.model += EXON_CONTIG2IR[j, k, i] == is_contig * self.R2I[i, k]
            for j in range(M - 1):
                EXON_CONTIG2I[j, k] = pulp.LpVariable(
                    name=f"EXON_CONTIG2I_{j},{k}",
                    cat=pulp.LpBinary,
                )
                # EXON_CONTIG2I[j, k] = max over all reads i of EXON_CONTIG2IR[j, k, i]
                self.ilp_max_binary(
                    lhs_var=EXON_CONTIG2I[j, k],
                    rhs_vars=[EXON_CONTIG2IR[j, k, i] for i in range(N)],
                )
        # Setting up C2IR and C2I vars
        for k in range(1, K):  # Start from k=1; no constraint on the garbage isoform
            # Add two extra intervals of no coverage at the start and end
            # of the isoform to account to ensure that CHANGE2I is at least 2
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
                    self.model += C2IR[j, k, i] == (
                        self.R2I[i, k] * (self.rows[i][j] != aln_t.unaln)
                    )
                # C2I[j, k] = max over of C2IR[j, k, i] for each read i
                self.ilp_max_binary(
                    lhs_var=C2I[j, k],
                    rhs_vars=[C2IR[j, k, i] for i in range(N)],
                )
        # Setting up CHANGE2I vars
        for k in range(1, K):
            for j in range(-1, M):
                CHANGE2I[j, k] = pulp.LpVariable(
                    name=f"CHANGE2I_{'n' if j < 0 else ''}{abs(j)},{k}",
                    cat=pulp.LpBinary,
                )
                self.ilp_xor_binary(
                    lhs_var=CHANGE2I[j, k],
                    x=C2I[j, k],
                    y=C2I[j + 1, k],
                )
        # Setting up I2T vars
        for k in range(1, K):
            for l in range(J):
                I2T[k, l] = pulp.LpVariable(
                    name=f"I2T_{k},{l}",
                    cat=pulp.LpBinary,
                )
                i_list = [i for i in range(N) if ctidx_to_ct[l] in self.cell_types[i]]
                # Max over all reads i with cell type l of R2I[i, k]
                self.ilp_max_binary(
                    lhs_var=I2T[k, l],
                    rhs_vars=[self.R2I[i, k] for i in i_list],
                )
        # Setting up C2IRT and C2IT vars
        for k in range(1, K):
            for j in range(M):
                for i in range(N):
                    for cell_type in self.cell_types[i]:
                        l = ct_to_ctidx[cell_type]
                        C2IRT[j, k, i, l] = pulp.LpVariable(
                            name=f"C2IRT_{j},{k},{i},{l}",
                            cat=pulp.LpBinary,
                        )
                        self.model += C2IRT[j, k, i, l] == (
                            self.R2I[i, k] * (self.rows[i][j] != aln_t.unaln)
                        )
        for k in range(1, K):
            for j in range(M):
                for l in range(J):
                    C2IT[j, k, l] = pulp.LpVariable(
                        name=f"C2IT_{j},{k},{l}",
                        cat=pulp.LpBinary,
                    )
                    i_list = [
                        i for i in range(N) if ctidx_to_ct[l] in self.cell_types[i]
                    ]
                    # If any read has cell type l and exon j, then C2IT[j, k, l] = 1
                    # I.e. C2IT[j, k, l] = max over all reads i with cell type l of C2IRT[j, k, i, l]
                    self.ilp_max_binary(
                        lhs_var=C2IT[j, k, l],
                        rhs_vars=[C2IRT[j, k, i, l] for i in i_list],
                    )
        # Setting up GAPI vars
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
        # ------------------------- Constraints -------------------------
        ## Contraint: On isoform contiguity
        # Each isoform coverage must be contiguous. Coverage includes all exons and introns intervals.
        # I.e. for each isoform k, the sum of CHANGE2I[j, k] over all j must be exactly 2
        for k in range(1, K):
            self.model += pulp.lpSum(CHANGE2I[j, k] for j in range(-1, M)) == 2
        # All adjacent exons in an isoform must be contiguous in one of the reads assigned to the isoform
        # I.e. for each isoform k, if E2I[j] = 1 and E2I[j + 1] = 1, then EXON_CONTIG2I[j, k] = 1
        for k in range(1, K):
            for j in range(M - 1):
                self.ilp_and_binary(
                    lhs_var=EXON_CONTIG2I[j, k],
                    x=self.E2I[j, k],
                    y=self.E2I[j + 1, k],
                )
        ## Contraint: On cell type
        # If isoform k has a read with cell type l (I2T[k, l] = 1),
        # and interval j is covered by isoform k (C2I[j, k] = 1),
        # then interval j must be covered by isoform k in cell type l reads (C2IT[j, k, l] = 1)
        # I.e. C2IT[j, k, l] = I2T[k, l] & C2I[j, k] - 1
        for k in range(1, K):
            for l in range(J):
                for j in range(M):
                    self.ilp_and_binary(
                        lhs_var=C2IT[j, k, l],
                        x=I2T[k, l],
                        y=C2I[j, k],
                    )
        ## Contraint: On polyA tail
        # There can be only one (or zero) polyA interval per isoform
        for k in range(1, K):
            self.model += self.E2I[0, k] + self.E2I[M - 1, k] <= 1
        ## Contraint: On unaligned gaps
        # Adding constraints for unaligned gaps
        # If read i is assigned to isoform k, and read i contains intron j1 <-> j2, and
        # the sum of the lengths of exons in isoform k between exons j1 and j2 is L
        # then L <= slack
        for i in range(N):
            for j1, j2 in self.get_introns(i):
                # Start from k=1; no constraint on the garbage isoform
                for k in range(1, K):
                    self.model += (
                        GAPI[(j1, j2, k)] - MAX_ISOFORM_LG * (1 - self.R2I[i, k])
                        <= self.params.max_correction_len
                    )

        # ----------------------- Objective function -------------------------
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
                    # Set OBJ[i, j, k] to conjuction of:
                    # - exon j is an intron in read i (ensured by if statement above),
                    # - read i is assigned to isoform k, and
                    # - exon j is in isoform k
                    self.ilp_and_binary(
                        lhs_var=OBJ[i, j, k],
                        x=self.R2I[i, k],
                        y=self.E2I[j, k],
                    )
                    OBJ_SUM += OBJ[i, j, k] * len(self.ridxs[i])
        # We add the chosen cost for each isoform assigned to the garbage isoform if any
        for i in range(N):
            OBJ_SUM += (
                len(self.ridxs[i])
                * self.R2I[i, 0]
                * (self.params.max_correction_count + 1)
            )
        self.model.setObjective(obj=OBJ_SUM)
        self.model.sense = pulp.LpMinimize

    def solve(
        self,
    ) -> tuple[int, tuple[list[aln_t], ...], tuple[list[int], ...]]:
        self.model.solve(
            solver=pulp.getSolver(
                self.params.ilp_solver,
                timeLimit=self.params.timeLimit,
                threads=self.params.ilp_threads,
                msg=0,
            )
        )
        status = self.model.status
        bins: tuple[list[int], ...] = tuple(list() for _ in range(self.K))
        bin_structures: tuple[list[aln_t], ...] = tuple(list() for _ in range(self.K))
        assert status in (pulp.LpStatusNotSolved, pulp.LpStatusOptimal), pulp.LpStatus[
            status
        ]
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
                    bin_structures[k].append(aln_t.exon if val > 0.5 else aln_t.intron)
        return self.model.status, bin_structures, bins
