#!/usr/bin/env python3
from collections import defaultdict
import os

import numpy.typing as npt
import numpy as np

from networkx.algorithms import components
from networkx import Graph
import gurobipy as gurobi
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

    def build_model(self, threads: int = 1, K: int = 2):
        assert threads >= 1
        assert K >= 2
        ISOFORM_INDEX_START: int = 1
        M: int = len(self.interval_lengths)
        N: int = len(self.rows)
        MAX_ISOFORM_LG: int = sum(self.interval_lengths)
        I = tint["ilp_data"]["I"]
        C = tint["ilp_data"]["C"]
        INCOMP_READ_PAIRS = incomp_rids
        GARBAGE_COST = tint["ilp_data"]["garbage_cost"]
        informative = informative_segs(tint, remaining_rids)

        # ILP model ------------------------------------------------------
        self.model = gurobi.Model("isoforms_v9_20231014")
        self.model.setParam("OutputFlag", 0)
        self.model.setParam(gurobi.GRB.Param.Threads, threads)
        # Decision variables
        # R2I[i, k] = 1 if read i assigned to isoform k
        R2I = np.ndarray((N, K), dtype=gurobi.Var)
        # Constraint enforcing that each read is assigned to exactly one isoform
        R2I_C1 = np.ndarray(N, dtype=gurobi.Constr)
        for i in range(N):
            for k in range(K):
                R2I[i, k] = self.model.addVar(
                    vtype=gurobi.GRB.BINARY,
                    name=f"R2I[{i}, {k}]",
                )
            R2I_C1[i] = self.model.addLConstr(
                lhs=gurobi.quicksum(R2I[(i, k)] for k in range(0, K)),
                sense=gurobi.GRB.EQUAL,
                rhs=1,
                name=f"R2I_C1[{i}]",
            )

        # Implied variable: canonical exons presence in isoforms
        # E2I[j,k]     = 1 if canonical exon j is in isoform k
        # E2I_min[j,k] = 1 if canonical exon j is in isoform k and is shared by all reads of that isoform
        # E2IR[j,k,i]  = 1 if read i assigned to isoform k AND exon j covered by read i
        # Auxiliary variable
        # E2IR[j,k,i]  = R2I[i,k] AND I[i,j]
        # E2I[j,k]     = max over  all reads i of E2IR[j,k,i]
        # E2I_min[j,k] = min over  all reads i of E2IR[j,k,i]
        
        # E2I: dict[tuple[int, int], gurobi.Var] = dict()
        E2I = np.ndarray((M, K), dtype=gurobi.Var)
        E2I_C1 = {}
        E2I_min = {}
        E2I_min_C1 = {}
        E2IR = {}
        E2IR_C1 = {}
        for j in range(0, M):
            if not informative[j]:
                continue
            E2I[j] = {}
            E2I_C1[j] = {}
            E2I_min[j] = {}
            E2I_min_C1[j] = {}
            E2IR[j] = {}
            E2IR_C1[j] = {}
            # No exon is assignd to the garbage isoform
            E2I[j][0] = self.model.addVar(
                vtype=gurobi.GRB.BINARY, name="E2I[{j}][{k}]".format(j=j, k=0)
            )
            E2I_C1[j][0] = self.model.addLConstr(
                lhs=E2I[j][0],
                sense=gurobi.GRB.EQUAL,
                rhs=0,
                name="E2I_C1[{j}][{k}]".format(j=j, k=0),
            )
            # We start assigning exons from the first isoform
            for k in range(ISOFORM_INDEX_START, K):
                E2I[j][k] = self.model.addVar(
                    vtype=gurobi.GRB.BINARY, name="E2I[{j}][{k}]".format(j=j, k=k)
                )
                E2I_min[j][k] = self.model.addVar(
                    vtype=gurobi.GRB.BINARY, name="E2I_min[{j}][{k}]".format(j=j, k=k)
                )
                E2IR[j][k] = {}
                E2IR_C1[j][k] = {}
                for i in remaining_rids:
                    E2IR[j][k][i] = self.model.addVar(
                        vtype=gurobi.GRB.BINARY,
                        name="E2IR[{j}][{k}][{i}]".format(j=j, k=k, i=i),
                    )
                    E2IR_C1[j][k][i] = self.model.addLConstr(
                        lhs=E2IR[j][k][i],
                        sense=gurobi.GRB.EQUAL,
                        rhs=R2I[i][k] * I[i][j],
                        name="E2IR_C1[{j}][{k}][{i}]".format(j=j, k=k, i=i),
                    )
                E2I_C1[j][k] = self.model.addGenConstrMax(
                    resvar=E2I[j][k],
                    vars=[E2IR[j][k][i] for i in remaining_rids],
                    constant=0.0,
                    name="E2I_C1[{j}][{k}]".format(j=j, k=k),
                )
                E2I_min_C1[j][k] = self.model.addGenConstrMin(
                    resvar=E2I_min[j][k],
                    vars=[E2IR[j][k][i] for i in remaining_rids],
                    constant=0.0,
                    name="E2I_min_C1[{j}][{k}]".format(j=j, k=k),
                )

        # Adding constraints for unaligned gaps
        # If read i is assigned to isoform k, and reads[i]['gaps'] contains ((j1,j2),l), and
        # the sum of the lengths of exons in isoform k between exons j1 and j2 is L
        # then (1-EPSILON)L <= l <= (1+EPSILON)L
        # GAPI[(j1,j2,k)] = sum of the length of the exons between exons j1 and j2 (inclusively) in isoform k
        GAPI = {}
        GAPI_C1 = {}  # Constraint fixing the value of GAPI
        GAPR_C1 = (
            {}
        )  # Constraint ensuring that the unaligned gap is not too short for every isoform and gap
        GAPR_C2 = (
            {}
        )  # Constraint ensuring that the unaligned gap is not too long for every isoform and gap
        for i in remaining_rids:
            for (j1, j2), l in tint["reads"][tint["read_reps"][i][0]]["gaps"].items():
                # No such constraint on the garbage isoform if any
                for k in range(ISOFORM_INDEX_START, K):
                    if not (j1, j2, k) in GAPI:
                        assert informative[j1 % M]
                        assert informative[j2 % M]
                        assert not any(informative[j + 1 : j2])
                        GAPI[(j1, j2, k)] = self.model.addVar(
                            vtype=gurobi.GRB.INTEGER,
                            name="GAPI[({j1},{j2},{k})]".format(j1=j1, j2=j2, k=k),
                        )
                        GAPI_C1[(j1, j2, k)] = self.model.addLConstr(
                            lhs=GAPI[(j1, j2, k)],
                            sense=gurobi.GRB.EQUAL,
                            rhs=gurobi.quicksum(
                                E2I[j][k] * tint["segs"][j][2]
                                for j in range(j1 + 1, j2)
                                if informative[j]
                            ),
                            name="GAPI_C1[({j1},{j2},{k})]".format(j1=j1, j2=j2, k=k),
                        )
                    GAPR_C1[(i, j1, j2, k)] = self.model.addLConstr(
                        lhs=(1.0 - ilp_settings["epsilon"]) * GAPI[(j1, j2, k)]
                        - ilp_settings["offset"]
                        - ((1 - R2I[i][k]) * MAX_ISOFORM_LG),
                        sense=GRB.LESS_EQUAL,
                        rhs=l,
                        name="GAPR_C1[({i},{j1},{j2},{k})]".format(
                            i=i, j1=j1, j2=j2, k=k
                        ),
                    )
                    GAPR_C2[(i, j1, j2, k)] = self.model.addLConstr(
                        lhs=(1.0 + ilp_settings["epsilon"]) * GAPI[(j1, j2, k)]
                        + ilp_settings["offset"]
                        + ((1 - R2I[i][k]) * MAX_ISOFORM_LG),
                        sense=GRB.GREATER_EQUAL,
                        rhs=l,
                        name="GAPR_C2[({i},{j1},{j2},{k})]".format(
                            i=i, j1=j1, j2=j2, k=k
                        ),
                    )
        # Adding constraints for incompatible read pairs
        INCOMP_READ_PAIRS_C1 = {}
        for i1, i2 in INCOMP_READ_PAIRS:
            if not (i1 in remaining_rids and i2 in remaining_rids):
                continue
            # Again, no such constraint on the garbage isoform if any
            for k in range(ISOFORM_INDEX_START, K):
                INCOMP_READ_PAIRS_C1[(i1, i2, k)] = self.model.addLConstr(
                    lhs=R2I[i1][k] + R2I[i2][k],
                    sense=gurobi.GRB.LESS_EQUAL,
                    rhs=1,
                    name="INCOMP_READ_PAIRS_C1[({i1},{i2},{k})]".format(
                        i1=i1, i2=i2, k=k
                    ),
                )

        OBJ = {}
        OBJ_C1 = {}
        OBJ_SUM = gurobi.LinExpr(0.0)
        for i in remaining_rids:
            OBJ[i] = {}
            OBJ_C1[i] = {}
            for j in range(0, M):
                if not informative[j]:
                    continue
                if C[i][j] > 0:  # 1 if exon j not in read i but can be added to it
                    OBJ[i][j] = {}
                    OBJ_C1[i][j] = {}
                    for k in range(ISOFORM_INDEX_START, K):
                        OBJ[i][j][k] = self.model.addVar(
                            vtype=gurobi.GRB.BINARY,
                            name="OBJ[{i}][{j}][{k}]".format(i=i, j=j, k=k),
                        )
                        OBJ_C1[i][j][k] = self.model.addGenConstrAnd(
                            resvar=OBJ[i][j][k],
                            vars=[R2I[i][k], E2I[j][k]],
                            name="OBJ_C1[{i}][{j}][{k}]".format(i=i, j=j, k=k),
                        )
                        OBJ_SUM.addTerms(1.0 * C[i][j], OBJ[i][j][k])
                        #     coeffs = 1.0,
                        #     vars   = OBJ[i][j][k]
                        # )
        # We add the chosen cost for each isoform assigned to the garbage isoform if any
        GAR_OBJ = {}
        GAR_OBJ_C = {}
        for i in remaining_rids:
            if ilp_settings["recycle_model"] in ["constant", "exons", "introns"]:
                OBJ_SUM.addTerms(1.0 * GARBAGE_COST[i], R2I[i][0])
            elif ilp_settings["recycle_model"] == "relative":
                GAR_OBJ[i] = {}
                GAR_OBJ_C[i] = {}
                for j in range(0, M):
                    if not informative[j]:
                        continue
                    GAR_OBJ[i][j] = {}
                    GAR_OBJ_C[i][j] = {}
                    for k in range(ISOFORM_INDEX_START, K):
                        if I[i][j] == 1:
                            GAR_OBJ[i][j][k] = self.model.addVar(
                                vtype=gurobi.GRB.BINARY,
                                name="GAR_OBJ[{i}][{j}][{k}]".format(i=i, j=j, k=k),
                            )
                            GAR_OBJ_C[i][j][k] = self.model.addGenConstrAnd(
                                resvar=GAR_OBJ[i][j][k],
                                vars=[R2I[i][0], E2I_min[j][k]],
                                name="GAR_OBJ_C[{i}][{j}][{k}]".format(i=i, j=j, k=k),
                            )
                            OBJ_SUM.addTerms(1.0, GAR_OBJ[i][j][k])
                        elif I[i][j] == 0 and C[i][j] == 1:
                            pass

        self.model.setObjective(expr=OBJ_SUM, sense=gurobi.GRB.MINIMIZE)


def preprocess_ilp(tint, ilp_settings):
    read_reps = tint["read_reps"]
    N = len(read_reps)
    M = len(tint["segs"])
    I = (
        dict()
    )  # For each read segment, 0 if segment has same value in all reads, 1 otherwise
    C = dict()  # For each read segment, 0 if segment is first or last, 1 otherwise
    FL = dict()  # For each read, indicate First/Last segment

    for i, read_idxs in enumerate(read_reps):
        read = tint["reads"][read_idxs[0]]
        I[i] = [0 for _ in range(M)]
        for j in range(0, M):
            I[i][j] = read["data"][j] % 2

        C[i] = [0 for _ in range(M)]
        (min_i, max_i) = find_segment_read(I, i)
        read["poly_tail_category"] = "N"
        if len(read["poly_tail"]) == 1:
            tail_key = next(iter(read["poly_tail"]))
            tail_val = read["poly_tail"][tail_key]
            if tail_key in ["SA", "ST"] and tail_val[0] > 10:
                read["poly_tail_category"] = "S"
                read["gaps"][(-1, min_i)] = tail_val[1]
                min_i = 0
            elif tail_key in ["EA", "ET"] and tail_val[0] > 10:
                read["poly_tail_category"] = "E"
                read["gaps"][(max_i, M)] = tail_val[1]
                max_i = M - 1
        FL[i] = (min_i, max_i)
        for j in range(0, M):
            if min_i <= j <= max_i and read["data"][j] == 0:
                C[i][j] = 1
            else:
                C[i][j] = 0
        for ridx in read_idxs:
            tint["reads"][ridx]["poly_tail_category"] = read["poly_tail_category"]
            tint["reads"][ridx]["gaps"] = read["gaps"]
    # Assigning a cost to the assignment of reads to the garbage isoform
    garbage_cost = {}
    for i in range(N):
        if ilp_settings["recycle_model"] == "exons":
            garbage_cost[i] = len(read_reps[i]) * garbage_cost_exons(I=I[i])
        elif ilp_settings["recycle_model"] == "introns":
            garbage_cost[i] = len(read_reps[i]) * garbage_cost_introns(C=C[i])
        elif ilp_settings["recycle_model"] == "constant":
            garbage_cost[i] = len(read_reps[i]) * 3
    tint["ilp_data"] = dict(
        FL=FL,
        I=I,
        C=C,
        garbage_cost=garbage_cost,
    )


def run_ilp(tint, remaining_rids, incomp_rids, ilp_settings, log_prefix):
    # Variables directly based on the input ------------------------------------
    # I[i,j] = 1 if reads[i]['data'][j]==1 and 0 if reads[i]['data'][j]==0 or 2
    # C[i,j] = 1 if exon j is between the first and last exons (inclusively)
    #   covered by read i and is not in read i but can be turned into a 1
    ISOFORM_INDEX_START = 1
    M = len(tint["segs"])
    MAX_ISOFORM_LG = sum(seg[2] for seg in tint["segs"])
    I = tint["ilp_data"]["I"]
    C = tint["ilp_data"]["C"]
    INCOMP_READ_PAIRS = incomp_rids
    GARBAGE_COST = tint["ilp_data"]["garbage_cost"]
    informative = informative_segs(tint, remaining_rids)

    # ILP model ------------------------------------------------------
    ILP_ISOFORMS = Model("isoforms_v8_20210209")
    ILP_ISOFORMS.setParam("OutputFlag", 0)
    ILP_ISOFORMS.setParam(GRB.Param.Threads, ilp_settings["threads"])
    # Decision variables
    # R2I[i,k] = 1 if read i assigned to isoform k
    R2I = {}
    R2I_C1 = (
        {}
    )  # Constraint enforcing that each read is assigned to exactly one isoform
    for i in remaining_rids:
        R2I[i] = {}
        for k in range(K):
            R2I[i][k] = ILP_ISOFORMS.addVar(
                vtype=GRB.BINARY, name="R2I[{i}][{k}]".format(i=i, k=k)
            )
        R2I_C1[i] = ILP_ISOFORMS.addLConstr(
            lhs=quicksum(R2I[i][k] for k in range(0, K)),
            sense=GRB.EQUAL,
            rhs=1,
            name="R2I_C1[{i}]".format(i=i),
        )

    # Implied variable: canonical exons presence in isoforms
    # E2I[j,k]     = 1 if canonical exon j is in isoform k
    # E2I_min[j,k] = 1 if canonical exon j is in isoform k and is shared by all reads of that isoform
    # E2IR[j,k,i]  = 1 if read i assigned to isoform k AND exon j covered by read i
    # Auxiliary variable
    # E2IR[j,k,i]  = R2I[i,k] AND I[i,j]
    # E2I[j,k]     = max over  all reads i of E2IR[j,k,i]
    # E2I_min[j,k] = min over  all reads i of E2IR[j,k,i]
    E2I = {}
    E2I_C1 = {}
    E2I_min = {}
    E2I_min_C1 = {}
    E2IR = {}
    E2IR_C1 = {}
    for j in range(0, M):
        if not informative[j]:
            continue
        E2I[j] = {}
        E2I_C1[j] = {}
        E2I_min[j] = {}
        E2I_min_C1[j] = {}
        E2IR[j] = {}
        E2IR_C1[j] = {}
        # No exon is assignd to the garbage isoform
        E2I[j][0] = ILP_ISOFORMS.addVar(
            vtype=GRB.BINARY, name="E2I[{j}][{k}]".format(j=j, k=0)
        )
        E2I_C1[j][0] = ILP_ISOFORMS.addLConstr(
            lhs=E2I[j][0],
            sense=GRB.EQUAL,
            rhs=0,
            name="E2I_C1[{j}][{k}]".format(j=j, k=0),
        )
        # We start assigning exons from the first isoform
        for k in range(ISOFORM_INDEX_START, K):
            E2I[j][k] = ILP_ISOFORMS.addVar(
                vtype=GRB.BINARY, name="E2I[{j}][{k}]".format(j=j, k=k)
            )
            E2I_min[j][k] = ILP_ISOFORMS.addVar(
                vtype=GRB.BINARY, name="E2I_min[{j}][{k}]".format(j=j, k=k)
            )
            E2IR[j][k] = {}
            E2IR_C1[j][k] = {}
            for i in remaining_rids:
                E2IR[j][k][i] = ILP_ISOFORMS.addVar(
                    vtype=GRB.BINARY, name="E2IR[{j}][{k}][{i}]".format(j=j, k=k, i=i)
                )
                E2IR_C1[j][k][i] = ILP_ISOFORMS.addLConstr(
                    lhs=E2IR[j][k][i],
                    sense=GRB.EQUAL,
                    rhs=R2I[i][k] * I[i][j],
                    name="E2IR_C1[{j}][{k}][{i}]".format(j=j, k=k, i=i),
                )
            E2I_C1[j][k] = ILP_ISOFORMS.addGenConstrMax(
                resvar=E2I[j][k],
                vars=[E2IR[j][k][i] for i in remaining_rids],
                constant=0.0,
                name="E2I_C1[{j}][{k}]".format(j=j, k=k),
            )
            E2I_min_C1[j][k] = ILP_ISOFORMS.addGenConstrMin(
                resvar=E2I_min[j][k],
                vars=[E2IR[j][k][i] for i in remaining_rids],
                constant=0.0,
                name="E2I_min_C1[{j}][{k}]".format(j=j, k=k),
            )

    # Adding constraints for unaligned gaps
    # If read i is assigned to isoform k, and reads[i]['gaps'] contains ((j1,j2),l), and
    # the sum of the lengths of exons in isoform k between exons j1 and j2 is L
    # then (1-EPSILON)L <= l <= (1+EPSILON)L
    # GAPI[(j1,j2,k)] = sum of the length of the exons between exons j1 and j2 (inclusively) in isoform k
    GAPI = {}
    GAPI_C1 = {}  # Constraint fixing the value of GAPI
    GAPR_C1 = (
        {}
    )  # Constraint ensuring that the unaligned gap is not too short for every isoform and gap
    GAPR_C2 = (
        {}
    )  # Constraint ensuring that the unaligned gap is not too long for every isoform and gap
    for i in remaining_rids:
        for (j1, j2), l in tint["reads"][tint["read_reps"][i][0]]["gaps"].items():
            # No such constraint on the garbage isoform if any
            for k in range(ISOFORM_INDEX_START, K):
                if not (j1, j2, k) in GAPI:
                    assert informative[j1 % M]
                    assert informative[j2 % M]
                    assert not any(informative[j + 1 : j2])
                    GAPI[(j1, j2, k)] = ILP_ISOFORMS.addVar(
                        vtype=GRB.INTEGER,
                        name="GAPI[({j1},{j2},{k})]".format(j1=j1, j2=j2, k=k),
                    )
                    GAPI_C1[(j1, j2, k)] = ILP_ISOFORMS.addLConstr(
                        lhs=GAPI[(j1, j2, k)],
                        sense=GRB.EQUAL,
                        rhs=quicksum(
                            E2I[j][k] * tint["segs"][j][2]
                            for j in range(j1 + 1, j2)
                            if informative[j]
                        ),
                        name="GAPI_C1[({j1},{j2},{k})]".format(j1=j1, j2=j2, k=k),
                    )
                GAPR_C1[(i, j1, j2, k)] = ILP_ISOFORMS.addLConstr(
                    lhs=(1.0 - ilp_settings["epsilon"]) * GAPI[(j1, j2, k)]
                    - ilp_settings["offset"]
                    - ((1 - R2I[i][k]) * MAX_ISOFORM_LG),
                    sense=GRB.LESS_EQUAL,
                    rhs=l,
                    name="GAPR_C1[({i},{j1},{j2},{k})]".format(i=i, j1=j1, j2=j2, k=k),
                )
                GAPR_C2[(i, j1, j2, k)] = ILP_ISOFORMS.addLConstr(
                    lhs=(1.0 + ilp_settings["epsilon"]) * GAPI[(j1, j2, k)]
                    + ilp_settings["offset"]
                    + ((1 - R2I[i][k]) * MAX_ISOFORM_LG),
                    sense=GRB.GREATER_EQUAL,
                    rhs=l,
                    name="GAPR_C2[({i},{j1},{j2},{k})]".format(i=i, j1=j1, j2=j2, k=k),
                )
    # Adding constraints for incompatible read pairs
    INCOMP_READ_PAIRS_C1 = {}
    for i1, i2 in INCOMP_READ_PAIRS:
        if not (i1 in remaining_rids and i2 in remaining_rids):
            continue
        # Again, no such constraint on the garbage isoform if any
        for k in range(ISOFORM_INDEX_START, K):
            INCOMP_READ_PAIRS_C1[(i1, i2, k)] = ILP_ISOFORMS.addLConstr(
                lhs=R2I[i1][k] + R2I[i2][k],
                sense=GRB.LESS_EQUAL,
                rhs=1,
                name="INCOMP_READ_PAIRS_C1[({i1},{i2},{k})]".format(i1=i1, i2=i2, k=k),
            )

    OBJ = {}
    OBJ_C1 = {}
    OBJ_SUM = LinExpr(0.0)
    for i in remaining_rids:
        OBJ[i] = {}
        OBJ_C1[i] = {}
        for j in range(0, M):
            if not informative[j]:
                continue
            if C[i][j] > 0:  # 1 if exon j not in read i but can be added to it
                OBJ[i][j] = {}
                OBJ_C1[i][j] = {}
                for k in range(ISOFORM_INDEX_START, K):
                    OBJ[i][j][k] = ILP_ISOFORMS.addVar(
                        vtype=GRB.BINARY,
                        name="OBJ[{i}][{j}][{k}]".format(i=i, j=j, k=k),
                    )
                    OBJ_C1[i][j][k] = ILP_ISOFORMS.addGenConstrAnd(
                        resvar=OBJ[i][j][k],
                        vars=[R2I[i][k], E2I[j][k]],
                        name="OBJ_C1[{i}][{j}][{k}]".format(i=i, j=j, k=k),
                    )
                    OBJ_SUM.addTerms(1.0 * C[i][j], OBJ[i][j][k])
                    #     coeffs = 1.0,
                    #     vars   = OBJ[i][j][k]
                    # )
    # We add the chosen cost for each isoform assigned to the garbage isoform if any
    GAR_OBJ = {}
    GAR_OBJ_C = {}
    for i in remaining_rids:
        if ilp_settings["recycle_model"] in ["constant", "exons", "introns"]:
            OBJ_SUM.addTerms(1.0 * GARBAGE_COST[i], R2I[i][0])
        elif ilp_settings["recycle_model"] == "relative":
            GAR_OBJ[i] = {}
            GAR_OBJ_C[i] = {}
            for j in range(0, M):
                if not informative[j]:
                    continue
                GAR_OBJ[i][j] = {}
                GAR_OBJ_C[i][j] = {}
                for k in range(ISOFORM_INDEX_START, K):
                    if I[i][j] == 1:
                        GAR_OBJ[i][j][k] = ILP_ISOFORMS.addVar(
                            vtype=GRB.BINARY,
                            name="GAR_OBJ[{i}][{j}][{k}]".format(i=i, j=j, k=k),
                        )
                        GAR_OBJ_C[i][j][k] = ILP_ISOFORMS.addGenConstrAnd(
                            resvar=GAR_OBJ[i][j][k],
                            vars=[R2I[i][0], E2I_min[j][k]],
                            name="GAR_OBJ_C[{i}][{j}][{k}]".format(i=i, j=j, k=k),
                        )
                        OBJ_SUM.addTerms(1.0, GAR_OBJ[i][j][k])
                    elif I[i][j] == 0 and C[i][j] == 1:
                        pass

    ILP_ISOFORMS.setObjective(expr=OBJ_SUM, sense=GRB.MINIMIZE)

    # Optimization
    # ILP_ISOFORMS.Params.PoolSearchMode=2
    # ILP_ISOFORMS.Params.PoolSolutions=5
    ILP_ISOFORMS.setParam("TuneOutput", 1)
    if not log_prefix == None:
        ILP_ISOFORMS.setParam("LogFile", "{}.glog".format(log_prefix))
        ILP_ISOFORMS.write("{}.lp".format(log_prefix))
    ILP_ISOFORMS.setParam("TimeLimit", ilp_settings["timeout"] * 60)
    ILP_ISOFORMS.optimize()

    ILP_ISOFORMS_STATUS = ILP_ISOFORMS.Status

    isoforms = {k: dict() for k in range(ISOFORM_INDEX_START, K)}
    # print('STATUS: {}'.format(ILP_ISOFORMS_STATUS))
    # if ILP_ISOFORMS_STATUS == GRB.Status.TIME_LIMIT:
    #     status = 'TIME_LIMIT'
    if ILP_ISOFORMS_STATUS != GRB.Status.OPTIMAL:
        status = "NO_SOLUTION"
    else:
        status = "OPTIMAL"
        # Writing the optimal solution to disk
        if not log_prefix == None:
            solution_file = open("{}.sol".format(log_prefix), "w+")
            for v in ILP_ISOFORMS.getVars():
                solution_file.write("{}\t{}\n".format(v.VarName, v.X))
            solution_file.close()
        # Isoform id to isoform structure
        for k in range(ISOFORM_INDEX_START, K):
            isoforms[k]["exons"] = list()
            for j in range(0, M):
                if informative[j]:
                    isoforms[k]["exons"].append(
                        int(E2I[j][k].getAttr(GRB.Attr.X) > 0.9)
                    )
                else:
                    isoforms[k]["exons"].append(I[next(iter(remaining_rids))][j])
            isoforms[k]["rid_to_corrections"] = dict()
        # Isoform id to read ids set
        for i in remaining_rids:
            isoform_id = -1
            for k in range(0, K):
                if R2I[i][k].getAttr(GRB.Attr.X) > 0.9:
                    assert (
                        isoform_id == -1
                    ), "Read {} has been assigned to multiple isoforms!".format(i)
                    isoform_id = k
            assert (
                isoform_id != -1
            ), "Read {} has not been assigned to any isoform!".format(i)
            if isoform_id == 0:
                continue
            isoforms[isoform_id]["rid_to_corrections"][i] = -1
        # Read id to its exon corrections
        for k in range(ISOFORM_INDEX_START, K):
            for i in isoforms[k]["rid_to_corrections"].keys():
                isoforms[k]["rid_to_corrections"][i] = [
                    str(tint["reads"][tint["read_reps"][i][0]]["data"][j])
                    for j in range(M)
                ]
                for j in range(0, M):
                    if not informative[j]:
                        isoforms[k]["rid_to_corrections"][i][j] = "-"
                    elif C[i][j] == 1 and OBJ[i][j][k].getAttr(GRB.Attr.X) > 0.9:
                        isoforms[k]["rid_to_corrections"][i][j] = "X"
    return ILP_ISOFORMS_STATUS, status, isoforms


def main():
    args = parse_args()
    args.segment_dir = args.segment_dir.rstrip("/")

    ilp_settings = dict(
        recycle_model=args.recycle_model,
        K=2,
        epsilon=args.epsilon,
        offset=args.gap_offset,
        timeout=args.timeout,
        max_rounds=args.max_rounds,
        threads=1,
    )
    cluster_args = list()
    for contig in os.listdir(args.segment_dir):
        if not os.path.isdir("{}/{}".format(args.segment_dir, contig)):
            continue
        os.makedirs("{}/{}".format(args.outdir, contig), exist_ok=False)
        if args.logs_dir != None:
            os.makedirs("{}/{}".format(args.logs_dir, contig), exist_ok=False)
        for tint_id in glob.iglob(
            "{}/{}/segment_*.tsv".format(args.segment_dir, contig)
        ):
            tint_id = int(tint_id[:-4].split("/")[-1].split("_")[-1])
            cluster_args.append(
                [
                    args.segment_dir,
                    args.outdir,
                    contig,
                    tint_id,
                    ilp_settings,
                    args.min_isoform_size,
                    args.max_ilp,
                    "{}/{}".format(args.logs_dir, contig)
                    if args.logs_dir != None
                    else None,
                ]
            )
