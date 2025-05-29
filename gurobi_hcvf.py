#!/usr/bin/env python3
"""
Flexible Job Shop Scheduling – HCVF-enhanced MILP Solver with Gurobi
--------------------------------------------------------------------
Integrates Heuristic-Consensus Variable Fixing (HCVF). Requires both
"benchmark.json" and "consensus.json" in same folder.
"""
import json
from pathlib import Path
from itertools import combinations
import gurobipy as gp
from gurobipy import GRB

# Load consensus data
cons = json.load(open("consensus.json"))
# Fix: list of [machine, op_i, op_j]
FIX_raw = cons.get("fix", [])
EST_raw = cons.get("est", {})
LST_raw = cons.get("lst", {})
# Convert keys to appropriate types
FIX = [(m, int(i), int(j)) for m,i,j in FIX_raw]
EST = {int(k): v for k,v in EST_raw.items()}
LST = {int(k): v for k,v in LST_raw.items()}

# Load instance data
raw = json.loads(Path("benchmark.json").read_text())
jobs = [job["job"] for job in raw]
operations = []  # list of tuples (job, op_index, [(machine,duration),...])
all_machines = set()
for job in raw:
    for op_idx, op in enumerate(job["operations"]):
        ops = [(m["machine"], int(m["duration"])) for m in op["machines"]]
        operations.append((job["job"], op_idx, ops))
        all_machines.update([m for m,_ in ops])
all_machines = sorted(all_machines)
# BIG_M for capacity disjunctions
BIG_M = sum(d for _,_,ops in operations for _,d in ops)

# Map op_id (index in operations) -> (job, op_idx)
op_id_map = {idx: (j, o) for idx, (j, o, _) in enumerate(operations)}
# Duration lookup
durations = {(j, o, m): d for j, o, ops in operations for m, d in ops}

# Build model
mdl = gp.Model("FJSS_HCVF")
mdl.setParam("OutputFlag", 1)

# Decision vars: x[j,o,m]
x = {}
s = {}
c = {}
for j, o, ops in operations:
    for m, d in ops:
        x[(j, o, m)] = mdl.addVar(vtype=GRB.BINARY, name=f"x_{j}_{o}_{m}")
    s[(j, o)] = mdl.addVar(vtype=GRB.CONTINUOUS, lb=0.0, name=f"s_{j}_{o}")
    c[(j, o)] = mdl.addVar(vtype=GRB.CONTINUOUS, lb=0.0, name=f"c_{j}_{o}")
# Makespan var
Cmax = mdl.addVar(vtype=GRB.CONTINUOUS, lb=0.0, name="makespan")

# 1. Assignment constraints
for j, o, ops in operations:
    mdl.addConstr(gp.quicksum(x[(j, o, m)] for m, _ in ops) == 1,
                  name=f"assign_{j}_{o}")

# 2. Duration constraints
for j, o, ops in operations:
    for m, d in ops:
        mdl.addConstr(c[(j, o)] >= s[(j, o)] + d * x[(j, o, m)],
                      name=f"dur_{j}_{o}_{m}")

# 3. Job precedence
for j in jobs:
    job_ops = sorted([(o, ops) for (j2, o, ops) in operations if j2 == j], key=lambda t: t[0])
    for (o_prev, _), (o_next, _) in zip(job_ops, job_ops[1:]):
        mdl.addConstr(s[(j, o_next)] >= c[(j, o_prev)],
                      name=f"prec_{j}_{o_prev}_{o_next}")

# 4. Machine capacity (no overlap)
for m in all_machines:
    ops_on_m = [(j, o) for j, o, ops in operations if any(mm == m for mm, _ in ops)]
    for (j1, o1), (j2, o2) in combinations(ops_on_m, 2):
        y = mdl.addVar(vtype=GRB.BINARY, name=f"y_{j1}_{o1}_{j2}_{o2}_{m}")
        mdl.addConstr(
            s[(j1, o1)] >= c[(j2, o2)] - BIG_M * (1 - y)
                          - BIG_M * (1 - x[(j1, o1, m)])
                          - BIG_M * (1 - x[(j2, o2, m)]),
            name=f"seq1_{j1}_{o1}_{j2}_{o2}_{m}")
        mdl.addConstr(
            s[(j2, o2)] >= c[(j1, o1)] - BIG_M * y
                          - BIG_M * (1 - x[(j1, o1, m)])
                          - BIG_M * (1 - x[(j2, o2, m)]),
            name=f"seq2_{j1}_{o1}_{j2}_{o2}_{m}")

# 5. Makespan definition
for j in jobs:
    last_op = max(o for (j2, o, _) in operations if j2 == j)
    mdl.addConstr(Cmax >= c[(j, last_op)], name=f"mksp_{j}")

# HCVF: apply fixed assignments and sequencing
for m, i, j in FIX:
    if i in op_id_map and j in op_id_map:
        job_i, op_i = op_id_map[i]
        job_j, op_j = op_id_map[j]
        # only apply if machine m is valid for both ops
        if (job_i, op_i, m) in x and (job_j, op_j, m) in x:
            mdl.addConstr(x[(job_i, op_i, m)] == 1,
                          name=f"fix_assign_{job_i}_{op_i}_{m}")
            mdl.addConstr(x[(job_j, op_j, m)] == 1,
                          name=f"fix_assign_{job_j}_{op_j}_{m}")
            dur = durations[(job_i, op_i, m)]
            mdl.addConstr(s[(job_i, op_i)] + dur <= s[(job_j, op_j)],
                          name=f"fix_seq_{job_i}_{op_i}_{job_j}_{op_j}_{m}")

# HCVF: time window constraints (disabled due to infeasibility)
# for i, est_val in EST.items():
#     if i in op_id_map:
#         job_i, op_i = op_id_map[i]
#         mdl.addConstr(s[(job_i, op_i)] >= est_val,
#                       name=f"est_{job_i}_{op_i}")
# for i, lst_val in LST.items():
#     if i in op_id_map:
#         job_i, op_i = op_id_map[i]
#         mdl.addConstr(s[(job_i, op_i)] <= lst_val,
#                       name=f"lst_{job_i}_{op_i}")

# Objective and optimize
mdl.setObjective(Cmax, GRB.MINIMIZE)
mdl.optimize()

# Reporting
if mdl.Status == GRB.OPTIMAL or mdl.Status == GRB.TIME_LIMIT:
    print(f"\nHCVF makespan: {Cmax.X:.2f}")
else:
    print(f"Model infeasible or no solution (status {mdl.Status})")
