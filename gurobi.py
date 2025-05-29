#!/usr/bin/env python3
"""
Flexible Job Shop Scheduling – Exact MILP Solver with Gurobi
-----------------------------------------------------------
Reads an instance from ``benchmark.json`` (same folder) and solves the
makespan-minimisation problem exactly, *without* any time-limit.  The JSON file
must have the following structure:

[
  {
    "job": "A",
    "operations": [
      {
        "machines": [
          {"machine": "7", "duration": 15},
          {"machine": "8", "duration": 11},
          {"machine": "4", "duration": 5},
          {"machine": "5", "duration": 19}
        ]
      },
      { "machines": [ {"machine": "3", "duration": 18}, {"machine": "4", "duration": 5} ] }
    ]
  },
  …
]

How to run
~~~~~~~~~~
1.   Place ``benchmark.json`` next to this script.
2.   ``pip install gurobipy`` and make sure you have a valid Gurobi licence.
3.   ``python fjss_exact_gurobi.py``

The solver prints the optimal makespan and the schedule in job/operation order.
"""
import json
from pathlib import Path
from itertools import combinations
import gurobipy as gp
from gurobipy import GRB

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
INSTANCE_FILE = Path(__file__).with_name("benchmark.json")

with INSTANCE_FILE.open() as fp:
    raw_data = json.load(fp)

jobs = [job["job"] for job in raw_data]
operations = []  # list of (job, op_index, [ (machine, duration), … ])
all_machines = set()

for job in raw_data:
    for op_idx, op in enumerate(job["operations" ]):
        ops = [(str(m["machine" ]), m["duration" ]) for m in op["machines" ]]
        operations.append((job["job" ], op_idx, ops))
        all_machines.update(m for m, _ in ops)

all_machines = sorted(all_machines)
BIG_M = sum(d for _, _, ops in operations for _, d in ops)  # simple safe bound

# ---------------------------------------------------------------------------
# Model building
# ---------------------------------------------------------------------------
mdl = gp.Model("FJSS_exact")
mdl.setParam("OutputFlag", 1)  # change to 0 to silence Gurobi

# Decision variables ---------------------------------------------------------
# x[j,o,m] == 1  if machine m processes operation (j,o)
x = {}
for j, o, ops in operations:
    for m, _ in ops:
        x[(j, o, m)] = mdl.addVar(vtype=GRB.BINARY, name=f"x_{j}_{o}_{m}")

# start time and completion time for each operation
s, c = {}, {}
for j, o, _ in operations:
    s[(j, o)] = mdl.addVar(vtype=GRB.CONTINUOUS, lb=0.0, name=f"s_{j}_{o}")
    c[(j, o)] = mdl.addVar(vtype=GRB.CONTINUOUS, lb=0.0, name=f"c_{j}_{o}")

# makespan
Cmax = mdl.addVar(vtype=GRB.CONTINUOUS, lb=0.0, name="makespan")

# Constraints ----------------------------------------------------------------
# 1. Each operation assigned to exactly one machine
for j, o, ops in operations:
    mdl.addConstr(gp.quicksum(x[(j, o, m)] for m, _ in ops) == 1,
                  name=f"assign_{j}_{o}")

# 2. Define completion time depending on chosen machine/duration
for j, o, ops in operations:
    for m, dur in ops:
        mdl.addConstr(c[(j, o)] >= s[(j, o)] + dur * x[(j, o, m)],
                      name=f"dur_{j}_{o}_{m}")

# 3. Job precedence (operations in sequence)
for j in jobs:
    # Collect ops in order for this job
    job_ops = [(o, ops) for (j2, o, ops) in operations if j2 == j]
    job_ops.sort(key=lambda t: t[0])
    for (o_prev, _), (o_next, _) in zip(job_ops, job_ops[1:]):
        mdl.addConstr(s[(j, o_next)] >= c[(j, o_prev)], name=f"prec_{j}_{o_prev}_{o_next}")

# 4. Machine capacity (no overlap)
for m in all_machines:
    # operations that *can* run on m
    ops_on_m = [(j, o) for j, o, ops in operations if any(mm == m for mm, _ in ops)]
    for (j1, o1), (j2, o2) in combinations(ops_on_m, 2):
        y = mdl.addVar(vtype=GRB.BINARY, name=f"y_{j1}_{o1}_{j2}_{o2}_{m}")
        # s_j1 >= c_j2  OR  s_j2 >= c_j1
        mdl.addConstr(s[(j1, o1)] >= c[(j2, o2)] - BIG_M * (1 - y)
                       - BIG_M * (1 - x.get((j1, o1, m), 0))
                       - BIG_M * (1 - x.get((j2, o2, m), 0)),
                       name=f"seq1_{j1}_{o1}_{j2}_{o2}_{m}")
        mdl.addConstr(s[(j2, o2)] >= c[(j1, o1)] - BIG_M * y
                       - BIG_M * (1 - x.get((j1, o1, m), 0))
                       - BIG_M * (1 - x.get((j2, o2, m), 0)),
                       name=f"seq2_{j1}_{o1}_{j2}_{o2}_{m}")

# 5. Makespan definition
for j in jobs:
    # last operation index for job j
    last_op_idx = max(o for (j2, o, _) in operations if j2 == j)
    mdl.addConstr(Cmax >= c[(j, last_op_idx)], name=f"mksp_{j}")

# Objective ------------------------------------------------------------------
mdl.setObjective(Cmax, GRB.MINIMIZE)

# ---------------------------------------------------------------------------
# Optimise (NO explicit time limit)
# ---------------------------------------------------------------------------
mdl.optimize()

# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
if mdl.Status == GRB.OPTIMAL:
    print(f"\nOptimal makespan: {Cmax.X:.2f}")
    print("Schedule:")
    for j, o, ops in sorted(operations):
        m_used = next(m for m, _ in ops if x[(j, o, m)].X > 0.5)
        dur = next(d for m, d in ops if m == m_used)
        print(f"  Job {j} – Op {o}: Machine {m_used} | Start {s[(j, o)].X:.2f} | Duration {dur} | Finish {c[(j, o)].X:.2f}")
else:
    print("Solver did not reach optimality (status", mdl.Status, ")")
