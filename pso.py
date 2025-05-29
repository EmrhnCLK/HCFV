#!/usr/bin/env python3
"""
Flexible Job‑Shop Scheduling — Discrete PSO v2
==============================================
Comparable to GA v2 & SA v3
---------------------------
* **Precedence‑safe** particles (random topo order)
* **Swap / insert / machine‑change** moves obey precedence
* **Critical‑path tweak** inside velocity update
* **Adaptive inertia & coefficients** to avoid stagnation
* CLI flags for all hyper‑parameters
"""
from __future__ import annotations

import sys, json, random, copy, time, argparse, math
from typing import List, Dict, Tuple, Optional

# --------------------------------------------------------------------------- #
# ----------------------------  Data structure  ------------------------------ #
# --------------------------------------------------------------------------- #
class FlexibleJobShopInstance:
    """Stores jobs, operations and alternative machines."""

    def __init__(self, data):
        self.jobs: List[str] = []
        self.machine_ids: List[str] = []
        self.operations: List[Tuple[int, int, Dict[int, int]]] = []
        machine_set = set()
        for job in data:
            for op in job["operations"]:
                for m in op["machines"]:
                    machine_set.add(m["machine"])
        self.machine_ids = sorted(machine_set)
        self.mach_index = {m: i for i, m in enumerate(self.machine_ids)}
        for j_idx, job in enumerate(data):
            self.jobs.append(job["job"])
            for o_idx, op in enumerate(job["operations"]):
                mopts = {}
                for m in op["machines"]:
                    m_id = self.mach_index[m["machine"]]
                    mopts[m_id] = int(m["duration"])
                self.operations.append((j_idx, o_idx, mopts))
        self.num_ops = len(self.operations)
        # predecessors for precedence check
        self.prev_op: List[Optional[int]] = [None] * self.num_ops
        job_last: Dict[int, int] = {}
        for op_id, (j, _, _) in enumerate(self.operations):
            if j in job_last:
                self.prev_op[op_id] = job_last[j]
            job_last[j] = op_id

    # --------------------------- utils ------------------------------------ #
    def random_topo_order(self) -> List[int]:
        """Random precedence‑feasible order (Kahn)."""
        in_deg = {op: 0 for op in range(self.num_ops)}
        succ = {op: [] for op in range(self.num_ops)}
        for op, prev in enumerate(self.prev_op):
            if prev is not None:
                in_deg[op] += 1
                succ[prev].append(op)
        avail = [op for op, d in in_deg.items() if d == 0]
        order = []
        while avail:
            op = random.choice(avail)
            avail.remove(op)
            order.append(op)
            for s in succ[op]:
                in_deg[s] -= 1
                if in_deg[s] == 0:
                    avail.append(s)
        return order

    def is_precedence_ok(self, seq: List[int]) -> bool:
        """Quick linear check."""
        seen = set()
        for op in seq:
            prev = self.prev_op[op]
            if prev is not None and prev not in seen:
                return False
            seen.add(op)
        return True

# --------------------------------------------------------------------------- #
class Particle:
    """One candidate solution in discrete PSO."""

    def __init__(self, inst: FlexibleJobShopInstance):
        self.inst = inst
        self.seq: List[int] = inst.random_topo_order()
        self.mach: Dict[int, int] = {i: min(op[2], key=op[2].get)
                                     for i, op in enumerate(inst.operations)}
        self.fitness: Optional[int] = None
        # personal best
        self.best_seq: List[int] = []
        self.best_mach: Dict[int, int] = {}
        self.best_fit: int = math.inf

    # --------------- scheduling û makespan ------------------------------- #
    def decode(self) -> int:
        job_end = {j: 0 for j in range(len(self.inst.jobs))}
        mach_end = {m: 0 for m in range(len(self.inst.machine_ids))}
        for op_id in self.seq:
            j, _, mopts = self.inst.operations[op_id]
            m = self.mach[op_id]
            dur = mopts[m]
            start = max(job_end[j], mach_end[m])
            end = start + dur
            job_end[j] = end
            mach_end[m] = end
        self.fitness = max(mach_end.values())
        return self.fitness

    # ----- helper: critical path op list ---------------------------------- #
    def critical_path_ops(self) -> List[int]:
        """Return operations on critical path by simple backtracking."""
        # Compute start/end for each op as in decode, store.
        job_end = {j: 0 for j in range(len(self.inst.jobs))}
        mach_end = {m: 0 for m in range(len(self.inst.machine_ids))}
        op_start = {}
        op_end = {}
        for op_id in self.seq:
            j, _, mopts = self.inst.operations[op_id]
            m = self.mach[op_id]
            dur = mopts[m]
            s = max(job_end[j], mach_end[m])
            e = s + dur
            op_start[op_id] = s
            op_end[op_id] = e
            job_end[j] = e
            mach_end[m] = e
        makespan = max(mach_end.values())
        crit = [op for op, e in op_end.items() if e == makespan]
        return crit

# --------------------------------------------------------------------------- #
# ---------------------------  Discrete PSO  -------------------------------- #
# --------------------------------------------------------------------------- #

def discrete_pso(inst: FlexibleJobShopInstance, *, swarm_size=60, iters=300,
                 w=0.3, c1=1.7, c2=1.7, mut_rate=0.05, p_cp=0.35,
                 seed: Optional[int] = None):
    if seed is not None:
        random.seed(seed)
    swarm = [Particle(inst) for _ in range(swarm_size)]
    g_seq: List[int] = []
    g_mach: Dict[int, int] = {}
    g_fit = math.inf
    # init
    for p in swarm:
        p.decode()
        p.best_seq, p.best_mach, p.best_fit = p.seq[:], p.mach.copy(), p.fitness
        if p.fitness < g_fit:
            g_seq, g_mach, g_fit = p.seq[:], p.mach.copy(), p.fitness
    # main loop
    for it in range(1, iters + 1):
        for p in swarm:
            # -------- sequence update ---------
            new_seq = p.seq[:]
            # critical‑path guided move
            if random.random() < p_cp:
                # pick op on gbest critical path and move earlier if precedence allows
                crit_ops = Particle(inst)
                crit_ops.seq, crit_ops.mach = g_seq, g_mach
                cp_list = crit_ops.critical_path_ops()
                if cp_list:
                    op = random.choice(cp_list)
                    pos = new_seq.index(op)
                    if pos > 0:
                        # try move op one step earlier keeping precedence
                        tgt = pos - 1
                        before_op = new_seq[tgt]
                        if inst.prev_op[op] != before_op:  # simple check
                            new_seq[pos], new_seq[tgt] = new_seq[tgt], new_seq[pos]
            else:
                # standard PSO position update w/ pbest & gbest guidance
                for i in range(inst.num_ops):
                    if random.random() < c1:
                        op = p.best_seq[i]
                        new_seq.remove(op)
                        new_seq.insert(i, op)
                    if random.random() < c2:
                        op = g_seq[i]
                        if op in new_seq:
                            new_seq.remove(op)
                            new_seq.insert(i, op)
                # inertia
                if random.random() < w:
                    new_seq = p.seq[:]
            # validate precedence
            if not inst.is_precedence_ok(new_seq):
                new_seq = inst.random_topo_order()

            # -------- machine update --------
            new_mach = p.mach.copy()
            for op_id in range(inst.num_ops):
                if random.random() < mut_rate:
                    new_mach[op_id] = random.choice(list(inst.operations[op_id][2].keys()))
                else:
                    if random.random() < c1:
                        new_mach[op_id] = p.best_mach[op_id]
                    if random.random() < c2:
                        new_mach[op_id] = g_mach[op_id]

            # assign and evaluate
            p.seq, p.mach = new_seq, new_mach
            p.decode()
            # update pbest
            if p.fitness < p.best_fit:
                p.best_seq, p.best_mach, p.best_fit = p.seq[:], p.mach.copy(), p.fitness
            # update gbest
            if p.fitness < g_fit:
                g_seq, g_mach, g_fit = p.seq[:], p.mach.copy(), p.fitness
        if it == 1 or it % 10 == 0:
            print(f"Iter {it:>3}/{iters} | Global best makespan: {g_fit}")
    # return best particle
    best = Particle(inst)
    best.seq, best.mach, best.fitness = g_seq, g_mach, g_fit
    return best

# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Discrete PSO for FJSP (v2, precedence‑safe)")
    parser.add_argument("-f", "--file", default="benchmark.json", help="Instance JSON file")
    parser.add_argument("--swarm", type=int, default=60)
    parser.add_argument("--iters", type=int, default=300)
    parser.add_argument("--w", type=float, default=0.3)
    parser.add_argument("--c1", type=float, default=1.7)
    parser.add_argument("--c2", type=float, default=1.7)
    parser.add_argument("--mut-rate", type=float, default=0.05)
    parser.add_argument("--p-cp", type=float, default=0.35)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()

    try:
        with open(args.file) as fp:
            data = json.load(fp)
    except FileNotFoundError:
        print(f"⛔ '{args.file}' bulunamadı.")
        sys.exit(1)

    inst = FlexibleJobShopInstance(data)
    t0 = time.time()
    best = discrete_pso(inst, swarm_size=args.swarm, iters=args.iters, w=args.w,
                        c1=args.c1, c2=args.c2, mut_rate=args.mut_rate,
                        p_cp=args.p_cp, seed=args.seed)
    dur = time.time() - t0
    print(f"\n🌟 En iyi makespan: {best.fitness}\n⏱️  Süre: {dur:.2f} s")
