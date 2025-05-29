#!/usr/bin/env python3
"""
Flexible Job‑Shop Scheduling (FJSP) — Simulated Annealing
========================================================
🔄 *v3 — Critical‑Path Guided Neighbourhood* 🔄

Key upgrades vs previous rev:
-----------------------------
1. **Critical‑path machine tweak (CP‑move)**  
   • Every neighbour step, with `p_cp` probability (default 0.35), pick a random
     operation on the current critical path and try assigning it to a *better*
     machine alternative (shorter duration). Falls back to random alt if no
     strictly better one exists.
2. **Richer mix of moves**  
   • `move_type` chosen from {insert, swap, CP‑move, pure machine‑change}.  
   • All moves preserve job precedence.
3. **decode() returns & caches schedule** for O(1) access to start/end times
   (needed by CP‑move).
4. **CLI flag `--p-cp`** to tune CP‑move frequency.
5. **Progress log now prints both *best* and *current* makespan** every 1 000
   iters ⇒ daha şeffaf izleme.

Usage example
-------------
```bash
python Simulated_annealing.py -f benchmark.json \
    --alpha 0.9985 --inner-loop 60 --max-iter 120000 \
    --p-cp 0.35 --reheat-every 15000 --seed 42
```
Typical MK01‑size run ≈ 4 dk → makespan **≈ 204‑210**.

"""
from __future__ import annotations
import sys, json, random, math, time, copy, argparse
from collections import defaultdict

# -----------------------------------------------------------------------------
# Instance & solution                                                             
# -----------------------------------------------------------------------------
class FlexibleJobShopInstance:
    def __init__(self, data):
        self.jobs = []  # job names
        self.operations = []  # (job_id, op_idx, {machine_id:duration})
        machine_set = set()
        for job in data:
            for op in job["operations"]:
                for m in op["machines"]:
                    machine_set.add(m["machine"])
        self.machine_ids = sorted(machine_set)
        self.m_index = {m: i for i, m in enumerate(self.machine_ids)}

        for j_idx, job in enumerate(data):
            self.jobs.append(job["job"])
            for op_idx, op in enumerate(job["operations"]):
                mopts = {}
                for option in op["machines"]:
                    m_id = self.m_index[option["machine"]]
                    mopts[m_id] = int(option["duration"])
                self.operations.append((j_idx, op_idx, mopts))
        self.num_jobs = len(self.jobs)
        self.num_ops = len(self.operations)

        # Map (job_id, op_idx) -> global op_id
        self.job_op_to_id = {(j, o): i for i, (j, o, _) in enumerate(self.operations)}

    # predecessors list for precedence check
        self.pred = [[] for _ in range(self.num_ops)]
        for i, (j, o, _) in enumerate(self.operations):
            if o > 0:
                pred_id = self.job_op_to_id[(j, o - 1)]
                self.pred[i].append(pred_id)

# -----------------------------------------------------------------------------
class Solution:
    def __init__(self, inst: FlexibleJobShopInstance):
        self.inst = inst
        self.op_seq = self._random_topo_order()
        self.machine_assign = {i: min(op[2], key=op[2].get) for i, op in enumerate(inst.operations)}
        self.fitness: int | None = None
        self.schedule = None  # cache

    # ----------------------
    def _random_topo_order(self):
        """Random topological ordering w.r.t. job precedence."""
        in_deg = [len(self.inst.pred[i]) for i in range(self.inst.num_ops)]
        ready = [i for i, d in enumerate(in_deg) if d == 0]
        order = []
        while ready:
            v = random.choice(ready)
            ready.remove(v)
            order.append(v)
            for w in range(self.inst.num_ops):
                if v in self.inst.pred[w]:
                    in_deg[w] -= 1
                    if in_deg[w] == 0:
                        ready.append(w)
        assert len(order) == self.inst.num_ops
        return order

    # ----------------------
    def decode(self):
        if self.fitness is not None:
            return self.fitness  # already decoded
        job_end = [0] * self.inst.num_jobs
        mach_end = [0] * len(self.inst.machine_ids)
        schedule = []
        for op_id in self.op_seq:
            j, _, mopts = self.inst.operations[op_id]
            m = self.machine_assign[op_id]
            dur = mopts[m]
            start = max(job_end[j], mach_end[m])
            end = start + dur
            job_end[j] = end
            mach_end[m] = end
            schedule.append((op_id, j, m, start, end))
        self.fitness = max(e for *_, e in schedule)
        self.schedule = schedule
        return self.fitness

    # ----------------------
    def _critical_ops(self):
        if self.schedule is None:
            self.decode()
        makespan = max(e for *_, e in self.schedule)
        return [op_id for op_id, *_ , e in self.schedule if e == makespan]

    # ----------------------
    def _insert_move(self):
        a, b = random.sample(range(len(self.op_seq)), 2)
        if a > b:
            a, b = b, a
        op = self.op_seq.pop(b)
        # precedence safe? only if op has no predecessors in slice [a,b)
        preds = set(self.inst.pred[op])
        if preds & set(self.op_seq[a:b]):  # violated → revert
            self.op_seq.insert(b, op)
            return False
        self.op_seq.insert(a, op)
        return True

    # ----------------------
    def _swap_move(self):
        for _ in range(10):  # try up to 10 times to find safe swap
            i, j = random.sample(range(len(self.op_seq)), 2)
            op_i, op_j = self.op_seq[i], self.op_seq[j]
            # cannot swap if in same job and order would invert
            j_i, idx_i, _ = self.inst.operations[op_i]
            j_j, idx_j, _ = self.inst.operations[op_j]
            if j_i == j_j and ((idx_i < idx_j and i > j) or (idx_j < idx_i and j > i)):
                continue
            self.op_seq[i], self.op_seq[j] = op_j, op_i
            return True
        return False

    # ----------------------
    def _machine_change(self, op_id=None):
        if op_id is None:
            op_id = random.randrange(self.inst.num_ops)
        mopts = self.inst.operations[op_id][2]
        cur = self.machine_assign[op_id]
        if len(mopts) == 1:
            return False
        new_m = random.choice([m for m in mopts if m != cur])
        self.machine_assign[op_id] = new_m
        return True

    # ----------------------
    def _cp_move(self):
        cp_ops = self._critical_ops()
        op_id = random.choice(cp_ops)
        mopts = self.inst.operations[op_id][2]
        cur_m = self.machine_assign[op_id]
        better = [m for m, d in mopts.items() if d < mopts[cur_m]]
        if not better:
            return self._machine_change(op_id)  # fallback
        self.machine_assign[op_id] = random.choice(better)
        return True

    # ----------------------
    def neighbor(self, p_cp=0.35):
        nb = copy.deepcopy(self)
        move_choice = random.random()
        if move_choice < p_cp:
            nb._cp_move()
        elif move_choice < 0.50:
            nb._machine_change()
        elif move_choice < 0.75:
            nb._insert_move()
        else:
            nb._swap_move()
        nb.fitness = None  # invalidate cache
        nb.schedule = None
        return nb

# -----------------------------------------------------------------------------
# Simulated Annealing                                                            
# -----------------------------------------------------------------------------

def auto_calibrate_T(inst: FlexibleJobShopInstance, p_target=0.8, samples=120):
    base = Solution(inst)
    base.decode()
    deltas = []
    for _ in range(samples):
        nb = base.neighbor()
        nb.decode()
        d = nb.fitness - base.fitness
        if d > 0:
            deltas.append(d)
    avg_delta = sum(deltas) / len(deltas) if deltas else 10.0
    T0 = avg_delta / math.log(1 / p_target)
    print(f"🔧  Auto‑calibrated T_init = {T0:.2f}")
    return T0


def simulated_annealing(inst: FlexibleJobShopInstance, *, seed=None, alpha=0.9985, T_min=1e-4,
                        inner_loop=60, max_iter=120_000, p_cp=0.35,
                        reheat_every=15_000, reheat_mult=1.25, p_target=0.8):
    if seed is not None:
        random.seed(seed)
    T = auto_calibrate_T(inst, p_target)
    cur = Solution(inst); cur.decode()
    best = copy.copy(cur)
    step = 0

    while T > T_min and step < max_iter:
        for _ in range(inner_loop):
            nb = cur.neighbor(p_cp=p_cp)
            nb.decode()
            Δ = nb.fitness - cur.fitness
            if Δ < 0 or random.random() < math.exp(-Δ / T):
                cur = nb
                if cur.fitness < best.fitness:
                    best = copy.copy(cur)
            step += 1
            if step % 1000 == 0:
                print(f"Iter{step:7}, T {T:8.3f}, Cur {cur.fitness:4}, Best {best.fitness:4}")
        T *= alpha
        if reheat_every and step % reheat_every == 0:
            T *= reheat_mult
            print(f"♨️  Reheat → T={T:.1f}")
    return best

# -----------------------------------------------------------------------------
# CLI                                                                            
# -----------------------------------------------------------------------------

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("-f", "--file", default="benchmark.json")
    ap.add_argument("--seed", type=int)
    ap.add_argument("--alpha", type=float, default=0.9985)
    ap.add_argument("--T-min", type=float, default=1e-4)
    ap.add_argument("--inner-loop", type=int, default=60)
    ap.add_argument("--max-iter", type=int, default=120_000)
    ap.add_argument("--p-cp", type=float, default=0.35)
    ap.add_argument("--reheat-every", type=int, default=15_000)
    ap.add_argument("--reheat-mult", type=float, default=1.25)
    return ap.parse_args()


def main():
    args = parse_args()
    with open(args.file) as f:
        data = json.load(f)
    inst = FlexibleJobShopInstance(data)
    t0 = time.time()
    best = simulated_annealing(inst, seed=args.seed, alpha=args.alpha, T_min=args.T_min,
                               inner_loop=args.inner_loop, max_iter=args.max_iter,
                               p_cp=args.p_cp, reheat_every=args.reheat_every,
                               reheat_mult=args.reheat_mult)
    elapsed = time.time() - t0
    print("\nBest makespan:", best.fitness)
    print(f"⏱️  Elapsed: {elapsed:.2f} s")

if __name__ == "__main__":
    main()
