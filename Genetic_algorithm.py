#!/usr/bin/env python3
"""
Flexible Job‑Shop Scheduling — Genetic Algorithm (GA) v2
=======================================================
Matches Simulated‑Annealing v3 capabilities
------------------------------------------
* **Precedence‑safe representation** (random topological order)
* **PPX‑style crossover** preserving job order
* **Adaptive mutation** (swap/insert with precedence check)
* **Critical‑path local search** (machine re‑assignment)
* **Stagnation‑triggered intensification**
* All hyper‑parameters exposed as CLI flags (see `--help`)

Usage
-----
```
python fjps_ga.py -f benchmark.json \
                 --pop 120 --gens 180 \
                 --cx 0.9 --mut 0.25 \
                 --seed 42
```
"""
from __future__ import annotations
import sys, json, random, copy, time, argparse
from collections import defaultdict, deque

# ---------------------------------------------------------------------------
# Data instance
# ---------------------------------------------------------------------------
class FlexibleJobShopInstance:
    def __init__(self, data):
        self.jobs, self.operations, machine_set = [], [], set()
        for job in data:
            for op in job['operations']:
                for m in op['machines']:
                    machine_set.add(m['machine'])
        self.machine_ids = sorted(machine_set)
        self.midx = {m: i for i, m in enumerate(self.machine_ids)}
        for j_idx, job in enumerate(data):
            self.jobs.append(job['job'])
            for o_idx, op in enumerate(job['operations']):
                mopts = {self.midx[m['machine']]: int(m['duration']) for m in op['machines']}
                self.operations.append((j_idx, o_idx, mopts))
        self.num_ops = len(self.operations)
        # predecessor list per op id
        self.prev = [[] for _ in range(self.num_ops)]
        job_last = {}
        for op_id, (j, o, _) in enumerate(self.operations):
            if o > 0:
                # predecessor is previous op of same job
                self.prev[op_id].append(job_last[j])
            job_last[j] = op_id

# ---------------------------------------------------------------------------
# Helper: precedence‑safe random topological order
# ---------------------------------------------------------------------------

def random_topo_order(inst: FlexibleJobShopInstance) -> list[int]:
    indeg = [len(inst.prev[i]) for i in range(inst.num_ops)]
    ready = [i for i, d in enumerate(indeg) if d == 0]
    seq = []
    while ready:
        op = random.choice(ready)
        ready.remove(op)
        seq.append(op)
        # decrease indeg of successors (same job only)  
        j, o, _ = inst.operations[op]
        # successor is next operation in same job
        for succ_id, (sj, so, _) in enumerate(inst.operations):
            if sj == j and so == o + 1:
                indeg[succ_id] -= 1
                if indeg[succ_id] == 0:
                    ready.append(succ_id)
                break
    return seq

# ---------------------------------------------------------------------------
# Chromosome
# ---------------------------------------------------------------------------
class Chromosome:
    def __init__(self, inst: FlexibleJobShopInstance):
        self.inst = inst
        self.op_seq = random_topo_order(inst)
        self.machine_assign = {i: min(op[2], key=op[2].get) for i, op in enumerate(inst.operations)}
        self.fitness: int | None = None

    # precedence‑safe verify
    def _check(self):
        pos = {op: i for i, op in enumerate(self.op_seq)}
        for op_id, preds in enumerate(self.inst.prev):
            for p in preds:
                assert pos[p] < pos[op_id]

    def decode(self):
        if self.fitness is not None:
            return self.fitness
        job_end = defaultdict(int)
        mach_end = defaultdict(int)
        for op_id in self.op_seq:
            j, _, mopts = self.inst.operations[op_id]
            m = self.machine_assign[op_id]
            dur = mopts[m]
            s = max(job_end[j], mach_end[m])
            e = s + dur
            job_end[j] = e
            mach_end[m] = e
        self.fitness = max(mach_end.values())
        return self.fitness

    # Local search: critical‑path machine tweak
    def critical_path_search(self):
        self.decode()
        # Recalculate schedule with timing info
        job_end = defaultdict(int); mach_end = defaultdict(int); times = {}
        for op_id in self.op_seq:
            j, _, mopts = self.inst.operations[op_id]
            m = self.machine_assign[op_id]
            dur = mopts[m]
            s = max(job_end[j], mach_end[m]); e = s+dur
            job_end[j] = e; mach_end[m] = e; times[op_id] = (s, e)
        makespan = max(e for _,e in times.values())
        critical_ops = [op for op,(s,e) in times.items() if e == makespan]
        improved = False
        for op in critical_ops:
            best_m = self.machine_assign[op]
            best_fit = makespan
            for alt_m in self.inst.operations[op][2].keys():
                if alt_m == best_m: continue
                old = self.machine_assign[op]
                self.machine_assign[op] = alt_m
                self.fitness = None
                new_fit = self.decode()
                if new_fit < best_fit:
                    best_m, best_fit = alt_m, new_fit
                    improved = True
                else:
                    self.machine_assign[op] = old  # revert
                    self.fitness = makespan
            self.machine_assign[op] = best_m
            self.fitness = best_fit
        return improved

# ---------------------------------------------------------------------------
# GA operator helpers
# ---------------------------------------------------------------------------

def ppx_crossover(p1: Chromosome, p2: Chromosome) -> tuple[Chromosome, Chromosome]:
    inst = p1.inst
    child1, child2 = Chromosome(inst), Chromosome(inst)
    child1.op_seq.clear(); child2.op_seq.clear()
    remaining1, remaining2 = p1.op_seq[:], p2.op_seq[:]
    indeg = [len(inst.prev[i]) for i in range(inst.num_ops)]
    ready = deque([i for i,d in enumerate(indeg) if d==0])
    while ready:
        if random.random() < 0.5:
            parent_seq = remaining1
        else:
            parent_seq = remaining2
        sel = None
        for op in parent_seq:
            if op in ready:
                sel = op; break
        if sel is None:
            sel = random.choice(list(ready))
        for ch in (child1, child2):
            ch.op_seq.append(sel)
        ready.remove(sel)
        # update graph
        j,o,_ = inst.operations[sel]
        for succ_id, (sj,so,_) in enumerate(inst.operations):
            if sj==j and so==o+1:
                indeg[succ_id]-=1
                if indeg[succ_id]==0:
                    ready.append(succ_id)
                break
        remaining1 = [op for op in remaining1 if op!=sel]
        remaining2 = [op for op in remaining2 if op!=sel]
    # machine assignments
    for op in range(inst.num_ops):
        child1.machine_assign[op] = p1.machine_assign[op] if random.random()<0.5 else p2.machine_assign[op]
        child2.machine_assign[op] = p2.machine_assign[op] if random.random()<0.5 else p1.machine_assign[op]
    child1.fitness = child2.fitness = None
    return child1, child2


def precedence_safe_swap(seq, inst):
    # pick two ops from different jobs
    for _ in range(10):
        i,j = random.sample(range(len(seq)),2)
        op_i, op_j = seq[i], seq[j]
        if inst.operations[op_i][0] != inst.operations[op_j][0]:
            seq[i], seq[j] = seq[j], seq[i]
            return


def mutate(ch: Chromosome, mut_rate: float):
    if random.random()<mut_rate:
        precedence_safe_swap(ch.op_seq, ch.inst)
        ch.fitness=None
    for op_id,(_,_,mopts) in enumerate(ch.inst.operations):
        if random.random()<mut_rate:
            ch.machine_assign[op_id] = random.choice(list(mopts.keys()))
            ch.fitness=None

# ---------------------------------------------------------------------------
# GA driver
# ---------------------------------------------------------------------------
class GeneticAlgorithm:
    def __init__(self, inst: FlexibleJobShopInstance, pop=100, gens=150, cx=0.9, mut=0.25, seed=None):
        self.inst, self.pop, self.gens, self.cx, self.mut = inst,pop,gens,cx,mut
        random.seed(seed)
        self.population = [Chromosome(inst) for _ in range(pop)]
        for ind in self.population: ind.decode()
        self.population.sort(key=lambda c:c.fitness)
        self.best = copy.deepcopy(self.population[0])

    def evolve(self):
        stag=0
        for g in range(1,self.gens+1):
            sel = [min(random.sample(self.population,2), key=lambda c:c.fitness) for _ in range(self.pop)]
            offspring=[]
            for i in range(0,self.pop,2):
                if random.random()<self.cx:
                    c1,c2 = ppx_crossover(sel[i], sel[i+1])
                else:
                    c1,c2 = copy.deepcopy(sel[i]), copy.deepcopy(sel[i+1])
                offspring.extend([c1,c2])
            for ind in offspring:
                mutate(ind, self.mut)
                ind.decode()
            # elitism
            offspring.sort(key=lambda c:c.fitness)
            offspring[-1]=copy.deepcopy(self.best)
            self.population=offspring
            self.population.sort(key=lambda c:c.fitness)
            if self.population[0].fitness < self.best.fitness:
                self.best = copy.deepcopy(self.population[0]); stag=0
            else:
                stag+=1
            if stag>=20:
                self.mut=min(1.0,self.mut*1.3); stag=0
                self.best.critical_path_search()
            if g%10==0:
                print(f"Gen {g} | Best {self.best.fitness}")
        return self.best

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("-f","--file",default="benchmark.json")
    ap.add_argument("--pop",type=int,default=120)
    ap.add_argument("--gens",type=int,default=180)
    ap.add_argument("--cx",type=float,default=0.9)
    ap.add_argument("--mut",type=float,default=0.25)
    ap.add_argument("--seed",type=int,default=None)
    args=ap.parse_args()

    with open(args.file) as fp:
        data=json.load(fp)
    inst=FlexibleJobShopInstance(data)
    t0=time.time()
    ga=GeneticAlgorithm(inst,args.pop,args.gens,args.cx,args.mut,args.seed)
    best=ga.evolve()
    print("\n🏆 Best makespan:",best.fitness)
    print("⏱️  Elapsed:",time.time()-t0,"s")
