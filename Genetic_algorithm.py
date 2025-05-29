#!/usr/bin/env python3
"""
Flexible Job‑Shop Scheduling — Genetic Algorithm (GA) v5.1
========================================================
* Memetic + Multi-Elite Path-Relinking + Tabu Swap + Self-Adaptive Parameters + Random-Restart Inner Search *

Usage:
------
```bash
python ga.py [-f benchmark.json] [--pop POP] [--gens GENS] \
             [--cx CX] [--mut MUT] [--seed SEED]
```

This version integrates multiple advanced intensification strategies:
- **PPX-style crossover** and **precedence-safe swap** mutation
- **Critical-path local search** on new bests and elite individuals
- **Multi-elite path-relinking** among top 5 solutions
- **Tabu-based swaps** on critical path
- **Self-adaptive** CX/MUT per chromosome
- **Random-restart** inner local search on stagnation

"""
from __future__ import annotations
import sys, json, random, copy, time, argparse, pathlib
from collections import defaultdict, deque

# ---------------------------------------------------------------------------
# Instance loader
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
        self.prev = [[] for _ in range(self.num_ops)]
        job_last = {}
        for op_id, (j, o, _) in enumerate(self.operations):
            if o > 0:
                self.prev[op_id].append(job_last[j])
            job_last[j] = op_id

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def random_topo_order(inst: FlexibleJobShopInstance) -> list[int]:
    indeg = [len(inst.prev[i]) for i in range(inst.num_ops)]
    ready = [i for i, d in enumerate(indeg) if d == 0]
    seq = []
    while ready:
        op = random.choice(ready)
        ready.remove(op)
        seq.append(op)
        j, o, _ = inst.operations[op]
        for succ_id, (sj, so, _) in enumerate(inst.operations):
            if sj == j and so == o + 1:
                indeg[succ_id] -= 1
                if indeg[succ_id] == 0:
                    ready.append(succ_id)
                break
    return seq

# ---------------------------------------------------------------------------
# Chromosome definition
# ---------------------------------------------------------------------------
class Chromosome:
    def __init__(self, inst: FlexibleJobShopInstance, cx=0.9, mut_rate=0.25):
        self.inst = inst
        self.op_seq = random_topo_order(inst)
        self.machine_assign = {i: min(op[2], key=op[2].get) for i, op in enumerate(inst.operations)}
        self.cx = cx
        self.mut_rate = mut_rate
        self.fitness: int | None = None

    def decode(self) -> int:
        if self.fitness is not None:
            return self.fitness
        job_end = defaultdict(int)
        mach_end = defaultdict(int)
        for op_id in self.op_seq:
            j, _, mopts = self.inst.operations[op_id]
            m = self.machine_assign[op_id]
            dur = mopts[m]
            s = max(job_end[j], mach_end[m]); e = s + dur
            job_end[j] = e; mach_end[m] = e
        self.fitness = max(mach_end.values())
        return self.fitness

    def export(self) -> dict:
        seqs = self._build_sequences()
        machine_sequences = {self.inst.machine_ids[m]: seqs[m] for m in seqs}
        times = self._times()
        return {
            "makespan": self.decode(),
            "machine_sequences": machine_sequences,
            "start": {str(op): t0 for op,(t0,_) in times.items()},
            "duration": {str(op): dur for op,(_,dur) in times.items()}
        }

    def _times(self) -> dict[int,tuple[int,int]]:
        job_end = defaultdict(int); mach_end = defaultdict(int)
        times = {}
        for op_id in self.op_seq:
            j, _, mopts = self.inst.operations[op_id]
            m = self.machine_assign[op_id]; dur = mopts[m]
            s = max(job_end[j], mach_end[m]); e = s + dur
            job_end[j] = e; mach_end[m] = e; times[op_id] = (s,dur)
        return times

    def _build_sequences(self) -> dict[int,list[int]]:
        seqs = {i:[] for i in range(len(self.inst.machine_ids))}
        for op_id in self.op_seq:
            seqs[self.machine_assign[op_id]].append(op_id)
        return seqs

    def critical_path_search(self) -> bool:
        self.decode()
        # compute times and critical ops
        job_end = defaultdict(int); mach_end = defaultdict(int); times = {}
        for op_id in self.op_seq:
            j, _, mopts = self.inst.operations[op_id]
            m = self.machine_assign[op_id]; dur = mopts[m]
            s = max(job_end[j], mach_end[m]); e = s+dur
            job_end[j]=e; mach_end[m]=e; times[op_id] = (s,e)
        makespan = max(e for (_,e) in times.values())
        crit_ops = [op for op,(s,e) in times.items() if e==makespan]
        improved=False
        for op in crit_ops:
            best_m = self.machine_assign[op]; best_fit=makespan
            for alt in self.inst.operations[op][2].keys():
                if alt==best_m: continue
                old_m = self.machine_assign[op]
                self.machine_assign[op]=alt; self.fitness=None
                new_fit = self.decode()
                if new_fit<best_fit:
                    best_fit, best_m=new_fit,alt; improved=True
                else:
                    self.machine_assign[op]=old_m; self.fitness=makespan
            self.machine_assign[op]=best_m; self.fitness=best_fit
        return improved

    def critical_path_tabu(self, tabu_size=50) -> bool:
            self.decode()
            # compute current makespan and critical operations
            times = self._times()  # {op: (start, dur)}
            ends = {op: s+dur for op,(s,dur) in times.items()}
            makespan = self.decode()
            crit = [op for op,end in ends.items() if end == makespan]
            attempts = 0
            while attempts < tabu_size and len(crit) >= 2:
                i,j = random.sample(crit,2)
                m_i = self.machine_assign[i]; m_j = self.machine_assign[j]
                # check validity: both ops can run on each other's machines
                if m_j in self.inst.operations[i][2] and m_i in self.inst.operations[j][2]:
                    # swap assignments
                    self.machine_assign[i], self.machine_assign[j] = m_j, m_i
                    self.fitness = None
                    new_fit = self.decode()
                    if new_fit < makespan:
                        return True
                    # revert
                    self.machine_assign[i], self.machine_assign[j] = m_i, m_j
                    self.fitness = makespan
                    return False
                attempts += 1
            return False

# ---------------------------------------------------------------------------
# GA Operators
# ---------------------------------------------------------------------------
def ppx_crossover(p1:Chromosome,p2:Chromosome) -> tuple[Chromosome,Chromosome]:
    inst=p1.inst
    c1,c2=Chromosome(inst,p1.cx,p1.mut_rate),Chromosome(inst,p2.cx,p2.mut_rate)
    c1.op_seq.clear();c2.op_seq.clear()
    rem1,rem2=p1.op_seq[:],p2.op_seq[:]
    indeg=[len(inst.prev[i]) for i in range(inst.num_ops)]
    ready=deque([i for i,d in enumerate(indeg) if d==0])
    while ready:
        parent=rem1 if random.random()<0.5 else rem2
        sel=next((op for op in parent if op in ready),random.choice(ready))
        c1.op_seq.append(sel);c2.op_seq.append(sel)
        ready.remove(sel)
        j,o,_=inst.operations[sel]
        for sid,(sj,so,_) in enumerate(inst.operations):
            if sj==j and so==o+1:
                indeg[sid]-=1
                if indeg[sid]==0: ready.append(sid)
                break
        rem1=[op for op in rem1 if op!=sel]; rem2=[op for op in rem2 if op!=sel]
    # param adaptation
    c1.cx=(p1.cx+p2.cx)/2+random.uniform(-0.05,0.05)
    c2.cx=(p1.cx+p2.cx)/2+random.uniform(-0.05,0.05)
    for op in range(inst.num_ops):
        c1.machine_assign[op]=p1.machine_assign[op] if random.random()<0.5 else p2.machine_assign[op]
        c2.machine_assign[op]=p2.machine_assign[op] if random.random()<0.5 else p1.machine_assign[op]
    # mutate rates
    c1.mut_rate=min(1.0,(p1.mut_rate+p2.mut_rate)/2+random.uniform(-0.05,0.05))
    c2.mut_rate=min(1.0,(p1.mut_rate+p2.mut_rate)/2+random.uniform(-0.05,0.05))
    c1.fitness=c2.fitness=None
    return c1,c2

def precedence_safe_swap(seq:list[int],inst:FlexibleJobShopInstance)->None:
    for _ in range(10):
        i,j=random.sample(range(len(seq)),2)
        if inst.operations[seq[i]][0]!=inst.operations[seq[j]][0]:
            seq[i],seq[j]=seq[j],seq[i]; return

def mutate(ch:Chromosome,mut_rate:float)->None:
    rate=ch.mut_rate
    if random.random()<rate:
        precedence_safe_swap(ch.op_seq,ch.inst)
        ch.fitness=None
    for op in range(ch.inst.num_ops):
        if random.random()<rate:
            opts=ch.inst.operations[op][2]; ch.machine_assign[op]=random.choice(list(opts.keys())); ch.fitness=None

# ---------------------------------------------------------------------------
# Path-Relinking
# ---------------------------------------------------------------------------
def path_relink(p1:Chromosome,p2:Chromosome)->tuple[list[int],dict[int,int],int]:
    inst=p1.inst; best_seq=p1.op_seq.copy(); best_assign=p1.machine_assign.copy(); best_fit=p1.decode()
    for i in range(len(best_seq)):
        if best_seq[i]!=p2.op_seq[i]:
            j=best_seq.index(p2.op_seq[i]); new_seq=best_seq.copy()
            new_seq[i],new_seq[j]=new_seq[j],new_seq[i]
            temp=Chromosome(inst,p1.cx,p1.mut_rate)
            temp.op_seq=new_seq.copy(); temp.machine_assign=best_assign.copy(); temp.fitness=None
            fit=temp.decode()
            if fit<best_fit:
                best_fit=fit; best_seq=new_seq.copy(); best_assign=temp.machine_assign.copy()
    return best_seq,best_assign,best_fit

# ---------------------------------------------------------------------------
# GA Driver
# ---------------------------------------------------------------------------
class GeneticAlgorithm:
    def __init__(self,inst, pop=120, gens=150, cx=0.9, mut=0.25, seed=None):
        self.inst, self.pop, self.gens, self.cx, self.mut = inst,pop,gens,cx,mut
        random.seed(seed)
        self.population=[Chromosome(inst,cx,mut) for _ in range(pop)]
        for ind in self.population: ind.decode()
        self.population.sort(key=lambda c:c.fitness)
        self.best=copy.deepcopy(self.population[0])
        self.tabu_queue=[]

    def evolve(self)->Chromosome:
        stag=0
        print(f"· Başladı: POP={self.pop}, GENS={self.gens}, CX={self.cx:.2f}, MUT={self.mut:.2f}")
        for g in range(1,self.gens+1):
            # selection
            sel=[min(random.sample(self.population,2), key=lambda c:c.fitness) for _ in range(self.pop)]
            # offspring
            offs=[]
            for i in range(0,self.pop,2):
                if random.random()<self.cx: c1,c2=ppx_crossover(sel[i],sel[i+1])
                else: c1,c2=copy.deepcopy(sel[i]),copy.deepcopy(sel[i+1])
                offs.extend([c1,c2])
            # mutate & decode
            for ind in offs: mutate(ind,ind.mut_rate); ind.decode()
            # memetic elite
            offs.sort(key=lambda c:c.fitness)
            for ind in offs[:5]: ind.critical_path_search(); ind.decode()
            # elitism replace worst
            offs[-1]=copy.deepcopy(self.best)
            self.population=offs; self.population.sort(key=lambda c:c.fitness)
            # multi-elite path-relink
            elites=self.population[:5]
            for _ in range(3):
                p,q=random.sample(elites,2)
                seq,assign,fit=path_relink(p,q)
                if fit<self.best.fitness:
                    self.best.op_seq=seq; self.best.machine_assign=assign; self.best.fitness=fit
                    print(f"🔄 Multi-Relink geliştirdi: {fit}")
            # update best
            if self.population[0].fitness<self.best.fitness:
                self.best=copy.deepcopy(self.population[0]); self.best.critical_path_search(); stag=0
            else: stag+=1
            # tabu swap on critical path
            if stag>=10:
                for _ in range(5): self.best.critical_path_tabu()
                stag=0
            # random-restart & inner search
            if stag>=50:
                k=int(0.3*self.pop)
                self.population[-k:]=[Chromosome(self.inst,self.cx,self.mut) for _ in range(k)]
                for ind in self.population[:-k]: ind.critical_path_search(); ind.decode()
                print("🔄 Popülasyon kısmi yeniden başlatıldı")
                stag=0
            # param adaptation
            if stag>=5:
                self.mut=min(1.0,self.mut*1.5); print(f"⚙️ Mutasyon artırıldı: {self.mut:.2f}")
            if stag>=8:
                self.cx=max(0.5,self.cx-0.1); print(f"⚙️ Çaprazlama azaltıldı: {self.cx:.2f}")
            # progress
            if g%10==0: print(f"Gen {g:<3d} | Best {self.best.fitness}")
        return self.best

# ---------------------------------------------------------------------------
# Main execution
# ---------------------------------------------------------------------------
if __name__=="__main__":
    ap=argparse.ArgumentParser(description="GA for FJSP")
    ap.add_argument("-f","--file",default="instance.json")
    ap.add_argument("--pop",type=int,default=120)
    ap.add_argument("--gens",type=int,default=150)
    ap.add_argument("--cx",type=float,default=0.9)
    ap.add_argument("--mut",type=float,default=0.25)
    ap.add_argument("--seed",type=int,default=None)
    args=ap.parse_args()

    try: data=json.load(open(args.file))
    except Exception as e: print(f"❌ Girdi okunamadı: {e}"); sys.exit(1)

    print("🚀 GA başladı…")
    inst=FlexibleJobShopInstance(data)
    t0=time.time()
    ga=GeneticAlgorithm(inst,args.pop,args.gens,args.cx,args.mut,args.seed)
    best=ga.evolve()
    elapsed=time.time()-t0

    print(f"🏆 En iyi makespan: {best.fitness}")
    print(f"⏱️  Süre: {elapsed:.2f}s")

    out_dir=pathlib.Path("/home/xibalba/Masaüstü/B-T-RME/pool")
    out_dir.mkdir(parents=True,exist_ok=True)
    ts=time.strftime("%Y-%m-%dT%H-%M-%S")
    fname=f"{pathlib.Path(args.file).stem}_ga_{ts}.json"
    out_path=out_dir/fname
    out_path.write_text(json.dumps(best.export(),indent=2))
    print(f"💾 Çıktı kaydedildi: {out_path}")
