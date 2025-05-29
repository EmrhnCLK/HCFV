#!/usr/bin/env python3
"""
Flexible Job‑Shop Scheduling (FJSP) — Simulated Annealing (SA) v5
================================================================
* Memetik Intensification + Random-Restart + Self-Adaptive p_cp + JSON Export *

Usage:
------
```bash
python Simulated_annealing.py [-f instance.json] [--seed SEED] \
    [--alpha ALPHA] [--inner-loop N] [--max-iter M] [--p-cp P] \
    [--reheat-every R] [--reheat-mult RM] [--save PATH]
```

Özellikler:
- **Critical‑Path Guided Neighbourhood (CP-move)**
- **Self-Adaptive** p_cp: stagnasyona göre p_cp artar
- **Critical-Path Local Search:** her yeni en iyi çözüme uygulanır
- **Random-Restart:** stagnasyon uzun sürerse yeni rastgele çözüm
- **JSON Export:** `--save` ile dosya kaydı, varsayılan PRE_MODEL dizini
- **Live Feedback:** iterasyon, reheat, restart logları
"""
from __future__ import annotations
import sys, json, random, math, time, copy, argparse, pathlib
from collections import defaultdict

# ----------------------------------------------------------------------------
# Instance & solution
# ----------------------------------------------------------------------------
class FlexibleJobShopInstance:
    def __init__(self, data):
        self.jobs, self.operations = [], []
        machine_set=set()
        for job in data:
            for op in job['operations']:
                for m in op['machines']:
                    machine_set.add(m['machine'])
        self.machine_ids=sorted(machine_set)
        self.m_index={m:i for i,m in enumerate(self.machine_ids)}
        for j_idx,job in enumerate(data):
            self.jobs.append(job['job'])
            for o_idx,op in enumerate(job['operations']):
                mopts={self.m_index[m['machine']]:int(m['duration']) for m in op['machines']}
                self.operations.append((j_idx,o_idx,mopts))
        self.num_jobs=len(self.jobs)
        self.num_ops=len(self.operations)
        self.pred=[[] for _ in range(self.num_ops)]
        for op_id,(j,o,_) in enumerate(self.operations):
            if o>0:
                prev=(j,o-1)
                pid=[i for i,(jj,oo,_) in enumerate(self.operations) if (jj,oo)==prev][0]
                self.pred[op_id].append(pid)

class Solution:
    def __init__(self, inst:FlexibleJobShopInstance):
        self.inst=inst
        self.op_seq=self._random_topo_order()
        self.machine_assign={i:min(op[2],key=op[2].get) for i,op in enumerate(inst.operations)}
        self.fitness=None; self.schedule=None
        # self-adaptive
        self.p_cp=0.35
    def _random_topo_order(self):
        indeg=[len(self.inst.pred[i]) for i in range(self.inst.num_ops)]
        ready=[i for i,d in enumerate(indeg) if d==0]; seq=[]
        while ready:
            v=random.choice(ready); ready.remove(v)
            seq.append(v)
            for w in range(self.inst.num_ops):
                if v in self.inst.pred[w]:
                    indeg[w]-=1
                    if indeg[w]==0: ready.append(w)
        return seq
    def decode(self):
        if self.fitness is not None: return self.fitness
        job_end=[0]*self.inst.num_jobs; mach_end=[0]*len(self.inst.machine_ids)
        sched=[]
        for op in self.op_seq:
            j,_,mopts=self.inst.operations[op]
            m=self.machine_assign[op]; d=mopts[m]
            s=max(job_end[j],mach_end[m]); e=s+d
            job_end[j]=e; mach_end[m]=e
            sched.append((op,j,m,s,e))
        self.schedule=sched; self.fitness=max(e for *_,e in sched)
        return self.fitness
    def export(self):
        seqs={i:[] for i in range(len(self.inst.machine_ids))}
        for op in self.op_seq: seqs[self.machine_assign[op]].append(op)
        machine_sequences={self.inst.machine_ids[m]:seqs[m] for m in seqs}
        start={str(op):s for op,_,_,s,_ in self.schedule}
        duration={str(op):e-s for op,_,_,s,e in self.schedule}
        return {"makespan":self.fitness, "machine_sequences":machine_sequences, "start":start, "duration":duration}
    def _critical_ops(self):
        if self.schedule is None: self.decode()
        return [op for op,_,_,_,e in self.schedule if e==max(e for *_,e in self.schedule)]
    
    def critical_path_search(self) -> bool:
        """Local search on critical path operations"""
        if self.fitness is None:
            self.decode()
        job_end = defaultdict(int)
        mach_end = defaultdict(int)
        times = {}
        for op_id in self.op_seq:
            j, _, mopts = self.inst.operations[op_id]
            m = self.machine_assign[op_id]
            dur = mopts[m]
            s = max(job_end[j], mach_end[m])
            e = s + dur
            job_end[j] = e
            mach_end[m] = e
            times[op_id] = (s, e)
        makespan = max(e for _, e in times.values())
        improved = False
        # try reassignment for each critical op
        for op in [op for op, (_, e) in times.items() if e == makespan]:
            best_m = self.machine_assign[op]
            best_fit = makespan
            for alt_m in self.inst.operations[op][2].keys():
                if alt_m == best_m:
                    continue
                old = self.machine_assign[op]
                self.machine_assign[op] = alt_m
                self.fitness = None
                new_fit = self.decode()
                if new_fit < best_fit:
                    best_m, best_fit = alt_m, new_fit
                    improved = True
                else:
                    self.machine_assign[op] = old
                    self.fitness = makespan
            self.machine_assign[op] = best_m
            self.fitness = best_fit
        return improved

    def neighbor(self):
        nb=copy.deepcopy(self)
        if random.random()<self.p_cp:
            # CP move
            ops=self._critical_ops(); op=random.choice(ops)
            mopts=self.inst.operations[op][2]; cur=self.machine_assign[op]
            better=[m for m,d in mopts.items() if d<mopts[cur]]
            nb.machine_assign[op]=random.choice(better) if better else random.choice(list(mopts.keys()))
        else:
            # random swap or insert or machine change
            if random.random()<0.33: nb._insert_move()
            elif random.random()<0.5: nb._swap_move()
            else: nb._machine_change()
        nb.fitness=None; nb.schedule=None
        return nb
    def _insert_move(self):
        a,b=random.sample(range(len(self.op_seq)),2); a,b=min(a,b),max(a,b)
        op=self.op_seq.pop(b)
        if set(self.inst.pred[op])&set(self.op_seq[a:b]): self.op_seq.insert(b,op)
        else: self.op_seq.insert(a,op)
    def _swap_move(self):
        for _ in range(5):
            i,j=random.sample(range(len(self.op_seq)),2)
            oi,oj=self.op_seq[i],self.op_seq[j]
            ji,idxi,_=self.inst.operations[oi]; jj,idxj,_=self.inst.operations[oj]
            if ji==jj and ((idxi<idxj and i>j) or (idxj<idxi and j>i)): continue
            self.op_seq[i],self.op_seq[j]=oj,oi; break
    def _machine_change(self):
        op=random.randrange(self.inst.num_ops); mopts=self.inst.operations[op][2]
        choices=[m for m in mopts if m!=self.machine_assign[op]]
        if choices: self.machine_assign[op]=random.choice(choices)

# ----------------------------------------------------------------------------
# Simulated Annealing with Intensification
# ----------------------------------------------------------------------------
def parse_args():
    ap=argparse.ArgumentParser()
    ap.add_argument("-f","--file",default="instance.json")
    ap.add_argument("--seed",type=int)
    ap.add_argument("--alpha",type=float,default=0.9985)
    ap.add_argument("--inner-loop",type=int,default=60)
    ap.add_argument("--max-iter",type=int,default=120000)
    ap.add_argument("--p-cp",type=float,default=0.35)
    ap.add_argument("--reheat-every",type=int,default=15000)
    ap.add_argument("--reheat-mult",type=float,default=1.25)
    ap.add_argument("--save",type=str,help="Output JSON path")
    return ap.parse_args()

def simulated_annealing(inst,args):
    random.seed(args.seed)
    cur=Solution(inst); cur.decode(); cur.p_cp=args.p_cp
    best=copy.copy(cur); stag=0
    T0=cur.fitness/10; T=T0
    print("🚀 SA başladı…")
    start=time.time()
    for step in range(1,args.max_iter+1):
        nb=cur.neighbor(); nb.decode()
        d=nb.fitness-cur.fitness
        if d<0 or random.random()<math.exp(-d/T): cur=nb
        if cur.fitness<best.fitness:
            best=copy.copy(cur); best.critical_path_search()
            stag=0; print(f"🔍 CP-intensified Best {best.fitness} at iter {step}")
        else: stag+=1
        if step%args.inner_loop==0: T*=args.alpha
        if args.reheat_every and step%args.reheat_every==0:
            T*=args.reheat_mult; print(f"♨️ Reheat → T={T:.2f}")
        # self-adapt p_cp
        if stag and stag%5000==0:
            cur.p_cp=min(1.0,cur.p_cp*1.1)
            print(f"⚙️ p_cp adapt {cur.p_cp:.2f} at iter {step}")
        # random-restart
        if stag and stag%20000==0:
            cur=Solution(inst); cur.decode(); cur.p_cp=args.p_cp
            print(f"🔄 Restart at iter {step}"); stag=0
    elapsed=time.time()-start
    print(f"🏆 Best makespan: {best.fitness}")
    print(f"⏱️  Süre: {elapsed:.2f}s")
    # export JSON
    out_dir=pathlib.Path("/home/xibalba/Masaüstü/B-T-RME/pool")
    out_dir.mkdir(parents=True,exist_ok=True)
    ts=time.strftime("%Y-%m-%dT%H-%M-%S")
    fname=f"{pathlib.Path(args.file).stem}_sa_{ts}.json"
    out_path=pathlib.Path(args.save) if args.save else out_dir/fname
    data=best.export()
    out_path.write_text(json.dumps(data,indent=2))
    print(f"💾 Çıktı kaydedildi: {out_path}")

if __name__=="__main__":
    args=parse_args()
    data=json.load(open(args.file))
    inst=FlexibleJobShopInstance(data)
    simulated_annealing(inst,args)
