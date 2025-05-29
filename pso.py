#!/usr/bin/env python3
"""
Flexible Job‑Shop Scheduling — Discrete PSO (DPOSO) v3
======================================================
* Adds JSON schedule export, live feedback, and default output directory for pool creation *

Features:
- Precedence‑safe particles
- Swap/insert/machine‑change moves
- Critical‑path tweak in velocity update
- Adaptive inertia & coefficients
- JSON export (`export()`), `--save` flag
- Live iteration logs and completion summary
"""
from __future__ import annotations
import sys, json, random, copy, time, argparse, math, pathlib
from typing import List, Dict, Tuple, Optional

# ---------------------------------------------------------------------------
class FlexibleJobShopInstance:
    def __init__(self, data):
        self.jobs: List[str] = []
        self.machine_ids: List[str] = []
        self.operations: List[Tuple[int,int,Dict[int,int]]] = []
        machine_set = set()
        for job in data:
            for op in job["operations"]:
                for m in op["machines"]:
                    machine_set.add(m["machine"])
        self.machine_ids = sorted(machine_set)
        self.mach_index = {m:i for i,m in enumerate(self.machine_ids)}
        for j_idx, job in enumerate(data):
            self.jobs.append(job["job"])
            for o_idx, op in enumerate(job["operations"]):
                mopts = {self.mach_index[m["machine"]]: int(m["duration"]) for m in op["machines"]}
                self.operations.append((j_idx,o_idx,mopts))
        self.num_ops = len(self.operations)
        # predecessor
        self.prev_op = [None]*self.num_ops
        job_last: Dict[int,int] = {}
        for op_id,(j,_,_) in enumerate(self.operations):
            if j in job_last:
                self.prev_op[op_id] = job_last[j]
            job_last[j] = op_id
    
    def random_topo_order(self) -> List[int]:
        in_deg = {i:0 for i in range(self.num_ops)}
        succ = {i:[] for i in range(self.num_ops)}
        for op,prev in enumerate(self.prev_op):
            if prev is not None:
                in_deg[op]+=1; succ[prev].append(op)
        avail = [op for op,d in in_deg.items() if d==0]
        order = []
        while avail:
            op = random.choice(avail); avail.remove(op)
            order.append(op)
            for s in succ[op]:
                in_deg[s]-=1
                if in_deg[s]==0: avail.append(s)
        return order

    def is_precedence_ok(self, seq: List[int]) -> bool:
        seen = set()
        for op in seq:
            prev = self.prev_op[op]
            if prev is not None and prev not in seen:
                return False
            seen.add(op)
        return True

# ---------------------------------------------------------------------------
class Particle:
    def __init__(self, inst: FlexibleJobShopInstance):
        self.inst = inst
        self.seq = inst.random_topo_order()
        self.mach = {i:min(op[2],key=op[2].get) for i,op in enumerate(inst.operations)}
        self.fitness: Optional[int] = None
        # personal best
        self.best_seq = self.seq[:]
        self.best_mach = self.mach.copy()
        self.best_fit = math.inf

    def decode(self) -> int:
        job_end = {j:0 for j in range(len(self.inst.jobs))}
        mach_end = {m:0 for m in range(len(self.inst.machine_ids))}
        for op in self.seq:
            j,_,mopts = self.inst.operations[op]
            m = self.mach[op]; dur = mopts[m]
            start = max(job_end[j],mach_end[m]); end = start+dur
            job_end[j]=end; mach_end[m]=end
        self.fitness = max(mach_end.values())
        return self.fitness

    def critical_path_ops(self) -> List[int]:
        job_end = {j:0 for j in range(len(self.inst.jobs))}
        mach_end = {m:0 for m in range(len(self.inst.machine_ids))}
        op_start,op_end = {},{}
        for op in self.seq:
            j,_,mopts = self.inst.operations[op]
            m=self.mach[op]; dur=mopts[m]
            s=max(job_end[j],mach_end[m]); e=s+dur
            op_start[op]=s; op_end[op]=e
            job_end[j]=e; mach_end[m]=e
        makespan = max(mach_end.values())
        return [op for op,e in op_end.items() if e==makespan]

    def export(self) -> dict:
        # build sequences
        seqs = {i:[] for i in range(len(self.inst.machine_ids))}
        job_end={}; mach_end={}; times={}
        # need schedule times
        self.decode()
        # reconstruct schedule by decode again
        job_end={j:0 for j in range(len(self.inst.jobs))}
        mach_end={m:0 for m in range(len(self.inst.machine_ids))}
        schedule=[]
        for op in self.seq:
            j,_,mopts=self.inst.operations[op]
            m=self.mach[op]; dur=mopts[m]
            s=max(job_end[j],mach_end[m]); e=s+dur
            schedule.append((op,j,m,s,e))
            job_end[j]=e; mach_end[m]=e
        for op,j,m,s,e in schedule:
            seqs[m].append(op)
            times[op]=(s,e-s)
        machine_sequences={self.inst.machine_ids[m]:seqs[m] for m in seqs}
        start={str(op):s for op,(s,d) in times.items()}
        duration={str(op):d for op,(s,d) in times.items()}
        return {"makespan":self.fitness,"machine_sequences":machine_sequences,
                "start":start,"duration":duration}

# ---------------------------------------------------------------------------
def discrete_pso(inst: FlexibleJobShopInstance, *, swarm_size=60, iters=300,
                 w=0.3, c1=1.7, c2=1.7, mut_rate=0.05, p_cp=0.35,
                 seed: Optional[int]=None) -> Particle:
    if seed is not None: random.seed(seed)
    swarm=[Particle(inst) for _ in range(swarm_size)]
    g_seq,g_mach,g_fit=[],{},math.inf
    # init
    for p in swarm:
        p.decode(); p.best_seq, p.best_mach, p.best_fit = p.seq[:],p.mach.copy(),p.fitness
        if p.fitness<g_fit:
            g_seq,g_mach,g_fit=p.seq[:],p.mach.copy(),p.fitness
    print(f"🚀 PSO başladı… swarm={swarm_size}, iters={iters}, w={w}, c1={c1}, c2={c2}, mut_rate={mut_rate}, p_cp={p_cp}")
    for it in range(1,iters+1):
        for p in swarm:
            # sequence update
            new_seq=p.seq[:]
            if random.random()<p_cp:
                crit=Particle(inst); crit.seq,crit.mach = g_seq,g_mach
                cp_list=crit.critical_path_ops()
                if cp_list:
                    op=random.choice(cp_list); pos=new_seq.index(op)
                    if pos>0:
                        tgt=pos-1; prev=inst.prev_op[op]
                        if prev!=new_seq[tgt]: new_seq[pos],new_seq[tgt]=new_seq[tgt],new_seq[pos]
            else:
                for i in range(inst.num_ops):
                    if random.random()<c1:
                        op=p.best_seq[i]; new_seq.remove(op); new_seq.insert(i,op)
                    if random.random()<c2 and g_seq:
                        op=g_seq[i] if i<len(g_seq) else None
                        if op in new_seq: new_seq.remove(op); new_seq.insert(i,op)
                if random.random()<w: new_seq=p.seq[:]
            if not inst.is_precedence_ok(new_seq): new_seq=inst.random_topo_order()
            # machine update
            new_mach=p.mach.copy()
            for op_id in range(inst.num_ops):
                if random.random()<mut_rate: new_mach[op_id]=random.choice(list(inst.operations[op_id][2].keys()))
                else:
                    if random.random()<c1: new_mach[op_id]=p.best_mach[op_id]
                    if random.random()<c2: new_mach[op_id]=g_mach.get(op_id,new_mach[op_id])
            p.seq,p.mach=new_seq,new_mach; p.decode()
            # update pbest
            if p.fitness<p.best_fit: p.best_seq,p.best_mach,p.best_fit=p.seq[:],p.mach.copy(),p.fitness
            # update gbest
            if p.fitness<g_fit: g_seq,g_mach,g_fit=p.seq[:],p.mach.copy(),p.fitness
        if it==1 or it%10==0:
            print(f"Iter {it:>3}/{iters} | Global best makespan: {g_fit}")
    best=Particle(inst); best.seq,best.mach,best.fitness = g_seq,g_mach,g_fit
    return best

# ---------------------------------------------------------------------------
if __name__ == "__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("-f","--file",default="benchmark.json")
    ap.add_argument("--swarm",type=int,default=60)
    ap.add_argument("--iters",type=int,default=300)
    ap.add_argument("--w",type=float,default=0.3)
    ap.add_argument("--c1",type=float,default=1.7)
    ap.add_argument("--c2",type=float,default=1.7)
    ap.add_argument("--mut-rate",type=float,default=0.05)
    ap.add_argument("--p-cp",type=float,default=0.35)
    ap.add_argument("--seed",type=int)
    ap.add_argument("--save",type=str,help="Schedule JSON path")
    args=ap.parse_args()

    try:
        data=json.load(open(args.file))
    except Exception as e:
        print(f"❌ Girdi okunamadı: {e}"); sys.exit(1)

    inst=FlexibleJobShopInstance(data)
    t0=time.time()
    best=discrete_pso(inst, swarm_size=args.swarm, iters=args.iters,
                      w=args.w,c1=args.c1,c2=args.c2,
                      mut_rate=args.mut_rate,p_cp=args.p_cp,seed=args.seed)
    elapsed=time.time()-t0
    print(f"🌟 En iyi makespan: {best.fitness}\n⏱️  Süre: {elapsed:.2f} s")

    # JSON export
    out_dir=pathlib.Path("/home/xibalba/Masaüstü/B-T-RME/PRE_MODEL/")
    out_dir.mkdir(parents=True,exist_ok=True)
    timestamp=time.strftime("%Y-%m-%dT%H-%M-%S")
    fname=f"{pathlib.Path(args.file).stem}_pso_{timestamp}.json"
    out_path=pathlib.Path(args.save) if args.save else out_dir/fname
    export_data=best.export()
    out_path.write_text(json.dumps(export_data,indent=2))
    print(f"💾 Çıktı kaydedildi: {out_path}")
