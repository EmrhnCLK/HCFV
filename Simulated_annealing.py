# fjps_sa.py
"""
Flexible Job Shop Scheduling using Simulated Annealing (SA)

Usage:
    1. Prepare `benchmark.json` in working directory
    2. Run: `python fjps_sa.py`
"""
import sys
import json
import random
import math
import time
import copy

class FlexibleJobShopInstance:
    def __init__(self, data):
        self.jobs = []
        self.machine_ids = []
        self.operations = []
        machine_set = set()
        for job in data:
            for op in job['operations']:
                opts = op.get('machines', [])
                for m in opts:
                    machine_set.add(m['machine'])
        self.machine_ids = sorted(machine_set)
        self.machine_index = {m: i for i, m in enumerate(self.machine_ids)}
        for j_idx, job in enumerate(data):
            self.jobs.append(job['job'])
            for op_idx, op in enumerate(job['operations']):
                mopts = {}
                for option in op['machines']:
                    m_name = option['machine']
                    dur = int(option['duration'])
                    m_id = self.machine_index[m_name]
                    mopts[m_id] = dur
                self.operations.append((j_idx, op_idx, mopts))
        self.num_ops = len(self.operations)

class Solution:
    def __init__(self, instance):
        self.instance = instance
        self.op_seq = list(range(instance.num_ops))
        random.shuffle(self.op_seq)
        self.machine_assign = {i: min(op[2], key=op[2].get) for i, op in enumerate(instance.operations)}
        self.fitness = None

    def decode(self):
        job_end = {j: 0 for j in range(len(self.instance.jobs))}
        mach_end = {m: 0 for m in range(len(self.instance.machine_ids))}
        schedule = []
        for op_id in self.op_seq:
            job_id, _, mopts = self.instance.operations[op_id]
            m_id = self.machine_assign[op_id]
            dur = mopts[m_id]
            start = max(job_end[job_id], mach_end[m_id])
            end = start + dur
            job_end[job_id] = end
            mach_end[m_id] = end
            schedule.append((op_id, job_id, m_id, start, end))
        self.fitness = max(end for *_, end in schedule)
        return self.fitness

    def neighbor(self):
        neighbor = copy.deepcopy(self)
        if random.random() < 0.5:
            i, j = random.sample(range(self.instance.num_ops), 2)
            neighbor.op_seq[i], neighbor.op_seq[j] = neighbor.op_seq[j], neighbor.op_seq[i]
        else:
            op_id = random.randint(0, self.instance.num_ops - 1)
            mopts = self.instance.operations[op_id][2]
            neighbor.machine_assign[op_id] = random.choice(list(mopts.keys()))
        return neighbor

def auto_calibrate_T_init(instance, sample_size=100, p_target=0.8):
    base = Solution(instance)
    base.decode()
    deltas = []
    for _ in range(sample_size):
        neighbor = base.neighbor()
        neighbor.decode()
        delta = neighbor.fitness - base.fitness
        if delta > 0:
            deltas.append(delta)
    if not deltas:
        return 100.0
    avg_delta = sum(deltas) / len(deltas)
    T_init = avg_delta / math.log(1 / p_target)
    print(f"🔧 Otomatik T_init ayarlandı: {T_init:.2f} (p_target={p_target})")
    return T_init

def simulated_annealing(instance, T_init=None, T_min=1e-6, alpha=0.98, max_iter=5000):
    if T_init is None:
        T_init = auto_calibrate_T_init(instance)

    current = Solution(instance)
    best = copy.deepcopy(current)
    current.decode()
    best.decode()
    T = T_init

    for iteration in range(max_iter):
        neighbor = current.neighbor()
        neighbor.decode()
        delta = neighbor.fitness - current.fitness
        if delta < 0 or random.random() < math.exp(-delta / T):
            current = neighbor
            if current.fitness < best.fitness:
                best = copy.deepcopy(current)
        T *= alpha
        if T < T_min:
            break
        if iteration % 100 == 0:
            print(f"Iter {iteration}, Temp {T:.6f}, Best makespan: {best.fitness}")
    return best

if __name__ == '__main__':
    start_time = time.time()
    filename = 'benchmark.json'
    try:
        with open(filename) as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: '{filename}' not found.")
        sys.exit(1)
    inst = FlexibleJobShopInstance(data)
    best = simulated_annealing(inst, T_init=None, T_min=1e-6, alpha=0.98, max_iter=5000)
    print("\nBest makespan found:", best.fitness)
    print(best.decode())
    end_time = time.time()
    print(f"\n⏱️ Çözüm süresi: {end_time - start_time:.4f} saniye")