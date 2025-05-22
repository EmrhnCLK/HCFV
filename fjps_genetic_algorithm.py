# fjps_ga.py
"""
Flexible Job Shop Scheduling using a Genetic Algorithm with true flexiblity:
- Each operation has multiple machine options with different processing times.
- The GA optimizes both operation sequencing and machine assignment.

Usage:
    1. Prepare `benchmark.json` in working directory with structure:
       [
         {
           "job": "Job A",
           "operations": [
             { "machines": [
                 {"machine": "M1", "duration": 5},
                 {"machine": "M2", "duration": 7}
               ]
             },
             ...
           ]
         },
         ...
       ]
    2. Run: `python fjps_ga.py`
"""
import sys
import json
import random
import copy
import time
class FlexibleJobShopInstance:
    """
    Represents a FJSP instance where each operation may run on multiple machines.
    operations: list of tuples (job_id, op_index, {machine_id: duration, ...})
    """
    def __init__(self, data):
        self.jobs = []
        self.machine_ids = []
        self.operations = []
        # collect unique machines
        machine_set = set()
        for job in data:
            for op in job['operations']:
                opts = op.get('machines', [])
                # single-option fallback
                if not opts and 'machine' in op and 'duration' in op:
                    opts = [ { 'machine': op['machine'], 'duration': op['duration'] } ]
                for m in opts:
                    machine_set.add(m['machine'])
        self.machine_ids = sorted(machine_set)
        # map machine to index
        self.machine_index = {m: i for i, m in enumerate(self.machine_ids)}
        # parse jobs & operations
        for j_idx, job in enumerate(data):
            self.jobs.append(job['job'])
            for op_idx, op in enumerate(job['operations']):
                opts = op.get('machines', [])
                if not opts and 'machine' in op:
                    opts = [{'machine': op['machine'], 'duration': op['duration']}]
                mopts = {}
                for option in opts:
                    m_name = option['machine']
                    try:
                        dur = int(option['duration'])
                    except:
                        raise ValueError(f"Invalid duration {option['duration']} in job {job['job']}")
                    m_id = self.machine_index[m_name]
                    mopts[m_id] = dur
                self.operations.append((j_idx, op_idx, mopts))
        self.num_ops = len(self.operations)

class Chromosome:
    """
    Encodes a solution:
      - op_seq: permutation of operations
      - machine_assign: dict op_id -> chosen machine_id
    """
    def __init__(self, instance):
        self.instance = instance
        # sequence of operations
        self.op_seq = list(range(instance.num_ops))
        random.shuffle(self.op_seq)
        # initial machine assignment: random choice among options
        self.machine_assign = {}
        for op_id, (_,_,mopts) in enumerate(instance.operations):
            self.machine_assign[op_id] = random.choice(list(mopts.keys()))
        self.fitness = None

    def decode(self):
        """
        List-scheduling decode: schedule operations by op_seq and assigned machines.
        Returns makespan (max completion time).
        """
        job_end = {j:0 for j in range(len(self.instance.jobs))}
        mach_end = {m:0 for m in range(len(self.instance.machine_ids))}
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

class GeneticAlgorithm:
    """
    GA with:
      - Tournament selection
      - Single-point crossover on op_seq
      - Uniform crossover on machine_assign
      - Swap mutation on op_seq and random assignment mutation
    """
    def __init__(self, instance, pop_size=50, gens=100, cx_rate=0.8, mut_rate=0.2):
        self.instance = instance
        self.pop_size = pop_size
        self.gens = gens
        self.cx_rate = cx_rate
        self.mut_rate = mut_rate
        self.population = []

    def init_pop(self):
        self.population = [Chromosome(self.instance) for _ in range(self.pop_size)]

    def evaluate(self):
        for ind in self.population:
            ind.decode()
        self.population.sort(key=lambda c: c.fitness)

    def select(self):
        selected = []
        for _ in range(self.pop_size):
            a, b = random.sample(self.population, 2)
            winner = a if a.fitness < b.fitness else b
            selected.append(copy.deepcopy(winner))
        return selected

    def crossover(self, p1, p2):
        c1, c2 = copy.deepcopy(p1), copy.deepcopy(p2)
        if random.random() < self.cx_rate:
            # crossover op_seq
            pt = random.randint(1, self.instance.num_ops-1)
            c1.op_seq = p1.op_seq[:pt] + [o for o in p2.op_seq if o not in p1.op_seq[:pt]]
            c2.op_seq = p2.op_seq[:pt] + [o for o in p1.op_seq if o not in p2.op_seq[:pt]]
            # uniform crossover machine assignments
            for op_id in c1.machine_assign:
                if random.random() < 0.5:
                    c1.machine_assign[op_id] = p2.machine_assign[op_id]
                    c2.machine_assign[op_id] = p1.machine_assign[op_id]
        return c1, c2

    def mutate(self, ind):
        # swap mutation for op_seq
        for i in range(self.instance.num_ops):
            if random.random() < self.mut_rate:
                j = random.randrange(self.instance.num_ops)
                ind.op_seq[i], ind.op_seq[j] = ind.op_seq[j], ind.op_seq[i]
        # random re-assignment for machine options
        for op_id, (_,_,mopts) in enumerate(self.instance.operations):
            if random.random() < self.mut_rate:
                ind.machine_assign[op_id] = random.choice(list(mopts.keys()))

    def evolve(self):
        self.init_pop()
        self.evaluate()
        best = copy.deepcopy(self.population[0])
        print(f"Gen 0 best makespan: {best.fitness}")
        for g in range(1, self.gens+1):
            selected = self.select()
            offspring = []
            for i in range(0, self.pop_size, 2):
                c1, c2 = self.crossover(selected[i], selected[i+1])
                offspring.extend([c1, c2])
            for ind in offspring:
                self.mutate(ind)
            self.population = offspring
            self.evaluate()
            if self.population[0].fitness < best.fitness:
                best = copy.deepcopy(self.population[0])
            if g % 10 == 0:
                print(f"Gen {g} best makespan: {best.fitness}")
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
    ga = GeneticAlgorithm(inst, pop_size=50, gens=100)
    best = ga.evolve()
    print("\nBest makespan found:", best.fitness)
    # print detailed schedule
    print(best.decode())
    end_time = time.time()
    print(f"\n⏱️ Çözüm süresi: {end_time - start_time:.4f} saniye")
