import json
from gurobipy import Model, GRB, quicksum
import time

def solve_fjsp(data):
    model = Model("FlexibleJobShop")
    model.setParam('OutputFlag', 1)  # Konsolu temiz tutmak için

    BIG_M = 1000
    jobs = data
    operations = []  # (job_id, op_index, [(machine_id, duration)])
    machine_set = set()

    for job_id, job in enumerate(jobs):
        for op_index, op in enumerate(job["operations"]):
            op_machines = [(m["machine"], int(m["duration"])) for m in op["machines"]]
            machine_set.update([m for m, _ in op_machines])
            operations.append((job_id, op_index, op_machines))

    machine_list = list(machine_set)

    # Değişkenler
    x = {}  # Atama değişkeni: (job, op, machine) -> binary
    s = {}  # Başlama zamanı: (job, op) -> continuous
    c = {}  # Bitiş zamanı: (job, op) -> continuous

    for job_id, op_index, mlist in operations:
        s[(job_id, op_index)] = model.addVar(vtype=GRB.CONTINUOUS, name=f"s_{job_id}_{op_index}")
        c[(job_id, op_index)] = model.addVar(vtype=GRB.CONTINUOUS, name=f"c_{job_id}_{op_index}")
        for m, dur in mlist:
            x[(job_id, op_index, m)] = model.addVar(vtype=GRB.BINARY, name=f"x_{job_id}_{op_index}_{m}")

    # Makespan değişkeni
    makespan = model.addVar(vtype=GRB.CONTINUOUS, name="makespan")

    # Her işlem bir makinaya atanmalı
    for job_id, op_index, mlist in operations:
        model.addConstr(quicksum(x[(job_id, op_index, m)] for m, _ in mlist) == 1)

    # Bitiş zamanı = başlama + seçilen süresi
    for job_id, op_index, mlist in operations:
        model.addConstr(
            c[(job_id, op_index)] == s[(job_id, op_index)] + quicksum(
                dur * x[(job_id, op_index, m)] for m, dur in mlist
            )
        )

    # İş sırası korunmalı
    for job_id, job in enumerate(jobs):
        for i in range(1, len(job["operations"])):
            model.addConstr(s[(job_id, i)] >= c[(job_id, i - 1)])

    # Aynı makinada çakışma olmamalı (disjunctive constraints)
    for m in machine_list:
        for (j1, o1, mlist1) in operations:
            if m not in [mach for mach, _ in mlist1]:
                continue
            for (j2, o2, mlist2) in operations:
                if (j1, o1) >= (j2, o2):
                    continue
                if m not in [mach for mach, _ in mlist2]:
                    continue
                y = model.addVar(vtype=GRB.BINARY, name=f"y_{j1}_{o1}_{j2}_{o2}_{m}")
                model.addConstr(s[(j1, o1)] >= c[(j2, o2)] - BIG_M * (1 - y) - BIG_M * (1 - x[(j1, o1, m)]) - BIG_M * (1 - x[(j2, o2, m)]))
                model.addConstr(s[(j2, o2)] >= c[(j1, o1)] - BIG_M * y - BIG_M * (1 - x[(j1, o1, m)]) - BIG_M * (1 - x[(j2, o2, m)]))

    # Makespan >= her işlemin bitiş zamanı
    for key in c:
        model.addConstr(makespan >= c[key])

    model.setObjective(makespan, GRB.MINIMIZE)
    model.optimize()

    if model.status == GRB.OPTIMAL:
        print("\n✅ Optimal çözüm bulundu!")
        print("Makespan:", makespan.X)
        for job_id, op_index, mlist in operations:
            m_selected = [m for m, _ in mlist if x[(job_id, op_index, m)].X > 0.5][0]
            print(f"Job {job_id}, Op {op_index} -> Machine {m_selected}, Start: {s[(job_id, op_index)].X:.1f}, End: {c[(job_id, op_index)].X:.1f}")
    else:
        print("\n❌ Optimal çözüm bulunamadı.")

if __name__ == '__main__':
    start_time = time.time()
    with open("benchmark.json") as f:
        data = json.load(f)
    solve_fjsp(data)
    end_time = time.time()
    print(f"\n⏱️ Çözüm süresi: {end_time - start_time:.4f} saniye")
