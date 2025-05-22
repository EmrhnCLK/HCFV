import json
from gurobipy import Model, GRB

def solve_fjs(data):
    # Modeli oluştur
    model = Model()

    jobs = data  # Gelen veri (jobs) formatında
    
    # İşler ve işlemler için gerekli sayıları al
    job_count = len(jobs)
    op_count = sum(len(job["operations"]) for job in jobs)
    machines = set(op["machine"] for job in jobs for op in job["operations"])

    # Değişkenleri tanımla
    start_times = {}  # start_times[(job_id, op_index)] = start_time
    end_times = {}  # end_times[(job_id, op_index)] = end_time

    # Her işlem için başlama ve bitiş zamanlarını değişken olarak ekle
    for job_id, job in enumerate(jobs):
        for op_index, op in enumerate(job["operations"]):
            start_times[(job_id, op_index)] = model.addVar(vtype=GRB.CONTINUOUS, name=f"start_{job_id}_{op_index}")
            end_times[(job_id, op_index)] = model.addVar(vtype=GRB.CONTINUOUS, name=f"end_{job_id}_{op_index}")

    # İşlem sürelerini ve makineleri al
    durations = {}
    for job_id, job in enumerate(jobs):
        for op_index, op in enumerate(job["operations"]):
            durations[(job_id, op_index)] = int(op["duration"])

    # Zaman kısıtlarını ekle
    for job_id, job in enumerate(jobs):
        for op_index, op in enumerate(job["operations"]):
            model.addConstr(end_times[(job_id, op_index)] == start_times[(job_id, op_index)] + durations[(job_id, op_index)])

    # İşler arasındaki sıralama kısıtlarını ekle
    for job_id, job in enumerate(jobs):
        for op_index in range(1, len(job["operations"])):
            model.addConstr(start_times[(job_id, op_index)] >= end_times[(job_id, op_index - 1)])

    # Makineler arasında çakışma olmamalı
    for machine in machines:
        for job_id in range(job_count):
            for op_index1 in range(len(jobs[job_id]["operations"])):
                for op_index2 in range(op_index1 + 1, len(jobs[job_id]["operations"])):
                    if jobs[job_id]["operations"][op_index1]["machine"] == machine and jobs[job_id]["operations"][op_index2]["machine"] == machine:
                        model.addConstr(
                            (start_times[(job_id, op_index1)] >= end_times[(job_id, op_index2)]) |
                            (start_times[(job_id, op_index2)] >= end_times[(job_id, op_index1)]))

    # Optimize et
    model.optimize()

    # Çözümü yazdır
    if model.status == GRB.OPTIMAL:
        print("Optimal çözüm bulundu!")
        makespan = 0  # Makespan'ı hesaplamak için bir değişken
        for job_id, job in enumerate(jobs):
            for op_index, op in enumerate(job["operations"]):
                start_time = start_times[(job_id, op_index)].x
                end_time = end_times[(job_id, op_index)].x
                print(f"Job {job['job']}, Operation {op_index}, Start Time: {start_time}, End Time: {end_time}")
                makespan = max(makespan, end_time)  # Bitiş zamanına göre makespan hesapla
        
        # Makespan'ı yazdır
        print(f"Makespan: {makespan}")
    else:
        print("Optimal çözüm bulunamadı!")

# JSON dosyasından veri oku
with open('benchmark.json', 'r') as file:
    data = json.load(file)

# Çözümü başlat
solve_fjs(data)
