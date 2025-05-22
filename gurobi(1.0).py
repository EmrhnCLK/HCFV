import gurobipy as grb
import json

# JSON dosyasından veri yüklemek için fonksiyon
def load_data_from_json(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

def solve_fjs(data):
    # Veri yapısını kontrol et
    print("Yüklenen veri:", data)
    
    # Model oluşturuluyor
    model = grb.Model("fjs_scheduling")

    # Değişkenler: Start ve End times için
    start_times = {}
    end_times = {}
    
    # Zaman dilimleri: her işlem için bir zaman dilimi olacak
    for job in data:  # 'data' şu an bir liste
        job_id = job["job"]  # "job" ID'sini metin (string) olarak kullanıyoruz
        for op_index, operation in enumerate(job["operations"]):
            machine_id = int(operation["machine"])  # "machine" ID'si integer'a dönüştürülmeli
            duration = int(operation["duration"])  # "duration" integer'a dönüştürülmeli
            
            # Start ve End time değişkenleri
            start_times[(job_id, op_index)] = model.addVar(vtype=grb.GRB.CONTINUOUS, name=f"start_{job_id}_{op_index}")
            end_times[(job_id, op_index)] = model.addVar(vtype=grb.GRB.CONTINUOUS, name=f"end_{job_id}_{op_index}")

            # Bitirme zamanı, başlama zamanına işlem süresi eklenerek hesaplanır
            model.addConstr(end_times[(job_id, op_index)] == start_times[(job_id, op_index)] + duration, 
                             name=f"end_time_{job_id}_{op_index}")

    # Makine kullanılabilirlik kısıtlaması: Aynı anda bir makine birden fazla işlem yapamaz
    for job in data:
        job_id = job["job"]  # "job" ID'sini metin (string) olarak kullanıyoruz
        for op_index, operation in enumerate(job["operations"]):
            machine_id = int(operation["machine"])  # "machine" ID'si integer'a dönüştürülmeli
            for other_job in data:
                other_job_id = other_job["job"]  # Diğer işlerin id'leri metin olarak
                for other_op_index, other_operation in enumerate(other_job["operations"]):
                    if job_id != other_job_id:  # Aynı işteki işlemlerle çakışmamalı
                        other_machine_id = int(other_operation["machine"])  # "machine" ID'si integer'a dönüştürülmeli
                        if machine_id == other_machine_id:  # Aynı makineyi kullanıyorsa
                            # Bu kısıtlama şu şekilde düzeltildi:
                            # Bir işlem, diğerinin bitişinden önce başlayamaz
                            model.addConstr(
                                start_times[(job_id, op_index)] >= end_times[(other_job_id, other_op_index)],
                                name=f"machine_conflict_{job_id}_{op_index}_vs_{other_job_id}_{other_op_index}_1"
                            )
                            model.addConstr(
                                start_times[(other_job_id, other_op_index)] >= end_times[(job_id, op_index)],
                                name=f"machine_conflict_{job_id}_{op_index}_vs_{other_job_id}_{other_op_index}_2"
                            )
    
    # İşlerin sıralanması: Bir işlemin bitişi, sonraki işlemin başlangıcından önce olmamalı
    for job in data:
        job_id = job["job"]  # "job" ID'sini metin (string) olarak kullanıyoruz
        for op_index in range(1, len(job["operations"])):
            model.addConstr(
                start_times[(job_id, op_index)] >= end_times[(job_id, op_index - 1)],
                name=f"job_order_{job_id}_{op_index}"
            )
    
    # Başlangıç zamanları pozitif olmalı (başlangıç zamanı < 0 olamaz)
    for job in data:
        job_id = job["job"]  # "job" ID'sini metin (string) olarak kullanıyoruz
        model.addConstr(start_times[(job_id, 0)] >= 0, name=f"start_time_non_negative_{job_id}")

    # Modeli optimize et
    model.optimize()

    if model.status == grb.GRB.Status.OPTIMAL:
        print("Optimal çözüm bulundu!")
        # Çözümleri yazdır
        for job in data:
            job_id = job["job"]  # "job" ID'sini metin (string) olarak kullanıyoruz
            for op_index, operation in enumerate(job["operations"]):
                print(f"Job {job_id}, Operation {op_index}, Start Time: {start_times[(job_id, op_index)].x}, End Time: {end_times[(job_id, op_index)].x}")
    else:
        print("Optimal çözüm bulunamadı!")

# JSON dosyasından veriyi yükle
data = load_data_from_json('benchmark.json')

# Veriyi kontrol et ve ardından fonksiyonu çağır
if isinstance(data, list):  # JSON artık bir liste, kontrolü buna göre yapıyoruz
    solve_fjs(data)
else:
    print("Geçersiz veri yapısı. Lütfen JSON dosyasını kontrol edin.")
