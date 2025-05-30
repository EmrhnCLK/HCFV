import json
from pathlib import Path

def parse_mk01_format(filepath):
    with open(filepath, 'r') as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]

    header = lines[0].split()
    num_jobs = int(header[0])  # not used explicitly, but parsed for validation
    num_machines = int(header[1])

    jobs = []
    for j_idx, line in enumerate(lines[1:]):
        tokens = list(map(int, line.split()))
        operations = []
        i = 0
        try:
            num_operations = tokens[i]
            i += 1
            for _ in range(num_operations):
                machine_count = tokens[i]
                i += 1
                machines = []
                for _ in range(machine_count):
                    machine_id = tokens[i] + 1  # 0-based to 1-based
                    duration = tokens[i + 1]
                    machines.append({"machine": str(machine_id), "duration": duration})
                    i += 2
                operations.append({"machines": machines})
        except Exception as e:
            print(f"⚠️ HATA: Job {j_idx}, Op {len(operations)} sırasında -> {e}")
        jobs.append({"job": chr(65 + j_idx), "operations": operations})
    return jobs

if __name__ == '__main__':
    # Uploader tarafından kaydedilen .txt dosyasının yolu
    input_path = Path(__file__).resolve().parent / 'instance.txt'
    if not input_path.exists():
        print(f"⚠️ Girdi dosyası bulunamadı: {input_path}")
        exit(1)

    # Çıktı klasörü ve dosya adı
    output_dir = Path('/home/xibalba/Masaüstü/B-T-RME')
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / 'instance.json'

    # Dönüştürme işlemi
    result = parse_mk01_format(str(input_path))
    with open(output_path, 'w') as f:
        json.dump(result, f, indent=2)

    print(f"✅ Dönüştürme tamamlandı. JSON dosyası: {output_path}")
