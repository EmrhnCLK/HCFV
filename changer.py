def parse_mk01_format(filepath):
    with open(filepath, 'r') as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]

    header = lines[0].split()
    num_jobs = int(header[0])
    num_machines = int(header[1])

    jobs = []
    for j_idx, line in enumerate(lines[1:]):
        tokens = list(map(int, line.strip().split()))
        operations = []
        i = 0

        try:
            num_operations = tokens[i]
            i += 1
            for op_idx in range(num_operations):
                machine_count = tokens[i]
                i += 1
                machines = []
                for _ in range(machine_count):
                    machine_id = tokens[i] + 1  # convert 0-based to 1-based
                    duration = tokens[i + 1]
                    machines.append({"machine": str(machine_id), "duration": duration})
                    i += 2
                operations.append({"machines": machines})
        except Exception as e:
            print(f"⚠️  HATA: Job {j_idx}, Op {len(operations)} sırasında -> {e}")

        jobs.append({"job": chr(65 + j_idx), "operations": operations})

    return jobs

if __name__ == '__main__':
    input_path = 'mk01.txt'  # Brandimarte formatındaki dosya
    output_path = 'benchmark.json'

    result = parse_mk01_format(input_path)

    import json
    with open(output_path, 'w') as f:
        json.dump(result, f, indent=2)

    print(f"✅ Dönüştürme tamamlandı. JSON dosyası: {output_path}")
