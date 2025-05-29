#!/usr/bin/env python3
# consensus_fix.py
import json
import pathlib
import collections
import itertools

# Ayarlar
POOL_DIR = pathlib.Path("pool")
THETA = 0.8      # %80 üzerinde ortak sıralamaları dondur
ALPHA = 0.20     # %5 makine zamanı toleransı ile LST hesap

def load_pool():
    files = sorted(POOL_DIR.glob("*.json"))
    pool = []
    for f in files:
        data = json.loads(f.read_text())
        pool.append(data)
    return pool

def consensus_pairs(pool):
    # (makine, önce op, sonra op) frekans sayacı
    freq = collections.Counter()
    for sched in pool:
        seqs = sched["machine_sequences"]
        for m, seq in seqs.items():
            for a, b in zip(seq, seq[1:]):
                freq[(m, a, b)] += 1
    limit = THETA * len(pool)
    # %80 ve üzeri ortak sıralamalar
    return {k for k, v in freq.items() if v >= limit}

def time_windows(pool):
    # en iyi makespan’a göre EST/LST pencereleri
    best = min(pool, key=lambda s: s["makespan"])
    cmax = best["makespan"]
    # her op için EST = en küçük start, LST = EST + ALPHA*cmax
    est = {}
    for op in best["start"]:
        est[op] = min(s["start"][op] for s in pool)
    lst = {op: est[op] + ALPHA * cmax for op in est}
    return est, lst

def main():
    pool = load_pool()
    print(f"🔍 Havuzdan {len(pool)} çözüm yüklendi.")
    FIX = consensus_pairs(pool)
    est, lst = time_windows(pool)
    out = {
        "fix": [list(item) for item in FIX],
        "est": est,
        "lst": lst
    }
    pathlib.Path("consensus.json").write_text(json.dumps(out, indent=2))
    print("🎯  consensus.json hazır")

if __name__ == "__main__":
    main()
