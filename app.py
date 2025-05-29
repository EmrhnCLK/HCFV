# -*- coding: utf-8 -*-
"""
Streamlit UI – Flexible Job‑Shop Scheduling Benchmark Builder **ve** Çözücü Karşılaştırma
=====================================================================================
Bu arayüz sayesinde ➡️
  • Brandimarte formatındaki `.txt` dosyaları **veya** elle tanımlanan işler → **benchmark.json** üretilir.
  • **Gurobi (Exact)**, **Genetic Algorithm (GA)** ve **Simulated Annealing (SA)** çözücüleri tek tuşla
    çalıştırılır **ve** LOG’ları anlık olarak terminal bölümüne akıtılır.
  • Makespan & çözüm süresi sonuçları tablo hâlinde kıyaslanır.

Gerekli dosyalar (aynı klasörde):
  changer.py, gurobi.py, Genetic_algorithm.py, Simulated_annealing.py

Not ➜ Gurobi lisansınız yoksa "Gurobi" butonu hata verebilir; loglarda ayrıntıyı görebilirsiniz.
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Any

import streamlit as st
from changer import parse_mk01_format

# -----------------------------------------------------------------------------
# Sabitler & yardımcılar
# -----------------------------------------------------------------------------
PROJECT_DIR = Path(__file__).resolve().parent
SOLVER_SCRIPTS = {
    "Gurobi (Exact)": PROJECT_DIR / "gurobi.py",
    "Genetic Algorithm": PROJECT_DIR / "Genetic_algorithm.py",
    "Simulated Annealing": PROJECT_DIR / "Simulated_annealing.py",
    "Parçacık Sürü Optimizasyonu": PROJECT_DIR / "pso.py",
    

}


def _is_num(s: str) -> bool:
    try:
        float(s)
        return True
    except ValueError:
        return False


def _parse_makespan(output: str) -> float | None:
    """Çözücü çıktısından makespan sayısını çeker (ilk bulduğu)."""
    for line in output.splitlines():
        line = line.strip()
        if not line:
            continue
        lower = line.lower()
        if "makespan" in lower:
            tokens = [t for t in line.replace(":", " ").split() if _is_num(t)]
            if tokens:
                return float(tokens[-1])
    return None


def run_solver_stream(script_path: Path, jobs_json: List[Dict[str, Any]], placeholder: st.delta_generator.DeltaGenerator) -> Dict[str, Any]:
    """Script’i alt‑process olarak başlatır; stdout/stderr satırlarını **anlık** olarak
    verilen *placeholder*’a yazar ve bittiğinde özet döner.
    """
    result: Dict[str, Any] = {"makespan": None, "runtime": 0.0, "stdout": "", "stderr": ""}
    collected_lines: List[str] = []
    with tempfile.TemporaryDirectory() as tmpdir:
        json_path = Path(tmpdir) / "benchmark.json"
        with open(json_path, "w", encoding="utf-8") as fp:
            json.dump(jobs_json, fp, indent=2)

        start = time.perf_counter()
        proc = subprocess.Popen(
            [sys.executable, str(script_path)],
            cwd=tmpdir,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
        )

        # Satır satır oku ↴
        assert proc.stdout is not None  # type: ignore
        for line in proc.stdout:
            collected_lines.append(line)
            # Anlık göster 👉
            placeholder.code("".join(collected_lines), language="text")

        proc.wait()
        end = time.perf_counter()

    output_text = "".join(collected_lines)
    result["runtime"] = end - start
    result["stdout"] = output_text
    result["stderr"] = ""  # stderr zaten stdout’a birleştirildi
    result["makespan"] = _parse_makespan(output_text)
    return result

# -----------------------------------------------------------------------------
# Streamlit Ekranı
# -----------------------------------------------------------------------------
st.set_page_config(page_title="FJSS Solver Bench", layout="wide")
st.title("🛠️ Flexible Job‑Shop Scheduling – Çözücüler ve Karşılaştırma")

# ---------------------------------------------
# Session‑state init
# ---------------------------------------------
if "jobs" not in st.session_state:
    st.session_state.jobs: Dict[str, List[Dict[str, Any]]] = {}
if "machines" not in st.session_state:
    st.session_state.machines: List[str] = []
if "results" not in st.session_state:
    st.session_state.results: Dict[str, Dict[str, Any]] = {}

# ---------------------------------------------
# 1) İş & Makine Tanımı (Sidebar)
# ---------------------------------------------
st.sidebar.header("➕ Yeni İş Tanımla")
_input_job = st.sidebar.text_input("İş Adı (örn: A)")
if st.sidebar.button("İşi Ekle"):
    if _input_job and _input_job not in st.session_state.jobs:
        st.session_state.jobs[_input_job] = []
        st.sidebar.success(f"İş '{_input_job}' eklendi.")
    else:
        st.sidebar.warning("Geçerli ve benzersiz bir iş adı girin.")

st.sidebar.header("🔧 Yeni Makine Tanımla")
_input_mach = st.sidebar.text_input("Makine ID (örn: M1)")
if st.sidebar.button("Makineyi Ekle"):
    if _input_mach and _input_mach not in st.session_state.machines:
        st.session_state.machines.append(_input_mach)
        st.sidebar.success(f"Makine '{_input_mach}' eklendi.")
    else:
        st.sidebar.warning("Geçerli ve benzersiz bir makine ID girin.")

st.sidebar.header("🔗 Makineyi İşe Ata")
if st.session_state.jobs and st.session_state.machines:
    sel_machine = st.sidebar.selectbox("Makine Seç", st.session_state.machines)
    sel_job = st.sidebar.selectbox("İş Seç", list(st.session_state.jobs.keys()))
    dur = st.sidebar.number_input("Süre Gir", min_value=1, step=1)
    if st.sidebar.button("İşe Operasyon Ekle"):
        op = {"machine": sel_machine, "duration": dur}
        st.session_state.jobs[sel_job].append({"machines": [op]})
        st.sidebar.success(f"{sel_machine} → {sel_job} [{dur} br]")

# ---------------------------------------------
# 2) Brandimarte TXT Yükleme
# ---------------------------------------------
st.header("📂 Brandimarte Formatlı .txt Dosyası Yükle")
_uploaded = st.file_uploader("Bir .txt dosyası yükleyin", type="txt")
if _uploaded:
    txt_content = _uploaded.read().decode("utf-8")
    with open("temp_uploaded.txt", "w", encoding="utf-8") as tempf:
        tempf.write(txt_content)
    try:
        parsed_jobs = parse_mk01_format("temp_uploaded.txt")
        st.session_state.jobs = {j["job"]: j["operations"] for j in parsed_jobs}
        st.success("✅ Dosya başarıyla içe aktarıldı.")
    except Exception as exc:
        st.error(f"Hata: {exc}")

# ---------------------------------------------
# 3) Tanımlı İşler
# ---------------------------------------------
st.header("📋 Tanımlı İşler ve Operasyonları")
if not st.session_state.jobs:
    st.info("Henüz iş tanımlanmadı.")
else:
    for job, ops in st.session_state.jobs.items():
        with st.expander(f"🔹 İş {job}", expanded=False):
            if not ops:
                st.markdown("*Bu iş için henüz operasyon yok.*")
            for idx, op in enumerate(ops):
                marks = [f"Makine **{m['machine']}** ⇒ {m['duration']} br" for m in op["machines"]]
                st.markdown(f"**Operasyon {idx+1}:**  " + " • ".join(marks))

# ---------------------------------------------
# 4) JSON İndirme
# ---------------------------------------------
#st.markdown("---")
#if st.button("📥 benchmark.json olarak indir"):
 #   json_jobs = [{"job": j, "operations": ops} for j, ops in st.session_state.jobs.items()]
  #  st.download_button("benchmark.json indir", json.dumps(json_jobs, indent=2), file_name="benchmark.json", mime="application/json")

# -----------------------------------------------------------------------------
# 5) Çözücüleri Çalıştır
# -----------------------------------------------------------------------------
st.markdown("## ⚙️ Çözücüleri Çalıştır & Karşılaştır")
if not st.session_state.jobs:
    st.info("Önce benchmark verisi oluşturun veya yükleyin.")
else:
    # Tüm log’ları tek geniş ekspander’da tutacağız.
    logs_expander = st.expander("📝 Terminal Çıktıları", expanded=True)
    cols = st.columns(len(SOLVER_SCRIPTS))
    jobs_json_data = [{"job": j, "operations": ops} for j, ops in st.session_state.jobs.items()]

    for (solver_name, script_path), col in zip(SOLVER_SCRIPTS.items(), cols):
        with col:
            if st.button(f"🚀 {solver_name}"):
                with logs_expander:
                    st.subheader(solver_name)
                    log_placeholder = st.empty()
                # Çalıştır ve canlı logla
                res = run_solver_stream(script_path, jobs_json_data, log_placeholder)
                st.session_state.results[solver_name] = res
                # Sonuç toast
                if res["makespan"] is not None:
                    st.toast(f"{solver_name} → Makespan {res['makespan']}  |  {res['runtime']:.2f} sn", icon="✅")
                else:
                    st.toast(f"{solver_name} bitti, makespan algılanamadı.", icon="⚠️")

    # Sonuç tablosu
    if st.session_state.results:
        st.markdown("### 📊 Karşılaştırma Tablosu")
        rows = [
            {"Çözücü": n, "Makespan": r["makespan"], "Çözüm Süresi (sn)": f"{r['runtime']:.2f}"}
            for n, r in st.session_state.results.items()
        ]
        st.table(rows)
