import streamlit as st
import subprocess
import sys
from pathlib import Path

# Sayfa yapılandırması
st.set_page_config(page_title="Flexible Job Shop Scheduling", layout="wide")

# Proje dizini
PROJECT_DIR = Path(__file__).resolve().parent

# ---------------------
# 📄 TXT Dosyası Yükleme ve JSON Dönüştürme
uploaded_txt = st.file_uploader(
    label="instance.txt dosyasını sürükle-bırak ile yükleyin", 
    type="txt"
)
if uploaded_txt:
    # .txt dosyasını kaydet
    txt_path = PROJECT_DIR / "instance.txt"
    with open(txt_path, "wb") as f:
        f.write(uploaded_txt.getvalue())
    st.success(".txt dosyası alındı ve 'instance.txt' olarak kaydedildi.")

    # changer.py ile JSON'e dönüştür
    conv_exp = st.expander("JSON Dönüştürme Logları", expanded=True)
    with conv_exp:
        proc = subprocess.Popen(
            [sys.executable, str(PROJECT_DIR / "changer.py")],
            cwd=PROJECT_DIR,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        for line in proc.stdout:
            st.code(line.rstrip(), language="text")
        proc.wait()

    st.success("instance.json başarıyla oluşturuldu ve ilgili klasöre kaydedildi.")
# ---------------------

# Çözücüler
EXACT = {"Kesin Çözücü (Gurobi Exact)": PROJECT_DIR / "gurobi.py"}
HEURISTICS = {
    "Genetic Algorithm": PROJECT_DIR / "Genetic_algorithm.py",
    "Simulated Annealing": PROJECT_DIR / "Simulated_annealing.py",
    "Particle Swarm Optimization": PROJECT_DIR / "pso.py",
}
CONSENSUS = {"Consensus Fix": PROJECT_DIR / "consensus_fix.py"}
HCVF = {"Gurobi (HCVF)": PROJECT_DIR / "gurobi_hcvf.py"}

# Başlık ve açıklama
st.title("Flexible Job Shop Scheduling")
st.markdown(
    """
    Bu uygulama ile:
    0️⃣ **Kesin Çözücü** (Gurobi Exact) ile tam optimum çözümü alın,  
    1️⃣ **Sezgisel Algoritmalar** (GA, SA, PSO) çalıştırın,  
    2️⃣ **Consensus Fix** ile ortak çözüm oluşturun,  
    3️⃣ **Gurobi (HCVF)** ile optimize edilmiş çözümü elde edin.

    JSON dosyaları `pool/` klasörüne otomatik kaydedilir.
    """
)

# 0️⃣ Kesin Çözücü
st.subheader("🧮 Kesin Çözücü (Gurobi Exact)")
for name, script_path in EXACT.items():
    if st.button(f"🚀 {name}"):
        exp = st.expander(f"{name} Logları", expanded=True)
        with exp:
            proc = subprocess.Popen(
                [sys.executable, str(script_path)],
                cwd=PROJECT_DIR,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            for line in proc.stdout:
                st.code(line.rstrip(), language="text")
            proc.wait()

st.markdown("---")

# 1️⃣ Sezgisel Algoritmalar
st.subheader("🧠 Sezgisel Algoritmalar")
cols = st.columns(3)
for idx, (name, script_path) in enumerate(HEURISTICS.items()):
    with cols[idx]:
        if st.button(f"🚀 {name}"):
            exp = st.expander(f"{name} Logları", expanded=True)
            with exp:
                proc = subprocess.Popen(
                    [sys.executable, str(script_path)],
                    cwd=PROJECT_DIR,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                )
                for line in proc.stdout:
                    st.code(line.rstrip(), language="text")
                proc.wait()

st.markdown("---")

# 2️⃣ Konsensus Birleştirme
st.subheader("🔧 Consensus Maker")
for name, script_path in CONSENSUS.items():
    if st.button(f"🚀 {name}"):
        exp = st.expander(f"{name} Logları", expanded=True)
        with exp:
            proc = subprocess.Popen(
                [sys.executable, str(script_path)],
                cwd=PROJECT_DIR,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            for line in proc.stdout:
                st.code(line.rstrip(), language="text")
            proc.wait()

st.markdown("---")

# 3️⃣ Gurobi (HCVF)
st.subheader("👑 Gurobi (HCVF)")
for name, script_path in HCVF.items():
    if st.button(f"🚀 {name}"):
        exp = st.expander(f"{name} Logları", expanded=True)
        with exp:
            proc = subprocess.Popen(
                [sys.executable, str(script_path)],
                cwd=PROJECT_DIR,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            for line in proc.stdout:
                st.code(line.rstrip(), language="text")
            proc.wait()
