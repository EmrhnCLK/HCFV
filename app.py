import streamlit as st
import json
from changer import parse_mk01_format

st.set_page_config(page_title="FJSP Builder", layout="wide")
st.title("🛠️ Flexible Job Shop Scheduling - JSON Creator")

# ---------------------------
# Session state initialization
# ---------------------------
if 'jobs' not in st.session_state:
    st.session_state.jobs = {}
if 'machines' not in st.session_state:
    st.session_state.machines = []

# ---------------------------
# İş Tanımlama
# ---------------------------
st.sidebar.header("➕ Yeni İş Tanımla")
job_name = st.sidebar.text_input("İş Adı (örn: A)")
if st.sidebar.button("İşi Ekle"):
    if job_name and job_name not in st.session_state.jobs:
        st.session_state.jobs[job_name] = []
        st.sidebar.success(f"İş '{job_name}' eklendi.")
    elif job_name in st.session_state.jobs:
        st.sidebar.warning("Bu iş zaten var.")

# ---------------------------
# Makine Tanımlama
# ---------------------------
st.sidebar.header("🔧 Yeni Makine Tanımla")
machine_name = st.sidebar.text_input("Makine ID (örn: M1)")
if st.sidebar.button("Makineyi Ekle"):
    if machine_name and machine_name not in st.session_state.machines:
        st.session_state.machines.append(machine_name)
        st.sidebar.success(f"Makine '{machine_name}' eklendi.")
    elif machine_name in st.session_state.machines:
        st.sidebar.warning("Bu makine zaten var.")

# ---------------------------
# Makineyi İşe Ata
# ---------------------------
st.sidebar.header("🔗 Makineyi İşe Ata")
if st.session_state.jobs and st.session_state.machines:
    selected_machine = st.sidebar.selectbox("Makine Seç", st.session_state.machines)
    selected_job = st.sidebar.selectbox("İş Seç", list(st.session_state.jobs.keys()))
    duration = st.sidebar.number_input("Süre Gir", min_value=1, step=1)
    if st.sidebar.button("İşe Operasyon Ekle"):
        op = {"machine": selected_machine, "duration": duration}
        st.session_state.jobs[selected_job].append({"machines": [op]})
        st.sidebar.success(f"{selected_machine} → {selected_job} [{duration} br]")

# ---------------------------
# TXT Dosyası Yükleme
# ---------------------------
st.header("📂 Brandimarte Formatlı .txt Dosyası Yükle")
uploaded_file = st.file_uploader("Bir .txt dosyası yükleyin", type="txt")
if uploaded_file:
    content = uploaded_file.read().decode("utf-8")
    with open("temp_uploaded.txt", "w") as f:
        f.write(content)
    try:
        parsed_jobs = parse_mk01_format("temp_uploaded.txt")
        st.session_state.jobs = {job["job"]: job["operations"] for job in parsed_jobs}
        st.success("✅ Dosya başarıyla yüklendi ve işler arayüze aktarıldı.")
    except Exception as e:
        st.error(f"Hata: {e}")

# ---------------------------
# İşler ve Operasyonlar Görsel
# ---------------------------
st.header("📋 Tanımlı İşler ve Operasyonları")
if not st.session_state.jobs:
    st.info("Henüz iş tanımlanmadı.")
else:
    for job, ops in st.session_state.jobs.items():
        with st.expander(f"🔹 İş {job}", expanded=True):
            if ops:
                for i, op in enumerate(ops):
                    machine_lines = []
                    for m in op["machines"]:
                        machine_lines.append(f"Makine: `{m['machine']}` → Süre: `{m['duration']}`")
                    st.markdown(f"- Operasyon {i+1}: " + " | ".join(machine_lines))
            else:
                st.markdown("*Bu iş için henüz operasyon yok.*")

# ---------------------------
# JSON İndirme
# ---------------------------
st.markdown("---")
if st.button("📥 JSON Dosyasını Hazırla ve İndir"):
    jobs_json = [
        {"job": job, "operations": ops} for job, ops in st.session_state.jobs.items()
    ]
    json_data = json.dumps(jobs_json, indent=2)
    st.download_button("benchmark.json olarak indir", json_data, file_name="benchmark.json", mime="application/json")
