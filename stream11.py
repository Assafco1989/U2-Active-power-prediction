import os
import csv
import datetime
import numpy as np
import streamlit as st
import onnxruntime as ort

# -------------------- Page Configuration --------------------
st.set_page_config(page_title="Unit 2 MW Prediction", layout="centered")

# -------------------- Performance/env tuning (must be set before creating session) --------------------
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")

# -------------------- Language Toggle --------------------
LANG = st.selectbox("🌐 Language / اللغة", ["English", "Arabic"])

labels = {
    "English": {
        "title": "Unit 2 Active Power Output Prediction (MW)",
        "steam": "Steam Flow (Ton/h)",
        "hrh_p": "HRH Pressure (MPa)",
        "hrh_t": "HRH Temperature (°C)",
        "main_p": "Main Steam Pressure (MPa)",
        "hp_t": "HP Steam Temperature (°C)",
        "ambient": "Ambient Temperature (°C)",
        "predict": "Predict",
        "status": "Status",
        "output": "Predicted Output",
        "log": "Show Prediction Log",
        "download": "Download Log as CSV",
        "clear": "Clear Log",
        "designer": "UI Created by Eng. Mohammed Assaf",
        "model_info": "Model Info",
        "trained": "Trained Date: 2025-08-20",
        "algo": "Algorithm: Random Forest Regressor (ONNX)",
        "importance": "Input Importance"
    },
    "Arabic": {
        "title": "توقع القدرة الكهربائية الفعالة للوحدة 2 (ميجاواط)",
        "steam": "تدفق البخار (طن/ساعة)",
        "hrh_p": "ضغط إعادة التسخين العالي (ميجا باسكال)",
        "hrh_t": "درجة حرارة إعادة التسخين العالي (°م)",
        "main_p": "ضغط البخار الرئيسي (ميجا باسكال)",
        "hp_t": "درجة حرارة البخار العالي (°م)",
        "ambient": "درجة حرارة الجو (°م)",
        "predict": "تنبؤ",
        "status": "الحالة",
        "output": "القيمة المتوقعة",
        "log": "عرض سجل التنبؤات",
        "download": "تحميل السجل",
        "clear": "مسح السجل",
        "designer": "الواجهة التفاعلية إنشاء م. محمد عساف",
        "model_info": "معلومات النموذج",
        "trained": "تاريخ التدريب: 2025-08-20",
        "algo": "الخوارزمية: غابة عشوائية (ONNX)",
        "importance": "أهمية المدخلات"
    }
}
l = labels[LANG]

# -------------------- Logo --------------------
try:
    st.image("OMCO_Logo.png", width=250)
except Exception:
    st.info("Logo not found (OMCO_Logo.png)")

# -------------------- Title and Designer --------------------
st.title(f"⚡ {l['title']}")
st.markdown(f"<div style='text-align: center;'><strong>{l['designer']}</strong></div>", unsafe_allow_html=True)

# -------------------- Optimized ONNX session (cached) --------------------
from onnxruntime import SessionOptions, GraphOptimizationLevel

@st.cache_resource(show_spinner=False)
def get_session(path: str):
    opts = SessionOptions()
    opts.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_ALL
    # Threading tuned for small CPU instances; tweak if needed
    opts.intra_op_num_threads = 2
    opts.inter_op_num_threads = 1
    opts.enable_mem_pattern = True
    opts.enable_cpu_mem_arena = True
    providers = [("CPUExecutionProvider", {
        "arena_extend_strategy": "kSameAsRequested",
        "intra_op_num_threads": 2,
        "inter_op_num_threads": 1
    })]
    return ort.InferenceSession(path, sess_options=opts, providers=providers)

MODEL_PATH = "unit2mwbig_model.onnx"
session = get_session(MODEL_PATH)
input_name = session.get_inputs()[0].name

# -------------------- Input Form (prevents rerun on every slider move) --------------------
with st.form("predict_form"):
    col1, col2 = st.columns(2)
    with col1:
        steam_flow = st.slider(l["steam"], 180.0, 910.0, 850.0)
        hrh_p      = st.slider(l["hrh_p"], 1.2, 4.39, 4.0)
        hrh_t      = st.slider(l["hrh_t"], 390, 540, 525)
    with col2:
        main_p     = st.slider(l["main_p"], 7.0, 17.39, 16.0)
        hp_t       = st.slider(l["hp_t"], 390, 540, 538)
        ambient    = st.slider(l["ambient"], -4.0, 50.0, 25.0)
    submitted = st.form_submit_button(l["predict"])  # only submits once

# -------------------- Prediction --------------------
if submitted:
    x = np.array([[steam_flow, hrh_p, hrh_t, main_p, hp_t, ambient]], dtype=np.float32)
    y = session.run(None, {input_name: x})[0]
    predicted_mw = float(y.ravel()[0])
    clipped_result = max(min(predicted_mw, 290.0), 140.0)

    st.markdown(f"<h2 style='color:darkblue;'>{l['output']}: {clipped_result:.2f} MW</h2>", unsafe_allow_html=True)

    # Fast CSV logging without pandas
    row = [
        datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        steam_flow, hrh_p, hrh_t, main_p, hp_t, ambient, clipped_result
    ]
    file_exists = os.path.exists("unit2_log.csv")
    try:
        with open("unit2_log.csv", "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["Time", "Steam Flow", "HRH P", "HRH T", "Main Steam P", "HP Temp", "Ambient", "Predicted MW"])  # header
            writer.writerow(row)
    except Exception as e:
        st.warning(f"Could not write log: {e}")

# -------------------- Log Viewer --------------------
st.markdown("---")
show_log = st.checkbox(l["log"])  

@st.cache_data(show_spinner=False)
def load_log(path: str):
    import pandas as pd
    if not os.path.exists(path):
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None

if show_log:
    df_log = load_log("unit2_log.csv")
    if df_log is not None:
        st.dataframe(df_log)
        csv_bytes = df_log.to_csv(index=False).encode("utf-8")
        st.download_button(label="📥 " + l["download"], data=csv_bytes, file_name="unit2_log.csv", mime="text/csv")
    else:
        st.info("📭 No predictions logged yet.")

# -------------------- Clear Log --------------------
if st.button("🧹 " + l["clear"]):
    try:
        if os.path.exists("unit2_log.csv"):
            os.remove("unit2_log.csv")
            st.success("✅ Log cleared successfully.")
        else:
            st.info("No log file to clear.")
    except Exception as e:
        st.warning(f"Could not clear log: {e}")

# -------------------- Model Info --------------------
st.markdown("---")
with st.expander(l["model_info"]):
    st.write(f"📅 {l['trained']}")
    st.write(f"🧠 {l['algo']}")
    st.markdown(f"**{l['importance']}:**")
    importance_list = [
        ("Steam Flow", 0.25), ("HRH P", 0.20), ("HRH T", 0.15),
        ("Main Steam P", 0.15), ("HP Temp", 0.15), ("Ambient", 0.10)
    ]
    for name, val in importance_list:
        bar = "█" * int(val * 20)
        st.write(f"{name}: {bar} {int(val * 100)}%")
