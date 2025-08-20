import streamlit as st
import pandas as pd
import numpy as np
import onnxruntime as ort
import datetime
import os

# -------------------- Page Configuration --------------------
st.set_page_config(page_title="Unit 2 MW Prediction", layout="centered")

# -------------------- Language Toggle --------------------
LANG = st.selectbox("🌐 Language / اللغة", ["English", "Arabic"])

labels = {
    "English": {
        "title": "Unit 2 Power Output Prediction (MW)",
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
        "designer": "Created by Eng. Mohammed Assaf",
        "model_info": "Model Info",
        "trained": "Trained Date: 2025-08-20",
        "algo": "Algorithm: Random Forest Regressor",
        "importance": "Input Importance"
    },
    "Arabic": {
        "title": "توقع القدرة الكهربائية للوحدة 2 (ميجاواط)",
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
        "designer": "إنشاء م. محمد عساف",
        "model_info": "معلومات النموذج",
        "trained": "تاريخ التدريب: 2025-08-20",
        "algo": "الخوارزمية: غابة عشوائية",
        "importance": "أهمية المدخلات"
    }
}
l = labels[LANG]

# -------------------- Logo --------------------
try:
    st.image("OMCO_Logo.png", width=250)
except:
    st.info("Logo not found (OMCO_Logo.png)")

# -------------------- Title and Designer --------------------
st.title(f"⚡ {l['title']}")
st.markdown(f"<div style='text-align: center;'><strong>{l['designer']}</strong></div>", unsafe_allow_html=True)

# -------------------- Load ONNX Model with Optimization --------------------
from onnxruntime import SessionOptions, GraphOptimizationLevel

opts = SessionOptions()
opts.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_ALL
session = ort.InferenceSession("unit2mwbig_model.onnx", sess_options=opts)
input_name = session.get_inputs()[0].name

# -------------------- Input Sliders --------------------
steam_flow = st.slider(l["steam"], 180.0, 910.0, 850.0)
hrh_p = st.slider(l["hrh_p"], 1.2, 4.39, 4.0)
hrh_t = st.slider(l["hrh_t"], 390, 540, 525)
main_p = st.slider(l["main_p"], 7.0, 17.39, 16.0)
hp_t = st.slider(l["hp_t"], 390, 540, 538)
ambient = st.slider(l["ambient"], -4.0, 50.0, 25.0)

# -------------------- Prediction --------------------
if st.button(l["predict"]):
    input_data = np.array([[
        steam_flow, hrh_p, hrh_t, main_p, hp_t, ambient
    ]], dtype=np.float32)

    result = session.run(None, {input_name: input_data})
    predicted_mw = result[0][0][0]
    clipped_result = max(min(predicted_mw, 290), 140)

    st.markdown(f"<h2 style='color:darkblue;'>{l['output']}: {clipped_result:.2f} MW</h2>", unsafe_allow_html=True)

    # Log result
    log = {
        "Time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Steam Flow": steam_flow, "HRH P": hrh_p, "HRH T": hrh_t,
        "Main Steam P": main_p, "HP Temp": hp_t, "Ambient": ambient,
        "Predicted MW": clipped_result
    }
    df = pd.DataFrame([log])
    if os.path.exists("unit2_log.csv"):
        df.to_csv("unit2_log.csv", mode='a', header=False, index=False)
    else:
        df.to_csv("unit2_log.csv", index=False)

# -------------------- Log Viewer --------------------
st.markdown("---")
if st.checkbox(l["log"]):
    if os.path.exists("unit2_log.csv"):
        df_log = pd.read_csv("unit2_log.csv")
        st.dataframe(df_log)
        csv = df_log.to_csv(index=False).encode('utf-8')
        st.download_button(label="📥 " + l["download"], data=csv, file_name="unit2_log.csv", mime="text/csv")
    else:
        st.info("📭 No predictions logged yet.")

# -------------------- Clear Log --------------------
if st.button("🧹 " + l["clear"]):
    if os.path.exists("unit2_log.csv"):
        os.remove("unit2_log.csv")
        st.success("✅ Log cleared successfully.")

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
