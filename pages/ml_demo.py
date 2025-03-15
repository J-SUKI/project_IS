import streamlit as st
import joblib
import numpy as np

@st.cache_resource
def load_model():
    return joblib.load("models/ml_model.pkl")

@st.cache_resource
def load_scaler():
    return joblib.load("models/scaler.pkl") 

model = load_model()
scaler = load_scaler()

feature_names = [
    "Alcohol", "Malic Acid", "Ash", "Alcalinity of Ash", "Magnesium",
    "Total Phenols", "Flavanoids", "Nonflavanoid Phenols", 
    "Proanthocyanins", "Color Intensity", "Hue"
]

def map_quality_label(quality):
    if quality <= 4:
        return f"คุณภาพต่ำ (ระดับ {quality})"
    elif quality <= 6:
        return f"คุณภาพปานกลาง (ระดับ {quality})"
    elif quality <= 8:
        return f"คุณภาพดี (ระดับ {quality})"
    else:
        return f"คุณภาพดีมาก  (ระดับ {quality})"

def show():
    st.title("🍷 Wine Quality Prediction")

    input_data = []
    cols = st.columns(4)  

    for i, feature in enumerate(feature_names):
        with cols[i % 4]:  
            input_data.append(st.number_input(f"{feature}", value=0.0))

    if st.button("🔍 คาดการณ์ Wine Quality"):
        if all(value == 0.0 for value in input_data):
            st.warning("⚠ กรุณากรอกค่าข้อมูลก่อนทำการคาดการณ์!")
        else:
            input_scaled = scaler.transform([np.array(input_data)])
            prediction = model.predict(input_scaled)

            quality_label = map_quality_label(prediction[0])

            st.success(f"🍷 ผลการคาดการณ์คุณภาพไวน์: {quality_label}")

show()
