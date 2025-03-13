import streamlit as st
import joblib
import numpy as np

@st.cache_resource
def load_model():
    return joblib.load("models/ml_model.pkl")  # โหลดโมเดลจากโฟลเดอร์ models/

@st.cache_resource
def load_scaler():
    return joblib.load("models/scaler.pkl")  # โหลด Scaler ที่ใช้ในตอนเทรน

model = load_model()
scaler = load_scaler()

# รายชื่อของ 11 Features
feature_names = [
    "Alcohol", "Malic Acid", "Ash", "Alcalinity of Ash", "Magnesium",
    "Total Phenols", "Flavanoids", "Nonflavanoid Phenols", 
    "Proanthocyanins", "Color Intensity", "Hue"
]

def show():
    st.title("🍷Wine Quality Prediction")
    
    # รับค่าอินพุตจากผู้ใช้ (ต้องมี 11 Feature)
    input_data = []
    cols = st.columns(4)  # จัด Layout เป็น 4 คอลัมน์ต่อแถว

    for i, feature in enumerate(feature_names):
        with cols[i % 4]:  # จัดเรียงให้แต่ละ feature ไปอยู่ใน 4 columns
            input_data.append(st.number_input(f"{feature}", value=0.0))

    # กดปุ่มเพื่อทำคาดการณ์ผล
    if st.button("🔍 คาดการณ์ Wine Quality"):
        if all(value == 0.0 for value in input_data):
            st.warning("⚠ กรุณากรอกค่าข้อมูลก่อนทำการคาดการณ์!")
        else:
            # แปลงค่า input ด้วย Scaler ก่อนใช้โมเดล
            input_scaled = scaler.transform([np.array(input_data)])  
            prediction = model.predict(input_scaled)
            st.success(f"🍷 ผลการคาดการณ์คุณภาพไวน์: {prediction[0]}")

show()
