import streamlit as st
import joblib
import numpy as np

@st.cache_resource
def load_model():
    return joblib.load("models/ml_model.pkl")  # โหลดโมเดลจากโฟลเดอร์ models/

model = load_model()

# รายชื่อของ 11 Features
feature_names = [
    "Alcohol", "Malic Acid", "Ash", "Alcalinity of Ash", "Magnesium",
    "Total Phenols", "Flavanoids", "Nonflavanoid Phenols", 
    "Proanthocyanins", "Color Intensity", "Hue"
]

def show():
    st.title("Machine Learning Demo")
    
    # รับค่าอินพุตจากผู้ใช้ (ต้องมี 11 Feature)
    input_data = []
    cols = st.columns(3)  # สร้าง layout 3 columns ต่อแถว

    for i, feature in enumerate(feature_names):
        with cols[i % 3]:  # จัดเรียงให้แต่ละ feature ไปอยู่ใน 3 columns
            input_data.append(st.number_input(f"{feature}", value=0.0))

    # กดปุ่มเพื่อทำนายผล
    if st.button("🔍 ทำนาย Wine Quality"):
        input_array = np.array(input_data).reshape(1, -1)  # ต้อง reshape เป็น (1, 11)
        prediction = model.predict(input_array)  # นำข้อมูลไปให้โมเดล
        st.write(f"🎯 ผลการทำนาย: {prediction[0]}")

# ✅ เรียกใช้งานฟังก์ชัน show() ตอนท้ายของไฟล์
show()
