import streamlit as st


# ใช้ HTML เพื่อปรับขนาดฟอนต์
st.markdown("<h1 style='font-size: 40px;'>แนวทางพัฒนา Machine Learning</h1>", unsafe_allow_html=True)
st.markdown("<h2 style='font-size: 20px;'>การพัฒนาโมเดลทำนายคุณภาพไวน์</h2>", unsafe_allow_html=True)
st.markdown("<h2 style='font-size: 20px;'>1. การเตรียมข้อมูล (Data Preparation)</h2>", unsafe_allow_html=True)

def show():
    st.write("1.1 การนำเข้าชุดข้อมูล")
    st.write("- ใช้ชุดข้อมูล Wine Quality ซึ่งมีข้อมูลคุณลักษณะทางเคมีของไวน์ เช่น ปริมาณแอลกอฮอล์ ความเป็นกรด")
    st.write("1.2 การตรวจสอบข้อมูล")
    st.write("- ตรวจสอบค่าว่าง (Missing Values) และใช้วิธีเติมค่าที่เหมาะสม เช่น ค่าเฉลี่ย (Mean)")
    st.write("- แปลงค่าข้อมูลให้เป็นตัวเลขเพื่อให้โมเดลสามารถนำไปใช้ได้")
    st.write("- ใช้ MinMaxScaler เพื่อปรับค่าคุณลักษณะให้อยู่ในช่วงมาตรฐาน เพื่อช่วยให้โมเดลเรียนรู้ได้ดีขึ้น ")
show()
st.markdown("<h2 style='font-size: 20px;'>2. ทฤษฎีของอัลกอริทึมที่พัฒนา (Algorithm Theory)</h2>", unsafe_allow_html=True)

def show():
    st.write("เลือกใช้ Support Vector Machine (SVM) และ Artificial Neural Network (ANN) ")
    st.write(" ")
    st.write(" ")
    st.write(" ")
    st.write(" ")
    st.write(" ")
show()