import streamlit as st


st.markdown("<h1 style='font-size: 40px;'>แนวทางพัฒนา Neural Network</h1>", unsafe_allow_html=True)
st.markdown("<h2 style='font-size: 20px;'>การพัฒนาโมเดลทำนายชนิดดอกไม้</h2>", unsafe_allow_html=True)

st.markdown("<h2 style='font-size: 20px;'>1. การเตรียมข้อมูล (Data Preparation)</h2>", unsafe_allow_html=True)

def show():
    st.write("1.1 การนำเข้าชุดข้อมูล")
    st.write("- เชื่อมต่อ Google Drive เและเข้าถึงข้อมูลที่เก็บไว้ในโฟลเดอร์ flowers")
    st.write("- รูปภาพถูกปรับขนาดเป็น (128, 128) และแบ่งเป็นแบตช์ขนาด 32")
    st.write("1.2 การเพิ่มความหลากหลายของข้อมูล ")
    st.write("- สร้าง Sequential layer สำหรับการเพิ่มความหลากหลายของข้อมูล")

show()
st.markdown("<h2 style='font-size: 20px;'>2. ทฤษฎีของอัลกอริทึมที่พัฒนา (Algorithm Theory)</h2>", unsafe_allow_html=True)

def show():
    st.write("เลือกใช้ CNN ")
    st.write("- เพราะเหมาะสำหรับการประมวลผลข้อมูลที่มีโครงสร้างกริด (grid-like structure) เช่น ภาพ")

show()

st.markdown("<h2 style='font-size: 20px;'>3. ขั้นตอนการพัฒนาโมเดล (Model Development)</h2>", unsafe_allow_html=True)
def show():
    st.write("3.1 การแบ่งชุดข้อมูล (Train-Test Split)")
    st.write("- แบ่งข้อมูลเป็น ชุดฝึกสอน (Training Set) 80% และ ชุดทดสอบ (Test Set) 20%")
    st.write("3.2 การฝึกโมเดล (Model Training)")
    st.write("- ใช้ Transfer Learning และเพิ่ม Fully Connected Layer (Dense) เพื่อลด Overfitting")
    st.write("3.3 การทดสอบและประเมินผล (Model Evaluation)")
    st.write("- ใช้ตัวชี้วัด  Accurac และ แสดงกราฟความแม่นยำ")
    st.write("3.4 การ Deploy โมเดล ")
    st.write("- บันทึกโมเดลที่ผ่านการฝึกแล้วเป็นไฟล์ .keras ")
    st.write("- สร้าง Web Application ด้วย Streamlit เพื่อให้สามารถใช้งานโมเดลได้ผ่านหน้าเว็บ")
show()

