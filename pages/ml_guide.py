import streamlit as st


st.markdown("<h1 style='font-size: 40px;'>แนวทางพัฒนา Machine Learning</h1>", unsafe_allow_html=True)
st.markdown("<h2 style='font-size: 20px;'>การพัฒนาโมเดลทำนายคุณภาพไวน์</h2>", unsafe_allow_html=True)

st.markdown("<h2 style='font-size: 20px;'>1. การเตรียมข้อมูล (Data Preparation)</h2>", unsafe_allow_html=True)

def show():
    st.write("1.1 การนำเข้าชุดข้อมูล")
    st.write("- ใช้ pandas เพื่อโหลดข้อมูลจากไฟล์ CSV (winequality-red.csv) ซึ่งเป็นข้อมูลเกี่ยวกับคุณภาพไวน์แดง ")
    st.write("1.2 การตรวจสอบข้อมูล")
    st.write("- จัดการข้อมูลที่หายไปโดยเติมค่าโหมดสำหรับข้อมูลประเภทข้อความและค่าเฉลี่ยสำหรับข้อมูลตัวเลข")
    st.write("- แยก Features (X) และ Target (y) โดย Target คือคอลัมน์ quality")
    st.write("- จัดการข้อมูลไม่สมดุลด้วยเทคนิค SMOTE")
    st.write("- ปรับขนาดข้อมูลด้วย MinMaxScaler")

show()
st.markdown("<h2 style='font-size: 20px;'>2. ทฤษฎีของอัลกอริทึมที่พัฒนา (Algorithm Theory)</h2>", unsafe_allow_html=True)

def show():
    
    st.write("เลือกใช้ KNN และ Random Forest ")
    st.write("2.1 KNN")
    st.write("- ใช้เพื่อนบ้านที่ใกล้ที่สุด k ตัวในการจำแนกข้อมูล ข้อดีคือใช้งานง่าย")
    st.write("2.2 Random Fores ")
    st.write("- ใช้การสร้างต้นไม้หลายต้นเพื่อเพิ่มความแม่นยำ ข้อดีคือไม่ต้องการการปรับขนาดข้อมูล")
show()

st.markdown("<h2 style='font-size: 20px;'>3. ขั้นตอนการพัฒนาโมเดล (Model Development)</h2>", unsafe_allow_html=True)
def show():
    st.write("3.1 การแบ่งชุดข้อมูล (Train-Test Split)")
    st.write("- แบ่งข้อมูลเป็น ชุดฝึกสอน (Training Set) 80% และ ชุดทดสอบ (Test Set) 20%")
    st.write("3.2 การฝึกโมเดล (Model Training)")
    st.write("- สำหรับ KNN: ทดลองค่า k ตั้งแต่ 1 ถึง 20 เพื่อหาค่า k ที่ให้ความแม่นยำสูงสุด")
    st.write("- สำหรับ Random Forest: ใช้ GridSearchCV เพื่อหาพารามิเตอร์ที่ดีที่สุด (n_estimators และ max_depth)")
    st.write("3.3 การทดสอบและประเมินผล (Model Evaluation)")
    st.write("- ใช้ตัวชี้วัดต่าง ๆ เช่น Accuracy, Precision, Recall และ F1-Score")
    st.write("3.4 การ Deploy โมเดล ")
    st.write("- บันทึกโมเดลที่ผ่านการฝึกแล้วเป็นไฟล์ .pkl ด้วย joblib")
    st.write("- สร้าง Web Application ด้วย Streamlit เพื่อให้สามารถใช้งานโมเดลได้ผ่านหน้าเว็บ")
show()

