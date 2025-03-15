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
    
    st.write("เลือกใช้ Support Vector Machine (SVM) และ Artificial Neural Network (ANN) ")
    st.write("2.1 Support Vector Machine (SVM)")
    st.write("- ใช้หลักการ Hyperplane เพื่อแบ่งแยกข้อมูลที่ดีที่สุด ")
    st.write("- ใช้ Kernel Trick เพื่อทำให้สามารถจำแนกข้อมูลที่ไม่เป็นเส้นตรงได้ดีขึ้น")
    st.write("2.2 Artificial Neural Network (ANN) ")
    st.write("- ประกอบด้วยชั้นอินพุต (Input Layer), ชั้นซ่อน (Hidden Layer) และชั้นเอาต์พุต (Output Layer)")
    st.write("- ใช้ฟังก์ชันกระตุ้น (Activation Function) เช่น ReLU และ Softmax เพื่อช่วยในการเรียนรู้ของโมเดล ")
show()

st.markdown("<h2 style='font-size: 20px;'>3. ขั้นตอนการพัฒนาโมเดล (Model Development)</h2>", unsafe_allow_html=True)
def show():
    st.write("3.1 การแบ่งชุดข้อมูล (Train-Test Split)")
    st.write("- แบ่งข้อมูลเป็น ชุดฝึกสอน (Training Set) 80% และ ชุดทดสอบ (Test Set) 20%")
    st.write("- ใช้ Stratified Sampling เพื่อให้การกระจายของข้อมูลเป็นไปอย่างเหมาะสม ")
    st.write("3.2 การฝึกโมเดล (Model Training)")
    st.write("- สำหรับ SVM: กำหนด Kernel เป็น rbf และปรับค่า C และ gamma ให้เหมาะสม")
    st.write("- สำหรับ ANN: ใช้โครงข่ายประสาทเทียมแบบ Feed Forward Neural Network (FNN) และปรับพารามิเตอร์ เช่น จำนวนชั้นซ่อน (Hidden Layers) และจำนวน Neurons")
    st.write("3.3 การปรับแต่งโมเดล (Hyperparameter Tuning)")
    st.write("- ใช้เทคนิค Grid Search หรือ Random Search เพื่อหาค่าพารามิเตอร์ที่ดีที่สุด")
    st.write("- ใช้ Cross Validation (k-Fold) เพื่อหลีกเลี่ยงปัญหา Overfitting")
    st.write("3.4 การทดสอบและประเมินผล (Model Evaluation)")
    st.write("- ใช้ตัวชี้วัดต่าง ๆ เช่น Accuracy, Precision, Recall และ F1-Score")
    st.write("3.5 การ Deploy โมเดล ")
    st.write("- บันทึกโมเดลที่ผ่านการฝึกแล้วเป็นไฟล์ .pkl ด้วย joblib")
    st.write("- สร้าง Web Application ด้วย Streamlit เพื่อให้สามารถใช้งานโมเดลได้ผ่านหน้าเว็บ")
show()

