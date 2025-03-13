import streamlit as st

#st.title("Project Intelligent System")

def home():
    st.title("Project Intelligent System")
    st.write("")

def page1():
    st.title("หลักการทำงาน")
    st.write("นี่คือเนื้อหาของหน้าที่ 1")

def page2():
    st.title("หน้าที่ 2")
    st.write("นี่คือเนื้อหาของหน้าที่ 2")

def page3():
    st.title("หน้าที่ 3")
    st.write("นี่คือเนื้อหาของหน้าที่ 3")
    
def page4():
    st.title("หน้าที่ 4")
    st.write("นี่คือเนื้อหาของหน้าที่ 4")

def main():
    st.sidebar.title("Details")
    page = st.sidebar.radio("Page", ["HOME", "แนวทางพัฒนา Machine Learning", 
    "แนวทางพัฒนา Neural Network", "Machine Learning Demo", "Neural Network Demo"])
    
    if page == "HOME":
        home()
    elif page == "แนวทางพัฒนา Machine Learning":
        page1()
    elif page == "แนวทางพัฒนา Neural Network":
        page2()
    elif page == "Machine Learning Demo":
        page3()
    elif page == "Neural Network Demo":
        page4()
if __name__ == "__main__":
    main()