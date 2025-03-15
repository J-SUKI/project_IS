import tensorflow as tf
from tensorflow import keras
from keras import layers, models
import numpy as np
from PIL import Image
import streamlit as st

# ขนาดของภาพ
img_height = 128
img_width = 128

num_classes = 5  

class_names = ["Daisy", "Dandelion", "Rose", "Sunflower", "Tulip"]

# Build the model
model = models.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.2),
    layers.RandomZoom(0.2),
    
    layers.Rescaling(1./255),
    
    tf.keras.applications.MobileNetV2(input_shape=(img_height, img_width, 3),
                                      include_top=False, weights='imagenet'),
    
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.5),
    layers.Dense(num_classes, activation="softmax")
])

# Compile the model (required before predicting)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Streamlit app title and description
st.title("🌸 Flower Classification App")
st.write("อัปโหลดรูปภาพดอกไม้ แล้วทายว่าเป็นดอกอะไร!")

# File uploader widget
uploaded_file = st.file_uploader("อัปโหลดรูปภาพ", type=["jpg", "png"])

if uploaded_file is not None:
    # Open and display the uploaded image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="รูปภาพที่อัปโหลด", use_container_width=True)
    
    # Preprocess the image
    img = image.resize((img_height, img_width))  
    img_array = np.array(img) / 255.0  # Rescale the image array
    
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    
    # Make a prediction
    prediction = model.predict(img_array)
    
    if prediction.shape[-1] == num_classes:  # Check if the output has the expected shape
        predicted_class = np.argmax(prediction)  # Get the index of the highest prediction
        predicted_class_name = class_names[predicted_class]  # Map to class name
        st.success(f"🌼 ผลการคาดการณ์ เป็นดอกชนิด: **{predicted_class_name}**")
