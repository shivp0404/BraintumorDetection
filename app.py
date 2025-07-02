import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import cv2
import os

# --- Load Model ---
model = tf.keras.models.load_model('newbrain_tumor_model.h5', compile=False)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# --- Preprocessing Function ---
def preprocess_image(image):
    image = image.convert("RGB")
    image = image.resize((128, 128))
    image = np.array(image)
    image = image / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# --- Class Labels ---
CLASS_LABELS = ["No Tumor", "Pituitary Tumor", "Meningioma Tumor", "Glioma Tumor"]

# --- Streamlit App UI ---
st.set_page_config(page_title="Brain Tumor Detection", layout="centered")

st.title("🧠 Brain Tumor Detection App")
st.write("Upload an MRI scan image to detect the type of brain tumor .")

st.sidebar.header("ℹ️ About")
st.sidebar.info("""
Welcome to the Brain Tumor Detection App!
Upload your MRI scan image to know if it shows signs of a tumor.
""")

uploaded_file = st.file_uploader("📤 Upload an MRI Image", type=["jpg", "png", "jpeg", "webp"])

if uploaded_file is not None:
    st.subheader("📷 Uploaded MRI Scan")

    image = Image.open(uploaded_file)

    # Layout using columns: First column for image, second column for result
    col1, col2 = st.columns([3, 2.5])  # ratio 1:2 for image and result

    # Display image in first column with smaller width
    with col1:
        st.image(image, caption="Uploaded MRI Scan", width=300)

    # Add vertical space between image and prediction result
    st.write("")  # Adding a blank line for space

    # Prediction in second column
    with col2:
        st.write("🔄 Processing Image...")
        processed_image = preprocess_image(image)

        # Prediction
        prediction = model.predict(processed_image)
        predicted_class = np.argmax(prediction)

        result = CLASS_LABELS[predicted_class]
        
        # Show result with confidence
        st.subheader("🎯 Prediction Result:")
        st.success(f"**{result}**")

else:
    st.info("👈 Please upload an MRI image to proceed.")
