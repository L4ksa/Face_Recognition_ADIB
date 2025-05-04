
import streamlit as st
from deepface import DeepFace
import os
from PIL import Image
import cv2

st.title("Face Recognition App (LFW-based)")

uploaded_file = st.file_uploader("Upload an image", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    image_path = "input.jpg"
    image.save(image_path)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    with st.spinner("Analyzing..."):
        try:
            result = DeepFace.find(img_path=image_path, db_path="known_faces", enforce_detection=False)
            df = result[0]
            if not df.empty:
                identity = os.path.basename(df.iloc[0]['identity'])
                st.success(f"Match Found: {identity}")
            else:
                st.warning("No Match Found")
        except Exception as e:
            st.error(f"Error: {e}")
