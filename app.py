import streamlit as st
import os
import joblib
import numpy as np
import cv2
from PIL import Image
from deepface import DeepFace
from utils.prepare_lfw_dataset import save_lfw_dataset
from utils.train_model import train_face_recognizer
from utils.face_utils import get_face_embeddings
import tempfile

# ---------------------- Configuration ----------------------
st.set_page_config(page_title="Face Recognition System", page_icon=":smiley:")
st.title("Face Recognition System")
st.write("Upload an image or train the model using the LFW dataset.")

MODEL_PATH = "models/face_recognizer.pkl"
DATASET_PATH = "dataset"

# ---------------------- Step 1: Upload Dataset ----------------------
st.sidebar.header("Step 1: Upload and Process Dataset")
uploaded_zip = st.sidebar.file_uploader("Upload LFW ZIP", type="zip")

if uploaded_zip:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".zip") as tmp_zip:
        tmp_zip.write(uploaded_zip.getvalue())
        tmp_zip_path = tmp_zip.name

    with st.spinner("Processing dataset..."):
        try:
            save_lfw_dataset(zip_path=tmp_zip_path, output_dir=DATASET_PATH)
            st.sidebar.success("Dataset successfully processed!")
        except Exception as e:
            st.sidebar.error("Dataset processing failed.")
            st.error(f"Error: {e}")

# ---------------------- Step 2: Train Model ----------------------
st.sidebar.header("Step 2: Train Model")

model_ready = os.path.exists(MODEL_PATH)

def update_progress_bar(progress_bar, status_text, progress):
    progress_bar.progress(progress)
    status_text.text(f"Training Progress: {int(progress * 100)}%")

if not model_ready:
    if st.sidebar.button("Train Model"):
        progress_bar = st.progress(0)
        status_text = st.empty()

        try:
            train_face_recognizer(DATASET_PATH, MODEL_PATH, progress_callback=lambda progress: update_progress_bar(progress_bar, status_text, progress))
            st.sidebar.success("Model trained successfully!")
        except Exception as e:
            st.sidebar.error("Model training failed.")
            st.error(f"Error: {e}")

# ---------------------- Step 3: Recognize Face ----------------------
if model_ready:
    uploaded_image = st.file_uploader("Upload Image for Recognition", type="jpg")

    if uploaded_image:
        st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)

        with st.spinner("Recognizing face..."):
            try:
                model_data = joblib.load(MODEL_PATH)
                img = Image.open(uploaded_image)
                img = np.array(img.convert("RGB"))
                embedding = get_face_embeddings(img)

                if embedding is None:
                    st.error("No face detected!")
                else:
                    pca = model_data['pca']
                    clf = model_data['model']
                    le = model_data['label_encoder']
                    
                    embedding_pca = pca.transform([embedding])
                    predicted_label = clf.predict(embedding_pca)
                    predicted_name = le.inverse_transform(predicted_label)[0]

                    st.success(f"Predicted Name: {predicted_name}")
            except Exception as e:
                st.error(f"Recognition failed: {e}")
