import streamlit as st
import os
os.environ["WATCHFILES_DISABLE_GLOBAL_WATCH"] = "1"
import joblib
import zipfile
import cv2
import numpy as np
from PIL import Image
from io import BytesIO
from utils.train_model import train_face_recognizer
from utils.prepare_lfw_dataset import prepare_lfw_dataset
from utils.face_utils import get_face_embeddings, display_sample_faces
import pandas as pd

# UI Setup
st.title("🧠 Face Recognition App")
st.sidebar.title("📁 Options")

# Paths
dataset_path = "dataset/processed"
extracted_dir = 'dataset/extracted'
processed_dir = 'dataset/processed'
model_path = "model/face_recognition_model.pkl"

# Session State
if "is_training" not in st.session_state:
    st.session_state.is_training = False
if "train_complete" not in st.session_state:
    st.session_state.train_complete = False
if "training_report" not in st.session_state:
    st.session_state.training_report = None
if "model_data" not in st.session_state:
    st.session_state.model_data = None

# Step 1: Upload ZIP
st.sidebar.header("STEP 1:")
uploaded_zip = st.sidebar.file_uploader("Upload ZIP of Dataset", type=["zip"])
if uploaded_zip is not None:
    os.makedirs(extracted_dir, exist_ok=True)
    with zipfile.ZipFile(uploaded_zip, "r") as zip_ref:
        zip_ref.extractall(extracted_dir)
    st.success("✅ Dataset extracted.")

# Step 2: Prepare Dataset
st.sidebar.header("STEP 2:")
if st.sidebar.button('Prepare Dataset'):
    st.write("🔧 Preparing dataset...")
    prepare_lfw_dataset(extracted_dir, processed_dir)
    st.success("✅ Dataset prepared!")

# Step 3: Train Model
st.sidebar.header("STEP 3:")

def start_training():
    st.session_state.is_training = True
    st.session_state.train_complete = False
    st.session_state.training_report = None

if st.sidebar.button("Train Model"):
    start_training()

if st.session_state.is_training and not st.session_state.train_complete:
    st.write("🤖 Training model...")
    progress_bar = st.progress(0)
    status_text = st.empty()

    import time
    start_time = time.time()

    def update_progress(p):
        elapsed = time.time() - start_time
        remaining = (elapsed / p - elapsed) if p > 0 else 0
        progress_bar.progress(p)
        status_text.text(f"⏱️ Estimated time remaining: {int(remaining)} seconds")

    try:
        report = train_face_recognizer(dataset_path, model_path, progress_callback=update_progress)
        st.success("🎉 Model trained successfully!")
        st.session_state.train_complete = True
        st.session_state.is_training = False
        st.session_state.training_report = report
        st.session_state.model_data = joblib.load(model_path)
    except Exception as e:
        st.error(f"Training error: {e}")
        st.session_state.is_training = False

# Step 3.5: Upload Pretrained Model
st.sidebar.header("OR: Upload Trained Model")
uploaded_model = st.sidebar.file_uploader("Upload .pkl model", type=["pkl"])
if uploaded_model is not None:
    try:
        st.session_state.model_data = joblib.load(uploaded_model)
        st.success("✅ Model uploaded and ready.")
    except Exception as e:
        st.error(f"❌ Failed to load model: {e}")

# Show training report
if st.session_state.train_complete and st.session_state.training_report:
    st.subheader("📊 Evaluation Report")
    report_df = pd.DataFrame(st.session_state.training_report).transpose()
    st.dataframe(report_df.style.format(precision=2))

# Offer download of trained model
if st.session_state.model_data:
    with open(model_path, "rb") as f:
        model_bytes = f.read()
    st.download_button("⬇️ Download Trained Model", model_bytes, file_name="face_recognition_model.pkl")

# Step 4: Predict
st.sidebar.header("STEP 4:")
uploaded_image = st.sidebar.file_uploader("Upload Image for Prediction", type=["jpg", "jpeg", "png"])
if uploaded_image is not None:
    st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)

    image = Image.open(uploaded_image).convert('RGB')
    img_np = np.array(image)
    img_cv2 = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    embedding = get_face_embeddings(img_cv2)

    if embedding is not None:
        st.write("✅ Embedding extracted.")
        model_data = st.session_state.model_data

        if model_data:
            clf = model_data['model']
            pca = model_data['pca']
            label_encoder = model_data['label_encoder']

            if pca:
                embedding = pca.transform([embedding])
            else:
                embedding = np.array([embedding])

            prediction = clf.predict(embedding)
            predicted_label = label_encoder.inverse_transform(prediction)[0]
            st.success(f"🧠 Predicted: {predicted_label}")
        else:
            st.error("⚠️ No trained model loaded.")
    else:
        st.error("❌ No face detected in image.")

# Optional: Sample faces
st.sidebar.header("Optional")
if st.sidebar.button("Show Sample Faces"):
    display_sample_faces("dataset/processed")
