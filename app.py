import streamlit as st
import os
import zipfile
import joblib
import cv2
import time
import numpy as np
from PIL import Image
from io import BytesIO
import gc
os.environ["WATCHFILES_DISABLE_GLOBAL_WATCH"] = "1"

from utils.train_model import train_face_recognizer
from utils.prepare_lfw_dataset import prepare_lfw_dataset
from utils.face_utils import get_face_embeddings, display_sample_faces

st.set_page_config(page_title="Face Recognition App", layout="wide")

st.title("🧠 Face Recognition App")
st.sidebar.title("📁 Options")

# Paths
dataset_path = "dataset/processed"
extracted_dir = "dataset/extracted"
model_path = "model/face_recognition_model.pkl"

# Step 1: ZIP dataset uploader
st.sidebar.header("STEP 1:")
uploaded_zip = st.sidebar.file_uploader("Upload ZIP of Dataset", type=["zip"])
if uploaded_zip is not None:
    os.makedirs(extracted_dir, exist_ok=True)
    with zipfile.ZipFile(uploaded_zip, "r") as zip_ref:
        zip_ref.extractall(extracted_dir)
    st.success("✅ Dataset extracted. Ready to train!")

# Step 2: Prepare dataset
st.sidebar.header("STEP 2:")
if st.sidebar.button('Prepare Dataset'):
    with st.spinner("🔧 Preparing dataset..."):
        prepare_lfw_dataset(extracted_dir, dataset_path)
        st.success("✅ Dataset prepared.")

# Step 3: Train model
st.sidebar.header("STEP 3:")
if st.sidebar.button("Train Model"):
    st.write("🤖 Training model...")
    progress_bar = st.progress(0)
    time_remaining_text = st.empty()
    start_time = time.time()

    def progress_callback(progress):
        elapsed_time = time.time() - start_time
        estimated_total_time = elapsed_time / progress if progress > 0 else 0
        estimated_remaining_time = estimated_total_time - elapsed_time
        progress_bar.progress(progress)
        time_remaining_text.text(f"⏱️ Estimated time remaining: {int(estimated_remaining_time)} seconds")

    try:
        from utils.train_model import load_dataset
        image_paths, _ = load_dataset(dataset_path)
        total_images = len(image_paths)

        # Limit memory usage
        with st.spinner("🧠 Extracting features and training model in memory-safe batches..."):
            train_face_recognizer(dataset_path, model_path, progress_callback)
            gc.collect()
            st.success("🎉 Model trained successfully!")

    except Exception as e:
        st.error(f"Training error: {e}")

# Step 4: Image prediction
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
        if os.path.exists(model_path):
            model_data = joblib.load(model_path)
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
            st.error("⚠️ Trained model not found.")
    else:
        st.error("❌ No face detected in image.")

# Optional: Display sample faces
st.sidebar.header("Optional")
if st.sidebar.button("Show Sample Faces"):
    display_sample_faces(dataset_path)
