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

# Streamlit UI setup
st.title("🧠 Face Recognition App")
st.sidebar.title("📁 Options")

# Paths
dataset_path = "dataset/processed"
extracted_dir = 'dataset/extracted'
processed_dir = 'dataset/processed'
model_path = "model/face_recognition_model.pkl"

# ZIP dataset uploader
st.sidebar.header("STEP 1:")
uploaded_zip = st.sidebar.file_uploader("Upload ZIP of Dataset", type=["zip"])
if uploaded_zip is not None:
    extract_dir = "dataset/extracted"
    os.makedirs(extract_dir, exist_ok=True)
    with zipfile.ZipFile(uploaded_zip, "r") as zip_ref:
        zip_ref.extractall(extract_dir)
    st.success("✅ Dataset extracted. Ready to train!")

# Button to trigger dataset preparation
st.sidebar.header("STEP 2:")
if st.sidebar.button('Prepare Dataset'):
    st.write("🔧 Preparing dataset...")
    prepare_lfw_dataset(extracted_dir, processed_dir)

st.sidebar.header("STEP 3:")

# Track training state in session
if "is_training" not in st.session_state:
    st.session_state.is_training = False
if "train_complete" not in st.session_state:
    st.session_state.train_complete = False

def start_training():
    st.session_state.is_training = True
    st.session_state.train_complete = False

# Button click sets training flag
if st.sidebar.button("Train Model"):
    start_training()

# If training has started
if st.session_state.is_training and not st.session_state.train_complete:
    st.write("🤖 Training model...")
    progress_bar = st.progress(0)
    status_text = st.empty()

    import time  # Used to show estimated time remaining
    start_time = time.time()

    try:
        def update_progress(p):
            elapsed = time.time() - start_time
            remaining = (elapsed / p - elapsed) if p > 0 else 0
            progress_bar.progress(p)
            status_text.text(f"⏱️ Estimated time remaining: {int(remaining)} seconds")

        # Call training function with embedded progress handler
        train_face_recognizer(dataset_path, model_path, progress_callback=update_progress)

        st.success("🎉 Model trained successfully!")
        st.session_state.train_complete = True
        st.session_state.is_training = False
    except Exception as e:
        st.error(f"Training error: {e}")
        st.session_state.is_training = False

# Image prediction
st.sidebar.header("STEP 4:")
uploaded_image = st.sidebar.file_uploader("Upload Image for Prediction", type=["jpg", "jpeg", "png"])
if uploaded_image is not None:
    st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)

    # Decode image as NumPy array (OpenCV format)
    image = Image.open(uploaded_image).convert('RGB')
    img_np = np.array(image)
    img_cv2 = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    # Extract embedding
    embedding = get_face_embeddings(img_cv2)

    if embedding is not None:
        st.write("✅ Embedding extracted.")

        if os.path.exists(model_path):
            model_data = joblib.load(model_path)
            clf = model_data['model']
            pca = model_data['pca']
            label_encoder = model_data['label_encoder']

            # Apply PCA if available
            if pca:
                embedding = pca.transform([embedding])
            else:
                embedding = np.array([embedding])

            # Predict
            prediction = clf.predict(embedding)
            predicted_label = label_encoder.inverse_transform(prediction)[0]
            st.success(f"🧠 Predicted: {predicted_label}")
        else:
            st.error("⚠️ Trained model not found.")

    else:
        st.error("❌ No face detected in image.")

# Show sample faces
st.sidebar.header("Optional")
if st.sidebar.button("Show Sample Faces"):
    display_sample_faces("dataset/processed")
