import streamlit as st
import os
import zipfile
import joblib
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
st.sidebar.header("STEP 1: Upload Dataset")
uploaded_zip = st.sidebar.file_uploader("Upload ZIP of Dataset", type=["zip"])
if uploaded_zip is not None:
    extract_dir = "dataset/extracted"
    os.makedirs(extract_dir, exist_ok=True)
    with zipfile.ZipFile(uploaded_zip, "r") as zip_ref:
        zip_ref.extractall(extract_dir)
    st.success("✅ Dataset extracted. Ready to train!")

# Button to trigger dataset preparation
st.sidebar.header("STEP 2: Prepare Dataset")
if st.sidebar.button('Prepare Dataset'):
    st.write("🔧 Preparing dataset...")
    progress_bar = st.progress(0)  # Initialize progress bar for dataset preparation
    try:
        prepare_lfw_dataset(extracted_dir, processed_dir, progress_callback=progress_bar.progress)
        st.success("✅ Dataset prepared successfully!")
    except Exception as e:
        st.error(f"⚠️ Error preparing dataset: {e}")

# Model training
st.sidebar.header("STEP 3: Train Model")
if st.sidebar.button("Train Model"):
    st.write("🤖 Training model...")
    progress_bar = st.progress(0)  # Initialize progress bar for training

    try:
        # Pass progress_bar to the training function
        train_face_recognizer(dataset_path, model_path, progress_callback=progress_bar.progress)

        st.success("🎉 Model trained successfully!")
    except Exception as e:
        st.error(f"⚠️ Error during model training: {e}")

# Image prediction
st.sidebar.header("STEP 4: Image Prediction")
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
            st.error("⚠️ Trained model not found. Please train the model first.")

    else:
        st.error("❌ No face detected in the image.")

# Show sample faces
st.sidebar.header("Optional: Show Sample Faces")
if st.sidebar.button("Show Sample Faces"):
    display_sample_faces(processed_dir)
