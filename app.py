import streamlit as st
import os
import pickle
import numpy as np
import cv2
import joblib
from PIL import Image
from deepface import DeepFace
from utils.prepare_lfw_dataset import save_lfw_dataset
from utils.train_model import train_face_recognizer
from utils.face_utils import get_face_embeddings
import tempfile

st.set_page_config(page_title="Face Recognition System", page_icon=":smiley:")
st.title("Face Recognition System")
st.write("Upload an image or train the model using the LFW dataset")

MODEL_PATH = "models/face_recognizer.pkl"
DATASET_PATH = "dataset"

# Sidebar Step 1: Prepare Dataset
st.sidebar.header("Step 1: Prepare Dataset")
uploaded_zip = st.sidebar.file_uploader("Upload LFW Dataset ZIP", type="zip")

if uploaded_zip is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".zip") as tmp_zip:
        tmp_zip.write(uploaded_zip.getvalue())
        tmp_zip_path = tmp_zip.name
        st.sidebar.write(f"Temporary ZIP path: {tmp_zip_path}")  # Debugging line

    # In app.py, update the call to save_lfw_dataset
    with st.spinner("Processing uploaded dataset..."):
        try:
            # Pass the correct argument name 'zip_file'
            save_lfw_dataset(zip_path=tmp_zip_path, output_dir=DATASET_PATH)
            st.sidebar.success("Dataset uploaded and processed.")
        except Exception as e:
            st.sidebar.error(f"Error during dataset processing: {e}")
            st.error(f"Detailed Error: {e}")
else:
    st.sidebar.write("No dataset uploaded yet.")

# Sidebar Step 2: Train Model
st.sidebar.header("Step 2: Train Model")
model_ready = os.path.exists(MODEL_PATH)

if not model_ready:
    if st.sidebar.button("Train Model", disabled=model_ready):
        progress_bar = st.progress(0)
        status_text = st.empty()
        with st.spinner("Training model..."):
            try:
                def update_progress(progress):
                    progress_bar.progress(progress)
                    status_text.text(f"Training Progress: {int(progress * 100)}%")

                train_face_recognizer(
                    dataset_path=os.path.join(DATASET_PATH, "processed"),
                    model_path=MODEL_PATH,
                    progress_callback=update_progress
                )
                progress_bar.empty()
                status_text.text("Training complete.")
                st.sidebar.success("Model trained and saved.")
                model_ready = True
            except Exception as e:
                st.sidebar.error(f"Error during model training: {e}")
else:
    st.sidebar.success("Model already trained.")
    if st.sidebar.button("Retrain Model", disabled=not model_ready):
        progress_bar = st.progress(0)
        status_text = st.empty()
        with st.spinner("Retraining model..."):
            try:
                def update_progress(progress):
                    progress_bar.progress(progress)
                    status_text.text(f"Training Progress: {int(progress * 100)}%")

                train_face_recognizer(
                    dataset_path=os.path.join(DATASET_PATH, "processed"),
                    model_path=MODEL_PATH,
                    progress_callback=update_progress
                )
                progress_bar.empty()
                status_text.text("Retraining complete.")
                st.sidebar.success("Model retrained and saved.")
                model_ready = True
            except Exception as e:
                st.sidebar.error(f"Error during retraining: {e}")

# Load model function
@st.cache_resource
def load_model():
    try:
        model_data = joblib.load(MODEL_PATH)
        return model_data['model'], model_data['label_encoder'], model_data['pca']
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None, None

# Load model if available
if model_ready:
    model_result = load_model()
    if model_result == (None, None, None):
        st.error("Failed to load model. Please ensure the model is trained and the file is not corrupted.")
    else:
        classifier, label_encoder, pca = model_result

    # Sidebar Step 3: Face Recognition
    st.sidebar.header("Step 3: Face Recognition")
    option = st.sidebar.radio("Select input type:", ("Upload Image",))

    def recognize_faces(image):
        image_cv = np.array(image)
        image_cv = cv2.cvtColor(image_cv, cv2.COLOR_RGB2BGR)

        try:
            embeddings = []
            boxes = []

            aligned_face = DeepFace.detectFace(image_cv, detector_backend='opencv', enforce_detection=False)

            if aligned_face is not None:
                embedding = get_face_embeddings(aligned_face)
                embeddings.append(embedding)
                boxes.append((0, 0, aligned_face.shape[1], aligned_face.shape[0]))
            else:
                st.warning("No faces detected in the image!")
                return image_cv

            if not embeddings:
                st.warning("No face embeddings were extracted.")
                return image_cv

            embeddings_pca = pca.transform(embeddings)
            predictions = classifier.predict(embeddings_pca)
            probs = classifier.predict_proba(embeddings_pca)
            top_probs = np.max(probs, axis=1)
            names = label_encoder.inverse_transform(predictions)

            for (x, y, w, h), name, prob in zip(boxes, names, top_probs):
                label = f"{name} ({prob:.2f})"
                cv2.rectangle(image_cv, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(image_cv, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            return image_cv

        except Exception as e:
            st.error(f"Error during face recognition: {e}")
            return image_cv

    if option == "Upload Image":
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_container_width=True)

            if st.button("Recognize Faces"):
                with st.spinner("Recognizing faces..."):
                    result_image = recognize_faces(image)
                    result_image = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)
                    st.image(result_image, caption="Recognition Result", use_container_width=True)
else:
    st.warning("Please upload the dataset and train the model first.")
