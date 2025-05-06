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
        with st.spinner("Training model..."):
            try:
                train_face_recognizer(
                    dataset_path=os.path.join(DATASET_PATH, "processed"),
                    model_path=MODEL_PATH,
                    progress_callback=lambda p: update_progress_bar(progress_bar, status_text, p)
                )
                progress_bar.empty()
                status_text.text("Training complete.")
                st.sidebar.success("Model trained and saved!")
                model_ready = True
            except Exception as e:
                st.sidebar.error("Model training failed.")
                st.error(f"Error: {e}")
else:
    st.sidebar.success("Model already trained.")
    if st.sidebar.button("Retrain Model"):
        progress_bar = st.progress(0)
        status_text = st.empty()
        with st.spinner("Retraining model..."):
            try:
                train_face_recognizer(
                    dataset_path=os.path.join(DATASET_PATH, "processed"),
                    model_path=MODEL_PATH,
                    progress_callback=lambda p: update_progress_bar(progress_bar, status_text, p)
                )
                progress_bar.empty()
                status_text.text("Retraining complete.")
                st.sidebar.success("Model retrained and saved!")
                model_ready = True
            except Exception as e:
                st.sidebar.error("Model retraining failed.")
                st.error(f"Error: {e}")

# ---------------------- Step 3: Face Recognition ----------------------

@st.cache_resource
def load_model():
    try:
        model_data = joblib.load(MODEL_PATH)
        return model_data['model'], model_data['label_encoder'], model_data['pca']
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None, None

if model_ready:
    st.sidebar.header("Step 3: Face Recognition")
    option = st.sidebar.radio("Input type", ("Upload Image",))

    classifier, label_encoder, pca = load_model()
    if classifier is None:
        st.warning("Model could not be loaded. Please retrain it.")
    else:
        def recognize_faces(image: Image.Image):
            image_cv = np.array(image)
            image_cv = cv2.cvtColor(image_cv, cv2.COLOR_RGB2BGR)

            try:
                aligned_face = DeepFace.detectFace(image_cv, detector_backend='opencv', enforce_detection=False)

                if aligned_face is None:
                    st.warning("No face detected!")
                    return image_cv

                embedding = get_face_embeddings(aligned_face)
                if embedding is None:
                    st.warning("Failed to extract embedding.")
                    return image_cv

                embedding_pca = pca.transform([embedding])
                prediction = classifier.predict(embedding_pca)
                probability = np.max(classifier.predict_proba(embedding_pca))
                name = label_encoder.inverse_transform(prediction)[0]

                label = f"{name} ({probability:.2f})"
                h, w = aligned_face.shape[:2]
                cv2.rectangle(image_cv, (0, 0), (w, h), (0, 255, 0), 2)
                cv2.putText(image_cv, label, (0, h + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                return image_cv

            except Exception as e:
                st.error(f"Recognition error: {e}")
                return image_cv

        if option == "Upload Image":
            uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
            if uploaded_file:
                image = Image.open(uploaded_file)
                st.image(image, caption="Uploaded Image", use_container_width=True)

                if st.button("Recognize Face"):
                    with st.spinner("Recognizing..."):
                        result_image = recognize_faces(image)
                        result_image = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)
                        st.image(result_image, caption="Recognition Result", use_container_width=True)
else:
    st.warning("Please upload and process the dataset, then train the model to continue.")
