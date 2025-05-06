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

st.set_page_config(page_title="Face Recognition System", page_icon=":smiley:")
st.title("Face Recognition System")
st.write("Upload an image or train the model using the LFW dataset")

MODEL_PATH = "models/face_recognizer.pkl"
DATASET_PATH = "dataset"

# Sidebar Step 1: Prepare Dataset
st.sidebar.header("Step 1: Prepare Dataset")
if not os.path.exists(DATASET_PATH):
    if st.sidebar.button("Download LFW Dataset"):
        with st.spinner("Downloading and preparing dataset..."):
            save_lfw_dataset(output_dir=DATASET_PATH)
        st.sidebar.success("LFW dataset ready!")
else:
    st.sidebar.success("LFW dataset already available.")

# Sidebar Step 2: Train Model
st.sidebar.header("Step 2: Train Model")
model_ready = os.path.exists(MODEL_PATH)

if not model_ready:
    if st.sidebar.button("Train Model", disabled=model_ready):
        with st.spinner("Training model..."):
            train_face_recognizer(
                dataset_path=DATASET_PATH,
                model_path=MODEL_PATH
            )
        st.sidebar.success("Model trained and saved.")
        model_ready = True
        st.cache_resource.clear()
else:
    st.sidebar.success("Model already trained.")
    if st.sidebar.button("Retrain Model", disabled=not model_ready):
        with st.spinner("Retraining model..."):
            train_face_recognizer(
                dataset_path=DATASET_PATH,
                model_path=MODEL_PATH,
            )
        st.sidebar.success("Model retrained and saved.")
        st.cache_resource.clear()
        model_ready = True

# Load model function
@st.cache_resource
def load_model():
    model_data = joblib.load(MODEL_PATH)
    return model_data['model'], model_data['label_encoder']

# Load model if available
if model_ready:
    classifier, label_encoder = load_model()

    # Sidebar Step 3: Face Recognition
    st.sidebar.header("Step 3: Face Recognition")
    option = st.sidebar.radio("Select input type:", ("Upload Image",))

    def recognize_faces(image):
        image_cv = np.array(image)
        image_cv = cv2.cvtColor(image_cv, cv2.COLOR_RGB2BGR)

        gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        detected = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        if len(detected) == 0:
            st.warning("No faces detected in the image!")
            return image_cv

        embeddings = []
        boxes = []

        for (x, y, w, h) in detected:
            face_img = image_cv[y:y+h, x:x+w]
            try:
                embedding = DeepFace.represent(
                    face_img,
                    model_name="VGG-Face",
                    enforce_detection=False
                )[0]["embedding"]
                embeddings.append(embedding)
                boxes.append((x, y, w, h))
            except Exception:
                continue  # Skip face if embedding fails

        if not embeddings:
            st.warning("No face embeddings were extracted.")
            return image_cv

        predictions = classifier.predict(embeddings)
        probs = classifier.predict_proba(embeddings)
        top_probs = np.max(probs, axis=1)
        names = label_encoder.inverse_transform(predictions)

        for (x, y, w, h), name, prob in zip(boxes, names, top_probs):
            label = f"{name} ({prob:.2f})"
            cv2.rectangle(image_cv, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(image_cv, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

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
    st.warning("Please prepare the dataset and train the model first.")
