import streamlit as st
import cv2
import numpy as np
import os
import pickle
from PIL import Image
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from utils.face_utils import detect_faces, align_face, get_face_embeddings
from pathlib import Path
from tqdm import tqdm

# Constants
DATASET_DIR = "known_faces"
MODEL_PATH = "models/face_recognizer.pkl"

st.set_page_config(page_title="Face Recognition with Auto-Training", page_icon="📷")
st.title("Face Recognition System")

# Function to train the model
def train_model_from_dataset():
    embeddings = []
    labels = []

    st.info("No model found. Training from dataset...")
    progress = st.progress(0)
    i = 0

    for person_name in os.listdir(DATASET_DIR):
        person_path = os.path.join(DATASET_DIR, person_name)
        if not os.path.isdir(person_path):
            continue

        for filename in os.listdir(person_path):
            file_path = os.path.join(person_path, filename)
            try:
                image = cv2.imread(file_path)
                if image is None:
                    continue
                faces = detect_faces(image)
                if len(faces) == 0:
                    continue
                aligned = align_face(image, faces[0])
                if aligned is None:
                    continue
                embedding = get_face_embeddings(aligned)
                embeddings.append(embedding)
                labels.append(person_name)
            except Exception as e:
                print(f"Failed processing {file_path}: {e}")

        i += 1
        progress.progress(i / len(os.listdir(DATASET_DIR)))

    if not embeddings:
        st.error("No valid data found in dataset. Cannot train model.")
        return None, None

    # Train classifier
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(labels)
    clf = SVC(probability=True)
    clf.fit(embeddings, y)

    # Save model
    os.makedirs("models", exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump({'classifier': clf, 'label_encoder': label_encoder}, f)

    st.success("✅ Model trained and saved successfully.")
    return clf, label_encoder

# Load or train model
@st.cache_resource
def load_or_train_model():
    if not os.path.exists(MODEL_PATH):
        return train_model_from_dataset()
    else:
        with open(MODEL_PATH, "rb") as f:
            model_data = pickle.load(f)
        return model_data['classifier'], model_data['label_encoder']

classifier, label_encoder = load_or_train_model()

# Sidebar
st.sidebar.header("Options")
option = st.sidebar.radio("Select input type:", ("Upload Image", "Use Webcam"))

# Face recognition
def recognize_faces(image_pil):
    image_np = np.array(image_pil)
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    faces = detect_faces(image_bgr)

    if not faces:
        st.warning("No faces detected.")
        return image_bgr

    embeddings = []
    valid_faces = []
    for face in faces:
        aligned = align_face(image_bgr, face)
        if aligned is not None:
            embeddings.append(get_face_embeddings(aligned))
            valid_faces.append(face)

    if not embeddings:
        st.warning("No embeddings could be extracted.")
        return image_bgr

    predictions = classifier.predict(embeddings)
    names = label_encoder.inverse_transform(predictions)

    for face, name in zip(valid_faces, names):
        x, y, w, h = face.left(), face.top(), face.width(), face.height()
        cv2.rectangle(image_bgr, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(image_bgr, name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

    return image_bgr

# Main app input
if option == "Upload Image":
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)

        if st.button("Recognize Faces"):
            with st.spinner("Processing..."):
                result_image = recognize_faces(image)
                st.image(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB), caption="Recognition Result", use_column_width=True)
