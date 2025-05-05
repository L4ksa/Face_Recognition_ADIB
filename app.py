import streamlit as st
import cv2
import numpy as np
import os
import shutil
import zipfile
from PIL import Image
import pickle

from utils.face_utils import detect_faces, align_face, get_face_embeddings
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder

MODEL_PATH = "models/face_recognizer.pkl"

st.set_page_config(page_title="Face Recognition System", page_icon="🧠")
st.title("Face Recognition System")
st.write("Train and recognize faces using your own dataset!")

# Caching model loading
@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as f:
        model_data = pickle.load(f)
    return model_data['classifier'], model_data['label_encoder']

# Helper: Train the face recognition model
def train_model(dataset_path):
    embeddings = []
    labels = []

    for person_name in os.listdir(dataset_path):
        person_path = os.path.join(dataset_path, person_name)
        if not os.path.isdir(person_path):
            continue
        for img_name in os.listdir(person_path):
            img_path = os.path.join(person_path, img_name)
            try:
                image = cv2.imread(img_path)
                faces = detect_faces(image)

                if len(faces) == 0:
                    continue

                for face in faces:
                    aligned = align_face(image, face)
                    embedding = get_face_embeddings(aligned)
                    embeddings.append(embedding)
                    labels.append(person_name)
            except Exception as e:
                st.warning(f"Failed to process {img_path}: {e}")

    if not embeddings:
        st.error("No valid face embeddings found in the dataset.")
        return None, None

    X = np.array(embeddings)
    y = np.array(labels)
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    clf = SVC(kernel='linear', probability=True)
    clf.fit(X, y_encoded)

    # Save model
    os.makedirs("models", exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump({'classifier': clf, 'label_encoder': label_encoder}, f)

    st.success("Model trained and saved successfully!")
    return clf, label_encoder

# ========== TRAINING SECTION ========== #
st.header("Step 1: Train Model with Dataset")
zip_file = st.file_uploader("Upload dataset as ZIP (folder/person/images)", type="zip")

if zip_file:
    with st.spinner("Extracting and training..."):
        dataset_dir = "dataset"
        if os.path.exists(dataset_dir):
            shutil.rmtree(dataset_dir)
        os.makedirs(dataset_dir, exist_ok=True)

        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall(dataset_dir)

        classifier, label_encoder = train_model(dataset_dir)

# ========== RECOGNITION SECTION ========== #
if os.path.exists(MODEL_PATH):
    st.header("Step 2: Recognize Faces in Image")
    uploaded_image = st.file_uploader("Upload an image to recognize faces", type=["jpg", "jpeg", "png"])

    classifier, label_encoder = load_model()

    def recognize_faces(image):
        img_cv = np.array(image)
        img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)

        faces = detect_faces(img_cv)
        if len(faces) == 0:
            st.warning("No faces detected.")
            return img_cv

        for face in faces:
            try:
                aligned = align_face(img_cv, face)
                embedding = get_face_embeddings(aligned).reshape(1, -1)
                pred = classifier.predict(embedding)
                name = label_encoder.inverse_transform(pred)[0]

                x, y, w, h = face.left(), face.top(), face.width(), face.height()
                cv2.rectangle(img_cv, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(img_cv, name, (x, y-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            except Exception as e:
                st.warning(f"Recognition error: {e}")

        return cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)

    if uploaded_image:
        image = Image.open(uploaded_image)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        if st.button("Recognize Faces"):
            with st.spinner("Processing..."):
                result = recognize_faces(image)
                st.image(result, caption="Result", use_column_width=True)
else:
    st.info("Please upload a training dataset first.")
