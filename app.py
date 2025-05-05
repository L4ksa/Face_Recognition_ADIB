# 1. Model Development

## Step 1: Import Required Libraries
import os
import numpy as np
import pandas as pd
from deepface import DeepFace
import cv2
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import tempfile

# Define path to known faces
KNOWN_FACES_DIR = "./known_faces"

## Step 2: Load Dataset (Assume LFW is organized in folders by person name)
def load_known_faces(base_dir):
    embeddings = []
    labels = []
    for person in os.listdir(base_dir):
        person_dir = os.path.join(base_dir, person)
        if not os.path.isdir(person_dir):
            continue
        for image in os.listdir(person_dir):
            img_path = os.path.join(person_dir, image)
            try:
                result = DeepFace.represent(img_path=img_path, model_name="Facenet", enforce_detection=False)
                if result and isinstance(result, list) and "embedding" in result[0]:
                    embedding = result[0]["embedding"]
                    embeddings.append(embedding)
                    labels.append(person)
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
    return np.array(embeddings), np.array(labels)

# Load embeddings and labels
known_embeddings, known_labels = load_known_faces(KNOWN_FACES_DIR)

# Train KNN Classifier
X_train, X_test, y_train, y_test = train_test_split(known_embeddings, known_labels, test_size=0.2, random_state=42)
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# 2. GUI Development (Streamlit)

import streamlit as st
from PIL import Image

st.title("Face Recognition System")
st.write("Upload an image and the system will recognize known faces.")

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption='Uploaded Image', use_column_width=True)

    # Save uploaded image to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
        img.save(temp_file.name)
        temp_img_path = temp_file.name

    try:
        result = DeepFace.represent(img_path=temp_img_path, model_name="Facenet", enforce_detection=False)
        if result and isinstance(result, list) and "embedding" in result[0]:
            embedding = np.array(result[0]["embedding"]).reshape(1, -1)

            # Predict using KNN
            distances, indices = knn.kneighbors(embedding)
            threshold = 0.7

            st.subheader("Recognition Result")
            if distances[0][0] < threshold:
                predicted_identity = knn.classes_[indices[0][0]]
                st.write(f"Predicted identity: {predicted_identity} (distance: {distances[0][0]:.3f})")
            else:
                st.write("No match found (distance too high).")
        else:
            st.write("No face detected or embedding could not be extracted.")

    except Exception as e:
        st.error(f"Error: {e}")

# 3. Model Evaluation - for offline evaluation purposes

def evaluate_model(test_dir):
    y_true, y_pred = [], []
    for person in os.listdir(test_dir):
        person_dir = os.path.join(test_dir, person)
        for image in os.listdir(person_dir):
            img_path = os.path.join(person_dir, image)
            try:
                result = DeepFace.represent(img_path=img_path, model_name="Facenet", enforce_detection=False)
                if result and isinstance(result, list) and "embedding" in result[0]:
                    embedding = np.array(result[0]["embedding"]).reshape(1, -1)
                    distances, indices = knn.kneighbors(embedding)
                    if distances[0][0] < 0.7:
                        prediction = knn.classes_[indices[0][0]]
                    else:
                        prediction = "unknown"
                    y_true.append(person)
                    y_pred.append(prediction)
            except:
                continue
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average='weighted', zero_division=0),
        "recall": recall_score(y_true, y_pred, average='weighted', zero_division=0),
        "f1_score": f1_score(y_true, y_pred, average='weighted', zero_division=0)
    }
