import os
import numpy as np
import pandas as pd
from deepface import DeepFace
from deepface.commons import functions
import cv2
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Define path to known faces
KNOWN_FACES_DIR = "./known_faces"

## Step 2: Preprocessing Function
def preprocess_image(img_path):
    img = functions.preprocess_face(img_path, target_size=(224, 224), enforce_detection=False)
    return img

## Step 3: Load Dataset (Assume LFW is organized in folders by person name)
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
                embedding = DeepFace.represent(img_path=img_path, model_name="Facenet", enforce_detection=False)[0]["embedding"]
                embeddings.append(embedding)
                labels.append(person)
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
    return np.array(embeddings), np.array(labels)

# Load embeddings and labels
known_embeddings, known_labels = load_known_faces(KNOWN_FACES_DIR)

# 2. GUI Development (Streamlit)

import streamlit as st
from PIL import Image

st.title("Face Recognition System")
st.write("Upload an image and the system will recognize known faces.")

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    img_array = np.array(img)
    st.image(img_array, caption='Uploaded Image', use_column_width=True)

    try:
        # Detect faces
        detections = DeepFace.analyze(img_path=img_array, actions=["emotion"], enforce_detection=False)
        results = DeepFace.find(img_path=img_array, db_path=KNOWN_FACES_DIR, model_name="Facenet", enforce_detection=False)

        # Display bounding boxes and identities
        st.subheader("Recognition Results")
        for i, res in enumerate(results[0].iterrows()):
            index, data = res
            st.write(f"Match {i+1}: {data['identity']} with {round(data['distance'], 3)} distance")

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
                df = DeepFace.find(img_path=img_path, db_path=KNOWN_FACES_DIR, model_name="Facenet", enforce_detection=False)
                prediction = os.path.basename(df[0].iloc[0]['identity']) if not df[0].empty else "unknown"
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
