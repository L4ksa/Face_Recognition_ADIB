import os
import numpy as np
import pandas as pd
from deepface import DeepFace
import cv2
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
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
    st.image(img, caption='Uploaded Image', use_column_width=True)

    # Save uploaded image to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
        img.save(temp_file.name)
        temp_img_path = temp_file.name

    try:
        # Recognize faces using the image path
        results = DeepFace.find(img_path=temp_img_path, db_path=KNOWN_FACES_DIR, model_name="Facenet", enforce_detection=False)

        # Display results
        st.subheader("Recognition Results")
        if results and not results[0].empty:
            for i, res in results[0].iterrows():
                st.write(f"Match {i+1}: {res['identity']} with distance {round(res['distance'], 3)}")
        else:
            st.write("No known faces detected.")

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
