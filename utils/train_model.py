import os
import numpy as np
from deepface import DeepFace
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import joblib
import cv2
from collections import Counter
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from utils.face_utils import get_face_embeddings  # Import from face_utils

# Function to prepare data from the dataset using face_utils
def prepare_data(dataset_path, model_name="ArcFace"):
    embeddings, labels = [], []

    for person_name in os.listdir(dataset_path):
        person_dir = os.path.join(dataset_path, person_name)

        # Skip if the current directory is not a folder
        if not os.path.isdir(person_dir):
            continue

        for img_name in os.listdir(person_dir):
            img_path = os.path.join(person_dir, img_name)

            # Skip non-image files
            if not img_name.endswith(('.jpg', '.jpeg', '.png')):
                continue

            try:
                img_bgr = cv2.imread(img_path)
                embedding = get_face_embeddings(img_bgr)
                if embedding:
                    embeddings.append(embedding)
                    labels.append(person_name)
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
                continue

    return embeddings, labels

# Function to train the face recognizer
def train_face_recognizer(dataset_path, model_path, model_name="ArcFace"):
    # Ensure the directory exists
    model_dir = os.path.dirname(model_path)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    X, y = prepare_data(dataset_path, model_name=model_name)

    if len(X) == 0:
        raise ValueError("No data found. Check dataset folder.")

    # Label encoding
    print("Encoding labels...")
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    # PCA reduction
    print("Reducing dimensions with PCA...")
    pca = PCA(n_components=100)
    X_pca = pca.fit_transform(X)

    # Train/test split
    X_train, X_val, y_train, y_val = train_test_split(X_pca, y_enc, test_size=0.2, random_state=42)

    # SVM classifier training
    print("Training SVM classifier...")
    clf = SVC(kernel="rbf", probability=True, C=1.0, gamma='scale')
    clf.fit(X_train, y_train)

    # Evaluation
    print("Evaluating model...")
    preds = clf.predict(X_val)
    acc = accuracy_score(y_val, preds)
    print(f"Validation Accuracy: {acc:.2f}")

    # Save the model
    print(f"Saving model to {model_path}...")
    joblib.dump({"model": clf, "label_encoder": le, "pca": pca}, model_path)
    print("Model saved.")
