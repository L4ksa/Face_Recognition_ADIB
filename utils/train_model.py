import os
import time
import numpy as np
import cv2
import joblib
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from tqdm import tqdm
import time
from utils.face_utils import get_face_embeddings
import streamlit as st

def load_dataset(dataset_path):
    """Load valid image paths and labels from the dataset directory."""
    image_paths = []
    labels = []

    if not os.path.exists(dataset_path):
        raise ValueError(f"Dataset path {dataset_path} does not exist.")

    for person_name in os.listdir(dataset_path):
        person_path = os.path.join(dataset_path, person_name)
        if not os.path.isdir(person_path):
            continue

        for image_name in os.listdir(person_path):
            if image_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_path = os.path.join(person_path, image_name)
                image = cv2.imread(image_path)

                if image is not None:
                    image_paths.append(image_path)
                    labels.append(person_name)

    return image_paths, labels

def train_face_recognizer(dataset_path, model_path, progress_callback=None):
    """
    Train a face recognition model using ArcFace embeddings and SVM.

    :param dataset_path: Path to processed face images.
    :param model_path: Path to save trained model.
    :param progress_callback: Optional progress bar hook.
    """
    print("📂 Loading dataset...")
    image_paths, labels = load_dataset(dataset_path)

    if not image_paths:
        raise ValueError("No valid images found in dataset.")

    X, y = [], []
    total_images = len(image_paths)
    start_time = time.time()  # Track start time for time estimation



    for i, (img_path, label) in enumerate(tqdm(zip(image_paths, labels), total=total_images, desc="🔍 Extracting embeddings")):


        image = cv2.imread(img_path)
        if image is None:
            print(f"⚠️ Skipped unreadable image: {img_path}")
            continue

        embedding = get_face_embeddings(image)
        if embedding is not None:
            X.append(embedding)
            y.append(label)
        else:
            print(f"⚠️ No embedding from image: {img_path}")

    if not X:
        raise ValueError("No embeddings extracted. Training aborted.")

    X = np.array(X)
    y = np.array(y)

    # Encode string labels to integers
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # Reduce dimensionality (ArcFace outputs 512-dim embeddings)
    if X.shape[0] > 100:  # Enough samples for PCA
        pca = PCA(n_components=100)
        X_transformed = pca.fit_transform(X)
        print("🧬 PCA applied (100 components).")
    else:
        pca = None
        X_transformed = X
        print("ℹ️ PCA skipped (too few samples).")

    # Train SVM classifier
    clf = SVC(kernel='linear', probability=True, class_weight='balanced')
    clf.fit(X_transformed, y_encoded)
    print("🤖 Model training completed.")

    # Save model, PCA, and label encoder
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump({
        'model': clf,
        'pca': pca,
        'label_encoder': label_encoder
    }, model_path)

    print(f"✅ Model saved to: {model_path}")
