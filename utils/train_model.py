import os
import time
import numpy as np
import cv2
import joblib
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from collections import Counter
from utils.face_utils import get_face_embeddings

def load_dataset(dataset_path):
    """Load valid image paths and labels from the dataset directory."""
    image_paths, labels = [], []

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

def train_face_recognizer(dataset_path, model_path):
    """
    Train a face recognition model using ArcFace embeddings and SVM.
    Returns classification metrics as a dictionary.
    """
    print("📂 Loading dataset...")
    image_paths, labels = load_dataset(dataset_path)

    if not image_paths:
        raise ValueError("No valid images found in dataset.")

    X, y = [], []
    for img_path, label in zip(image_paths, labels):
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

    # Filter out classes with fewer than 2 samples
    label_counts = Counter(y)
    X_filtered = []
    y_filtered = []

    for x_val, label in zip(X, y):
        if label_counts[label] >= 2:
            X_filtered.append(x_val)
            y_filtered.append(label)

    if not X_filtered:
        raise ValueError("No class has at least 2 samples. Training cannot proceed.")

    X = np.array(X_filtered)
    y = np.array(y_filtered)

    # Encode string labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # Optional PCA
    if X.shape[0] > 100:
        pca = PCA(n_components=100)
        X_transformed = pca.fit_transform(X)
        print("🧬 PCA applied (100 components).")
    else:
        pca = None
        X_transformed = X
        print("ℹ️ PCA skipped (too few samples).")

    # Train/Test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_transformed, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    # Train SVM classifier
    clf = SVC(kernel='linear', probability=True, class_weight='balanced')
    clf.fit(X_train, y_train)
    print("✅ Model trained.")

    # Evaluate
    y_pred = clf.predict(X_test)
    report = classification_report(
        y_test, y_pred,
        target_names=label_encoder.classes_,
        output_dict=True
    )

    # Save model
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump({
        'model': clf,
        'pca': pca,
        'label_encoder': label_encoder
    }, model_path)

    print(f"💾 Model saved to {model_path}")
    return report
