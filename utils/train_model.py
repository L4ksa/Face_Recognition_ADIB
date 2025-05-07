import os
import time
import numpy as np
import cv2
import joblib
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from utils.face_utils import get_face_embeddings
import gc
import streamlit as st

BATCH_SIZE = 100  # The number of images per batch
FEATURES_PATH = "model/features_batch.npz"  # Path to store the extracted features
MODEL_PATH = "model/face_recognition_model.pkl"  # Path to save the trained model

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


def extract_features_in_batches(dataset_path, features_path, batch_size=BATCH_SIZE):
    """Extract face embeddings from images in batches and save to disk."""
    image_paths, labels = load_dataset(dataset_path)
    total_images = len(image_paths)
    all_embeddings = []
    all_labels = []

    print(f"🔍 Extracting features from {total_images} images...")

    for i in range(0, total_images, batch_size):
        batch_image_paths = image_paths[i:i + batch_size]
        batch_labels = labels[i:i + batch_size]

        embeddings = []
        for img_path, label in zip(batch_image_paths, batch_labels):
            try:
                image = cv2.imread(img_path)
                embedding = get_face_embeddings(image)
                if embedding is not None:
                    embeddings.append(embedding)
                    all_labels.append(label)
            except Exception as err:
                print(f"⚠️ Failed processing {img_path}: {err}")

        # Save batch embeddings and labels to disk after processing each batch
        if embeddings:
            all_embeddings.extend(embeddings)
            print(f"🧠 Extracted embeddings for batch {i // batch_size + 1}")

        np.savez(features_path, embeddings=np.array(all_embeddings), labels=np.array(all_labels))

    return all_embeddings, all_labels


def train_face_recognizer(dataset_path, model_path, features_path=FEATURES_PATH, progress_callback=None):
    try:
        # Step 1: Feature extraction in batches and save to disk
        print("📂 Extracting features in batches...")
        embeddings, labels = extract_features_in_batches(dataset_path, features_path)

        if not embeddings:
            raise ValueError("No embeddings extracted. Training aborted.")

        print(f"✅ Extracted {len(embeddings)} embeddings.")

        # Step 2: Prepare data for training
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(labels)
        X = np.array(embeddings)

        if len(np.unique(y_encoded)) < 2:
            raise ValueError(f"The number of classes has to be greater than one; got {len(np.unique(y_encoded))} class.")

        # Apply PCA if necessary
        pca = None
        if X.shape[0] > 100:
            pca = PCA(n_components=100)
            X = pca.fit_transform(X)
            print("🧬 PCA applied.")

        # Step 3: Train the SVM classifier
        clf = SVC(kernel='linear', probability=True, class_weight='balanced')
        clf.fit(X, y_encoded)
        print("🤖 Model training completed.")

        # Step 4: Save the trained model and PCA
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        joblib.dump({
            'model': clf,
            'pca': pca,
            'label_encoder': label_encoder
        }, model_path)
        print(f"✅ Model saved to: {model_path}")

        gc.collect()

    except Exception as e:
        error_message = f"❌ Training failed: {e}"
        print(error_message)
        if progress_callback:
            progress_callback(error_message)
