import os
import numpy as np
import cv2
import joblib
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from tqdm import tqdm
from utils.face_utils import get_face_embeddings


def train_face_recognizer(dataset_path, model_path, progress_callback=None):
    X = []
    y = []

    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset path '{dataset_path}' does not exist.")

    person_dirs = [
        d for d in os.listdir(dataset_path)
        if os.path.isdir(os.path.join(dataset_path, d))
    ]

    if not person_dirs:
        raise ValueError("No class folders found in dataset.")

    total_images = sum(
        len(files)
        for person in person_dirs
        for _, _, files in os.walk(os.path.join(dataset_path, person))
    )

    current = 0

    for person in tqdm(person_dirs, desc="Processing persons"):
        person_path = os.path.join(dataset_path, person)
        for img_file in os.listdir(person_path):
            img_path = os.path.join(person_path, img_file)
            img = cv2.imread(img_path)

            if img is None:
                print(f"[WARN] Failed to read image: {img_path}")
                continue

            try:
                embedding = get_face_embeddings(img)
                if embedding is not None:
                    X.append(embedding)
                    y.append(person)
                else:
                    print(f"[INFO] No embedding found for image: {img_path}")
            except Exception as e:
                print(f"[ERROR] Exception processing {img_path}: {e}")

            current += 1
            if progress_callback:
                progress_callback(current / total_images)

    if not X:
        raise ValueError("No valid images found for training.")

    X = np.array(X)
    y = np.array(y)

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    pca = PCA(n_components=100)
    X_pca = pca.fit_transform(X)

    clf = SVC(kernel='linear', probability=True)
    clf.fit(X_pca, y_encoded)

    model_data = {
        'model': clf,
        'label_encoder': label_encoder,
        'pca': pca
    }

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model_data, model_path)
    print(f"[INFO] Model saved to {model_path}")
