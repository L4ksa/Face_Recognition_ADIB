import os
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.svm import SVC
import joblib
from utils.face_utils import get_face_embeddings
import cv2


def train_face_recognizer(dataset_path="dataset", model_path="models/face_recognizer.pkl"):
    embeddings = []
    labels = []

    for person_name in os.listdir(dataset_path):
        person_dir = os.path.join(dataset_path, person_name)
        if not os.path.isdir(person_dir):
            continue

        for img_name in os.listdir(person_dir):
            img_path = os.path.join(person_dir, img_name)
            img = cv2.imread(img_path)
            if img is None:
                continue

            try:
                embedding = get_face_embeddings(img)
                embeddings.append(embedding)
                labels.append(person_name)
            except Exception as e:
                print(f"Skipping {img_path}: {e}")

    if not embeddings:
        raise ValueError("No valid face embeddings found. Training aborted.")

    embeddings = np.array(embeddings)
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(labels)

    pca = PCA(n_components=100, whiten=True, random_state=42)
    embeddings_pca = pca.fit_transform(embeddings)

    clf = SVC(kernel="linear", probability=True)
    clf.fit(embeddings_pca, y)

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump({
        'model': clf,
        'label_encoder': label_encoder,
        'pca': pca
    }, model_path)
