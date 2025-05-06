import os
import numpy as np
import cv2
import joblib
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from tqdm import tqdm
from utils.face_utils import get_face_embeddings

def load_dataset(dataset_path):
    """Load valid image paths and labels from the dataset directory."""
    image_paths = []
    labels = []

    for person_name in os.listdir(dataset_path):
        person_path = os.path.join(dataset_path, person_name)
        if not os.path.isdir(person_path):
            continue

        for image_name in os.listdir(person_path):
            image_path = os.path.join(person_path, image_name)
            if image_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_paths.append(image_path)
                labels.append(person_name)

    return image_paths, labels

def train_face_recognizer(dataset_path, model_path, progress_callback=None):
    """
    Trains an SVM face recognition model using embeddings extracted from images.
    
    :param dataset_path: Path to processed dataset (faces).
    :param model_path: Where to save the model (joblib file).
    :param progress_callback: Optional progress bar callback (for Streamlit).
    """
    print("Loading dataset...")
    image_paths, labels = load_dataset(dataset_path)
    
    if not image_paths:
        raise ValueError("No valid images found in dataset.")

    print(f"Found {len(image_paths)} images across {len(set(labels))} individuals.")

    X, y = [], []
    for i, (img_path, label) in enumerate(tqdm(zip(image_paths, labels), total=len(image_paths), desc="Extracting embeddings")):
        try:
            image = cv2.imread(img_path)
            if image is None:
                print(f"Warning: Failed to load image {img_path}")
                continue

            embedding = get_face_embeddings(image)
            if embedding is not None:
                X.append(embedding)
                y.append(label)
            else:
                print(f"Warning: No embedding extracted from {img_path}")
        except Exception as e:
            print(f"Error processing {img_path}: {e}")

        if progress_callback:
            progress_callback((i + 1) / len(image_paths))

    if not X:
        raise ValueError("No valid embeddings extracted from dataset.")

    # Convert to arrays
    X = np.array(X)
    y = np.array(y)

    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # Dimensionality reduction (optional)
    try:
        pca = PCA(n_components=100)
        X_pca = pca.fit_transform(X)
    except ValueError:
        print("PCA failed (possibly due to too few samples). Skipping PCA.")
        pca = None
        X_pca = X

    # Train classifier
    clf = SVC(kernel='linear', probability=True)
    clf.fit(X_pca, y_encoded)

    # Save model components
    model_data = {
        'model': clf,
        'label_encoder': label_encoder,
        'pca': pca
    }

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model_data, model_path)
    print(f"✅ Model saved to {model_path}")
