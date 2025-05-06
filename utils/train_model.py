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

    # Loop through all folders in the dataset directory (each folder corresponds to a person)
    for person_name in os.listdir(dataset_path):
        person_dir = os.path.join(dataset_path, person_name)

        # Skip if the current directory is not a folder
        if not os.path.isdir(person_dir):
            continue

        # Loop through all images for this person
        for img_name in os.listdir(person_dir):
            img_path = os.path.join(person_dir, img_name)

            # Skip if it's not an image file (e.g., non-jpg/jpeg/png files)
            if not img_name.endswith(('.jpg', '.jpeg', '.png')):
                continue

            try:
                img_bgr = cv2.imread(img_path)
                # Get the face embeddings from the face_utils function
                embedding = get_face_embeddings(img_bgr, model_name=model_name)
                if embedding is not None:
                    embeddings.append(embedding)
                    labels.append(person_name)
            except Exception:
                # Skip images that raise errors
                continue

    return embeddings, labels

# Function to train the face recognizer using all images with PCA and SVM
def train_face_recognizer(dataset_path, model_path, model_name="ArcFace"):
    # Ensure the directory exists
    model_dir = os.path.dirname(model_path)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)  # Create the directory if it doesn't exist

    # Load all images from the dataset and create embeddings
    X, y = prepare_data(dataset_path, model_name=model_name)

    if len(X) == 0:
        raise ValueError("No data found. Check dataset folder.")

    # Log the label distribution
    print("\nLabel distribution:")
    print(Counter(y))

    # Optional: Warn if one class dominates
    label_counts = Counter(y)
    most_common = label_counts.most_common(1)[0]
    if most_common[1] > len(y) * 0.5:
        print(f"⚠️ Warning: '{most_common[0]}' dominates dataset with {most_common[1]} samples.")

    # Encode labels
    print("\nEncoding labels...")
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    # Dimensionality Reduction (PCA)
    print("\nReducing dimensions with PCA...")
    pca = PCA(n_components=100)  # Reducing to 100 components
    X_pca = pca.fit_transform(X)

    # Split data into training and validation set (cross-validation)
    X_train, X_val, y_train, y_val = train_test_split(X_pca, y_enc, test_size=0.2, random_state=42)

    # Train the SVM classifier
    print("\nTraining SVM classifier...")
    clf = SVC(kernel="rbf", probability=True, C=1.0, gamma='scale')
    clf.fit(X_train, y_train)

    # Evaluate the model on the validation set
    print("\nEvaluating model on validation set...")
    preds = clf.predict(X_val)
    acc = accuracy_score(y_val, preds)
    print(f"Validation Accuracy: {acc:.2f}")

    # Save the trained model
    print("\nSaving model...")
    joblib.dump({"model": clf, "label_encoder": le, "pca": pca}, model_path)
    print(f"Model saved to {model_path}")
