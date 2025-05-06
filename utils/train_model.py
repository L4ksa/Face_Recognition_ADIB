import os
import pandas as pd
import numpy as np
from deepface import DeepFace
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import joblib
import cv2
from collections import Counter

# Function to prepare data from the dataset (use all images)
def prepare_data(dataset_path):
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
                # Get the embedding for the current image using DeepFace
                embedding = DeepFace.represent(
                    img_bgr, 
                    model_name="VGG-Face", 
                    enforce_detection=False
                )[0]["embedding"]
                embeddings.append(embedding)
                labels.append(person_name)
            except Exception:
                # Skip images that raise errors (no logging here to avoid lag)
                continue

    return embeddings, labels

# Function to train the face recognizer using all images
def train_face_recognizer(dataset_path, model_path):
    # Ensure the directory exists
    model_dir = os.path.dirname(model_path)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)  # Create the directory if it doesn't exist

    # Load all images from the dataset and create embeddings
    X, y = prepare_data(dataset_path)

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

    # Train the SVM classifier
    print("\nTraining SVM classifier...")
    clf = SVC(kernel="linear", probability=True)
    clf.fit(X, y_enc)

    # Evaluate the model on the entire dataset
    print("\nEvaluating model on the entire dataset...")
    preds = clf.predict(X)
    acc = accuracy_score(y_enc, preds)
    print(f"Accuracy: {acc:.2f}")

    # Save the trained model
    print("\nSaving model...")
    joblib.dump({"model": clf, "label_encoder": le}, model_path)
    print(f"Model saved to {model_path}")
