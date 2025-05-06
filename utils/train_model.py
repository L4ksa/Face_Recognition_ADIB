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

# Function to prepare data from the dataset
def prepare_data(dataset_path, train_csv, test_csv):
    # Load CSV files
    train_df = pd.read_csv(train_csv)
    test_df = pd.read_csv(test_csv)

    def load_embeddings(df):
        embeddings, labels = [], []

        # Iterate through CSV and process images
        for _, row in df.iterrows():
            name = row["name"]
            image_count = int(row["images"])  
            person_dir = os.path.join(dataset_path, name)

            # Skip if person's folder does not exist
            if not os.path.exists(person_dir):
                continue

            for i in range(image_count):
                filename = f"{name}_{i}.jpg"
                img_path = os.path.join(person_dir, filename)

                # Skip if image does not exist
                if not os.path.exists(img_path):
                    continue

                try:
                    img_bgr = cv2.imread(img_path)
                    embedding = DeepFace.represent(
                        img_bgr, 
                        model_name="VGG-Face", 
                        enforce_detection=False
                    )[0]["embedding"]
                    embeddings.append(embedding)
                    labels.append(name)
                except Exception:
                    # Skip images that raise errors (no logging here to avoid lag)
                    continue

        return embeddings, labels

    # Load training and testing data
    X_train, y_train = load_embeddings(train_df)
    X_test, y_test = load_embeddings(test_df)

    return (X_train, y_train), (X_test, y_test)

# Function to train the face recognizer
def train_face_recognizer(dataset_path, model_path, train_csv, test_csv):
    # Ensure the directory exists
    model_dir = os.path.dirname(model_path)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)  # Create the directory if it doesn't exist

    (X_train, y_train), (X_test, y_test) = prepare_data(dataset_path, train_csv, test_csv)

    if len(X_train) == 0:
        raise ValueError("No training data found. Check CSV paths and dataset folder.")

    # Log the training label distribution
    print("\nTraining label distribution:")
    print(Counter(y_train))

    # Optional: Warn if one class dominates
    label_counts = Counter(y_train)
    most_common = label_counts.most_common(1)[0]
    if most_common[1] > len(y_train) * 0.5:
        print(f"⚠️ Warning: '{most_common[0]}' dominates training set with {most_common[1]} samples.")

    # Encode labels
    print("\nEncoding labels...")
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_test_enc = le.transform(y_test) if y_test else []

    # Train the SVM classifier
    print("\nTraining SVM classifier...")
    clf = SVC(kernel="linear", probability=True)
    clf.fit(X_train, y_train_enc)

    # Evaluate the model on the test set
    if X_test and y_test_enc:
        print("\nEvaluating model on test set...")
        preds = clf.predict(X_test)
        acc = accuracy_score(y_test_enc, preds)
        print(f"Test Accuracy: {acc:.2f}")
    else:
        print("\nNo valid test set available. Evaluating on training set (not recommended for real validation).")
        preds = clf.predict(X_train)
        acc = accuracy_score(y_train_enc, preds)
        print(f"Training Accuracy: {acc:.2f}")

    # Save the trained model
    print("\nSaving model...")
    joblib.dump({"model": clf, "label_encoder": le}, model_path)
    print(f"Model saved to {model_path}")
