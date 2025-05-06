import os
import pandas as pd
import numpy as np
from deepface import DeepFace
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import joblib
import cv2

def prepare_data(dataset_path, train_csv, test_csv):
    # Check dataset structure (optional, you can remove this block if no need to print directory structure)
    # for person in os.listdir(dataset_path):
    #     person_dir = os.path.join(dataset_path, person)
    #     if os.path.isdir(person_dir):
    #         for img in os.listdir(person_dir):
    #             pass  # No print statements here to avoid lag

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

def train_face_recognizer(dataset_path, model_path, train_csv, test_csv):
    # Ensure the directory exists
    model_dir = os.path.dirname(model_path)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)  # Create the directory if it doesn't exist

    (X_train, y_train), (X_test, y_test) = prepare_data(dataset_path, train_csv, test_csv)

    if len(X_train) == 0:
        raise ValueError("No training data found. Check CSV paths and dataset folder.")

    # Filter test set to contain only labels seen in training
    valid_labels = set(y_train)
    filtered_test = [(emb, label) for emb, label in zip(X_test, y_test) if label in valid_labels]

    if filtered_test:
        X_test, y_test = zip(*filtered_test)
        X_test = list(X_test)
        y_test = list(y_test)
    else:
        X_test, y_test = [], []
        print("Warning: No valid test data after filtering.")

    print("\nEncoding labels...")
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_test_enc = le.transform(y_test) if y_test else []

    print("\nTraining SVM classifier...")
    clf = SVC(kernel="linear", probability=True)
    clf.fit(X_train, y_train_enc)

    if X_test and y_test_enc:
        print("\nEvaluating model...")
        preds = clf.predict(X_test)
        acc = accuracy_score(y_test_enc, preds)
        print(f"Test Accuracy: {acc:.2f}")
    else:
        print("\nNo test evaluation performed due to empty test set.")

    print("\nSaving model...")
    joblib.dump({"model": clf, "label_encoder": le}, model_path)
    print(f"Model saved to {model_path}")
