import os
import cv2
import pickle
import numpy as np
import pandas as pd
from PIL import Image
from deepface import DeepFace
from utils.face_utils import (
    detect_faces, 
    get_face_embeddings
)
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder

# File paths (relative to project root, since script is in utils/)
dataset_path = "dataset"
base_dir = "dataset\lfw-deepfunneled"
CSV_TRAIN = os.path.join("utils", "peopleDevTrain.csv")
CSV_TEST = os.path.join("utils", "peopleDevTest.csv")

def load_people_split(csv_path, base_dir):
    images = []
    labels = []
    df = pd.read_csv(csv_path)

    for _, row in df.iterrows():
        person = row['name'].replace(" ", "_")
        image_path = os.path.join(base_dir, person, row['Image'])
        if os.path.exists(image_path):
            try:
                img = Image.open(image_path).convert("RGB")
                img = np.array(img)
                images.append(img)
                labels.append(person)
            except Exception as e:
                print(f"Error loading image {image_path}: {e}")

    return images, labels

def train_face_recognizer(dataset_path, model_path):
    if not os.path.exists(CSV_TRAIN) or not os.path.exists(CSV_TEST):
        raise FileNotFoundError("CSV training/testing files not found in utils/ directory")

    print("Loading training and testing sets from CSV...")
    train_images, train_labels = load_people_split(CSV_TRAIN, dataset_path)
    test_images, test_labels = load_people_split(CSV_TEST, dataset_path)

    print(f"Loaded {len(train_images)} training images and {len(test_images)} testing images.")

    X_train, y_train = [], []
    for img, label in zip(train_images, train_labels):
        try:
            embedding = DeepFace.represent(img, model_name="VGG-Face", enforce_detection=True)[0]['embedding']
            X_train.append(embedding)
            y_train.append(label)
        except Exception as e:
            print(f"Error processing train image: {e}")

    X_test, y_test = [], []
    for img, label in zip(test_images, test_labels):
        try:
            embedding = DeepFace.represent(img, model_name="VGG-Face", enforce_detection=True)[0]['embedding']
            X_test.append(embedding)
            y_test.append(label)
        except Exception as e:
            print(f"Error processing test image: {e}")

    print(f"Train shape: {len(X_train)}, Test shape: {len(X_test)}")

    # Encode labels
    label_encoder = LabelEncoder()
    y_train_enc = label_encoder.fit_transform(y_train)
    y_test_enc = label_encoder.transform(y_test)

    # Train classifier
    print("Training SVM classifier...")
    classifier = SVC(kernel="linear", probability=True)
    classifier.fit(X_train, y_train_enc)

    # Evaluate
    print("Evaluating model...")
    y_pred = classifier.predict(X_test)
    evaluate_model(y_test_enc, y_pred)

    # Save model
    model_data = {
        "classifier": classifier,
        "label_encoder": label_encoder
    }
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    with open(model_path, "wb") as f:
        pickle.dump(model_data, f)

    print("Model trained and saved successfully!")


if __name__ == "__main__":
    train_face_recognizer()
