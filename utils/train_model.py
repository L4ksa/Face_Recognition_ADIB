import os
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import joblib

def prepare_data(dataset_path, train_csv, test_csv):
    print("\nChecking dataset folder structure...")
    for person in os.listdir(dataset_path):
        person_dir = os.path.join(dataset_path, person)
        if os.path.isdir(person_dir):
            print(f"{person}/")
            for img in os.listdir(person_dir):
                print(f"    {img}")

    print("\nLoading CSV files:")
    print(f"Train CSV: {train_csv}")
    print(f"Test CSV: {test_csv}")

    # Load CSVs
    train_df = pd.read_csv(train_csv)
    test_df = pd.read_csv(test_csv)

    def load_images_from_df(df):
        images = []
        labels = []
        for _, row in df.iterrows():
            name = row["name"]
            image_count = row["images"]
            person_dir = os.path.join(dataset_path, name)

            if not os.path.exists(person_dir):
                print(f"Warning: {person_dir} not found!")
                continue

            for i in range(int(image_count)):
                img_filename = f"{name}_{i}.jpg"
                img_path = os.path.join(person_dir, img_filename)

                if not os.path.exists(img_path):
                    print(f"Warning: Image not found: {img_path}")
                    continue

                img = Image.open(img_path).convert("L").resize((100, 100))  # Grayscale, resize
                images.append(np.array(img).flatten())  # Flatten image to 1D
                labels.append(name)

        return images, labels

    X_train, y_train = load_images_from_df(train_df)
    X_test, y_test = load_images_from_df(test_df)

    print(f"\nLoaded {len(X_train)} training images and {len(X_test)} testing images.")

    return (X_train, y_train), (X_test, y_test)

def train_face_recognizer(dataset_path, model_path, train_csv, test_csv):
    (X_train, y_train), (X_test, y_test) = prepare_data(dataset_path, train_csv, test_csv)

    if len(X_train) == 0:
        raise ValueError("No training data found. Check CSV paths and dataset folder.")

    # Keep only test samples that exist in training labels
    valid_labels = set(y_train)
    filtered_test = [(img, label) for img, label in zip(X_test, y_test) if label in valid_labels]
    
    # Unzip filtered test data
    if filtered_test:
        X_test, y_test = zip(*filtered_test)
        X_test = list(X_test)
        y_test = list(y_test)
    else:
        X_test, y_test = [], []
        print("Warning: No valid test data after filtering!")

    print("\nEncoding labels...")
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_test_enc = le.transform(y_test)

    print("\nTraining SVM classifier...")
    clf = SVC(kernel="linear", probability=True)
    clf.fit(X_train, y_train_enc)

    print("\nEvaluating model...")
    preds = clf.predict(X_test)
    acc = accuracy_score(y_test_enc, preds)
    print(f"Test Accuracy: {acc:.2f}")

    print("\nSaving model...")
    joblib.dump({"model": clf, "label_encoder": le}, model_path)
    print(f"Model saved to {model_path}")
