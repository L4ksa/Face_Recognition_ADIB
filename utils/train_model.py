import os
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from skimage.feature import hog
from sklearn.model_selection import train_test_split
import joblib
import pandas as pd
from utils.prepare_lfw_dataset import load_dataset  # Assuming prepare_lfw_dataset.py is in the same directory
from PIL import Image

def extract_hog_features(images):
    """
    Extracts HOG features from a list of images.
    """
    features = []
    for img in images:
        # Convert the image to grayscale (if it's colored)
        if img.ndim == 3:
            img = np.mean(img, axis=-1)
        
        # Extract HOG features
        fd, _ = hog(img, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True)
        features.append(fd)
    return np.array(features)

def load_csv(csv_path):
    """
    Load the CSV file that contains image paths and labels.
    """
    print(f"Loading CSV file: {csv_path}")
    df = pd.read_csv(csv_path)
    return df

def print_dataset_structure(dataset_path):
    print(f"\n📂 Current structure of '{dataset_path}':")
    for root, dirs, files in os.walk(dataset_path):
        level = root.replace(dataset_path, '').count(os.sep)
        indent = ' ' * 4 * level
        print(f"{indent}{os.path.basename(root)}/")
        subindent = ' ' * 4 * (level + 1)
        for f in files:
            print(f"{subindent}{f}")
    print()  # Blank line for readability

def prepare_data(dataset_path="dataset", train_csv="utils/peopleDevTrain.csv", test_csv="utils/peopleDevTest.csv"):
    """
    Loads the dataset, using CSVs for splitting and labeling, and processes it.
    """
    # Load CSV files for training and testing data
    train_df = load_csv(train_csv)
    test_df = load_csv(test_csv)

    images = []
    labels = []
    target_names = []

    # Process training data
    for _, row in train_df.iterrows():
        img_path = os.path.join(dataset_path, row['name'], f"{row['name']}_{row['images']}.jpg")
        img = Image.open(img_path)
        img = np.array(img)  # Convert the image to a numpy array
        images.append(img)
        labels.append(row['name'])
        if row['name'] not in target_names:
            target_names.append(row['name'])

    # Process testing data
    test_images = []
    test_labels = []
    for _, row in test_df.iterrows():
        img_path = os.path.join(dataset_path, row['name'], f"{row['name']}_{row['images']}.jpg")
        img = Image.open(img_path)
        img = np.array(img)  # Convert the image to a numpy array
        test_images.append(img)
        test_labels.append(row['name'])

    return (images, labels, target_names), (test_images, test_labels)

def train_face_recognizer(dataset_path="dataset", model_path="model.pkl", train_csv="utils/peopleDevTrain.csv", test_csv="utils/peopleDevTest.csv"):
    """
    Loads dataset, extracts features, trains an SVM classifier, and saves the model.
    """
    print("Loading dataset...")
    # Show dataset folder structure
    print_dataset_structure(dataset_path)
    
    (images, labels, target_names), (test_images, test_labels) = prepare_data(dataset_path, train_csv, test_csv)

    if len(images) == 0:
        print("No images found in the dataset.")
        return
    
    print(f"Loaded {len(images)} images from {len(target_names)} people for training.")
    print(f"Loaded {len(test_images)} images from {len(set(test_labels))} people for testing.")

    # Extract features from the images
    print("Extracting HOG features from images...")
    X_train = extract_hog_features(images)
    X_test = extract_hog_features(test_images)
    
    # Encode the labels
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(labels)
    y_test = label_encoder.transform(test_labels)  # Apply the same encoding for test labels
    
    # Train the SVM classifier
    print("Training SVM classifier...")
    classifier = SVC(kernel="linear", probability=True)
    classifier.fit(X_train, y_train)

    # Save the model
    print(f"Saving model to {model_path}...")
    joblib.dump({
        'classifier': classifier,
        'label_encoder': label_encoder,
        'target_names': target_names
    }, model_path)

    # Evaluate the model
    print("Evaluating model...")
    y_pred = classifier.predict(X_test)
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

if __name__ == "__main__":
    train_face_recognizer()
