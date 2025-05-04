import os
import cv2
import numpy as np
from deepface import DeepFace
from deepface.commons import functions
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
import pandas as pd

# Path to LFW dataset
lfw_path = "path_to_lfw_dataset/lfw/"

# Preprocessing function
def preprocess_lfw_data():
    images = []
    labels = []
    for folder_name in os.listdir(lfw_path):
        folder_path = os.path.join(lfw_path, folder_name)
        if os.path.isdir(folder_path):
            for image_name in os.listdir(folder_path):
                image_path = os.path.join(folder_path, image_name)
                img = functions.preprocess_face(image_path, enforce_detection=False)
                images.append(img)
                labels.append(folder_name)
    
    return np.array(images), np.array(labels)

# Dataset loading and preprocessing
images, labels = preprocess_lfw_data()

# Splitting dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# Optionally: Train a model (Here we use a pretrained DeepFace model)
model = DeepFace.build_model("Facenet")

# Evaluate the model using the test set
def evaluate_model(model, X_test, y_test):
    predictions = []
    for img in X_test:
        result = DeepFace.find(img_path=img, db_path=lfw_path, model_name='Facenet')
        predicted_label = result['identity'][0]
        predictions.append(predicted_label)
    
    # Metrics Calculation
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, average='weighted')
    recall = recall_score(y_test, predictions, average='weighted')
    f1 = f1_score(y_test, predictions, average='weighted')
    
    print("Evaluation Metrics:")
    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")

# Run evaluation on the test set
evaluate_model(model, X_test, y_test)
