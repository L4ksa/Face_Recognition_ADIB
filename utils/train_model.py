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
    """Load images and labels from dataset directory."""
    images = []
    labels = []
    
    for person_name in os.listdir(dataset_path):
        person_path = os.path.join(dataset_path, person_name)
        
        if os.path.isdir(person_path):
            for image_name in os.listdir(person_path):
                image_path = os.path.join(person_path, image_name)
                
                try:
                    image = cv2.imread(image_path)
                    if image is not None:
                        images.append(image)
                        labels.append(person_name)
                except Exception as e:
                    print(f"Error loading {image_path}: {e}")
    
    return images, labels


def train_face_recognizer(dataset_path, model_path, progress_callback=None):
    """
    Trains the face recognition model with a given dataset and saves the model to the specified path.
    
    :param dataset_path: Path to the directory containing processed images.
    :param model_path: Path to save the trained model.
    :param progress_callback: Optional callback to show training progress.
    """
    
    X = []
    y = []
    
    # Load dataset and check if it's valid
    images, labels = load_dataset(dataset_path)
    if not images:
        raise ValueError("No valid images found in dataset.")
    
    print(f"Loaded {len(images)} images from dataset.")
    
    total_images = len(images)
    current = 0

    # Extract embeddings for each image
    for img, label in tqdm(zip(images, labels), total=total_images, desc="Extracting embeddings"):
        try:
            embedding = get_face_embeddings(img)
            if embedding is not None:
                X.append(embedding)
                y.append(label)
            else:
                print(f"No embedding found for image: {label}")
        except Exception as e:
            print(f"Error processing image: {e}")
            continue
        
        current += 1
        if progress_callback:
            progress_callback(current / total_images)
    
    if not X:
        raise ValueError("No valid embeddings found for training.")
    
    # Convert to numpy arrays for model training
    X = np.array(X)
    y = np.array(y)

    # Encode labels to numeric values
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    # Apply PCA to reduce dimensionality
    pca = PCA(n_components=100)
    X_pca = pca.fit_transform(X)

    # Train the model using SVM
    clf = SVC(kernel='linear', probability=True)
    clf.fit(X_pca, y_encoded)

    # Save model components (SVM, label encoder, PCA)
    model_data = {
        'model': clf,
        'label_encoder': label_encoder,
        'pca': pca
    }

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model_data, model_path)
    print(f"Model trained and saved to: {model_path}")
