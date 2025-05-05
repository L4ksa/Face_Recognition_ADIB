import os
import pickle
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from deepface import DeepFace
from utils.prepare_lfw_dataset import load_dataset  # We need to implement this to load images from the saved LFW dataset

# Update the dataset path
DATASET_PATH = "dataset"  # This matches where save_lfw_dataset() saves the images

def load_dataset(dataset_path):
    """
    Loads images and their corresponding labels from the directory.
    Each subfolder corresponds to a person and contains their images.
    """
    images = []
    labels = []
    target_names = []

    # Iterate through each person's folder
    for person_name in os.listdir(dataset_path):
        person_dir = os.path.join(dataset_path, person_name)
        if os.path.isdir(person_dir):
            for img_name in os.listdir(person_dir):
                img_path = os.path.join(person_dir, img_name)
                if img_path.endswith(".jpg"):  # Ensure we're processing only jpg images
                    img = Image.open(img_path)
                    img = np.array(img)  # Convert the image to a numpy array
                    images.append(img)
                    labels.append(person_name)
                    if person_name not in target_names:
                        target_names.append(person_name)

    return images, labels, target_names

def train_face_recognizer():
    print("Loading dataset...")
    images, labels, target_names = load_dataset(DATASET_PATH)
    
    print(f"Loaded {len(images)} images with {len(set(labels))} unique people")
    
    # Extract face embeddings
    X = []
    y = []
    
    print("Processing images and extracting face embeddings...")
    for image, label in zip(images, labels):
        try:
            # Detect face using DeepFace (DeepFace will handle face detection and alignment automatically)
            faces = DeepFace.detectFace(image, detector_backend='opencv', enforce_detection=False)
            
            if len(faces) == 1:  # Only use images with one face
                # Get the embedding for the face
                embeddings = DeepFace.represent(faces[0], model_name="VGG-Face", enforce_detection=False)
                if embeddings:
                    X.append(embeddings[0]['embedding'])
                    y.append(label)
        except Exception as e:
            print(f"Error processing image: {e}")
    
    X = np.array(X)
    y = np.array(y)
    
    print(f"Final dataset shape: {X.shape}")
    
    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42
    )
    
    # Train SVM classifier
    print("Training SVM classifier...")
    classifier = SVC(kernel='linear', probability=True)
    classifier.fit(X_train, y_train)
    
    # Evaluate on test set
    print("Evaluating model...")
    y_pred = classifier.predict(X_test)
    evaluate_model(y_test, y_pred)
    
    # Save the model
    model_data = {
        'classifier': classifier,
        'label_encoder': label_encoder
    }
    
    os.makedirs("models", exist_ok=True)
    with open("models/face_recognizer.pkl", "wb") as f:
        pickle.dump(model_data, f)
    
    print("Model trained and saved successfully!")

if __name__ == "__main__":
    train_face_recognizer()
