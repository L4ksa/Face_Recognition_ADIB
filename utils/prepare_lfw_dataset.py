import os
import cv2
import shutil
from tqdm import tqdm

def prepare_lfw_dataset(lfw_root, processed_dir):
    """Prepare the LFW dataset for training by extracting faces and saving to processed directory."""
    
    if not os.path.exists(processed_dir):
        os.makedirs(processed_dir)

    # Ensure the processed directory is clean before processing
    if os.path.exists(processed_dir):
        shutil.rmtree(processed_dir)
    os.makedirs(processed_dir)

    print("Processing face images...")
    
    for person_name in tqdm(os.listdir(lfw_root), desc="Extracting faces"):
        person_dir = os.path.join(lfw_root, person_name)
        if not os.path.isdir(person_dir):
            continue
        
        output_person_dir = os.path.join(processed_dir, person_name)
        os.makedirs(output_person_dir, exist_ok=True)
        
        for img_name in os.listdir(person_dir):
            img_path = os.path.join(person_dir, img_name)
            img = cv2.imread(img_path)
            if img is None:
                print(f"Error reading {img_name} for person {person_name}, skipping this image.")
                continue

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
            faces = face_cascade.detectMultiScale(gray, 1.1, 5)

            if len(faces) > 0:
                (x, y, w, h) = faces[0]
                face = img[y:y+h, x:x+w]
                save_path = os.path.join(output_person_dir, img_name)
                cv2.imwrite(save_path, face)
                print(f"Face detected and saved: {img_name} for person {person_name}")
            else:
                print(f"No face detected in {img_name} for person {person_name}, skipping this image.")
    
    print(f"Dataset processed and saved to: {processed_dir}")
    
    # Return the dataset ready for training
    images, labels = load_dataset(processed_dir)
    return images, labels

def load_dataset(dataset_path):
    """Load images and labels from processed dataset"""
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
