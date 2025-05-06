import os
import shutil
from sklearn.datasets import fetch_lfw_people
from PIL import Image
import numpy as np
import cv2
from deepface import DeepFace
from deepface.commons import functions
from tqdm import tqdm

def save_lfw_dataset(output_dir="dataset", lfw_root="lfw-deepfunneled"):
    if not os.path.exists(lfw_root):
        print("Please manually download the LFW dataset and extract it to 'lfw-deepfunneled'")
        return

    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    for person_name in tqdm(os.listdir(lfw_root), desc="Extracting faces"):
        person_dir = os.path.join(lfw_root, person_name)
        if not os.path.isdir(person_dir):
            continue

        for img_name in os.listdir(person_dir):
            img_path = os.path.join(person_dir, img_name)
            img = cv2.imread(img_path)
            if img is None:
                continue

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 5)

            for i, (x, y, w, h) in enumerate(faces):
                face = img[y:y+h, x:x+w]
                save_path = os.path.join(output_dir, f"{person_name}_{img_name}")
                cv2.imwrite(save_path, face)
                break  # Save only one face per image

    print(f"Saved cropped faces to '{output_dir}'")
    
def load_dataset(dataset_path="dataset"):
    """
    Loads the dataset for training by reading images and their corresponding labels
    from the directory structure.
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

if __name__ == "__main__":
    save_lfw_dataset()
