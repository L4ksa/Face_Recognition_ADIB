import os
import shutil
import zipfile
import cv2
from tqdm import tqdm
import kagglehub  # Make sure to install this: pip install kagglehub

def download_lfw_from_kagglehub(download_path="lfw-data"):
    os.makedirs(download_path, exist_ok=True)

    print("Downloading LFW dataset from kagglehub (jessicali9530/lfw-dataset)...")
    path = kagglehub.dataset_download("jessicali9530/lfw-dataset")  # Downloads and returns path to ZIP

    zip_path = os.path.join(path, "lfw-dataset.zip")
    if os.path.exists(zip_path):
        print("Extracting LFW dataset...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(download_path)
        print("Extraction complete.")
        os.remove(zip_path)
    else:
        raise FileNotFoundError("Failed to download LFW dataset via kagglehub.")

def save_lfw_dataset(kaggle_download_dir="lfw-data", output_dir="dataset", face_cascade_path=None):
    # Search recursively for the correct LFW folder structure
    for root, dirs, files in os.walk(kaggle_download_dir):
        if 'lfw-deepfunneled' in dirs:
            lfw_root = os.path.join(root, 'lfw-deepfunneled', 'lfw-deepfunneled')
            break
    else:
        raise FileNotFoundError(f"LFW folder not found in '{kaggle_download_dir}'. Ensure download succeeded.")

    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    if face_cascade_path is None:
        face_cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

    face_cascade = cv2.CascadeClassifier(face_cascade_path)

    for person_name in tqdm(os.listdir(lfw_root), desc="Extracting faces"):
        person_dir = os.path.join(lfw_root, person_name)
        if not os.path.isdir(person_dir):
            continue

        output_person_dir = os.path.join(output_dir, person_name)
        os.makedirs(output_person_dir, exist_ok=True)

        for img_name in os.listdir(person_dir):
            img_path = os.path.join(person_dir, img_name)
            img = cv2.imread(img_path)
            if img is None:
                continue

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 5)

            for (x, y, w, h) in faces:
                face = img[y:y+h, x:x+w]
                save_path = os.path.join(output_person_dir, img_name)
                cv2.imwrite(save_path, face)
                break  # Save only the first detected face
                
if __name__ == "__main__":
    download_lfw_from_kagglehub()
    save_lfw_dataset()
