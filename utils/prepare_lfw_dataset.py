import os
import cv2
import shutil
from tqdm import tqdm

def prepare_lfw_dataset(extracted_dir, processed_dir, face_cascade_path=None):
    # Load face detection model
    if face_cascade_path is None:
        face_cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(face_cascade_path)

    # Clean the processed output dir
    if os.path.exists(processed_dir):
        shutil.rmtree(processed_dir)
    os.makedirs(processed_dir)

    print("Processing face images...")
    for person_name in tqdm(os.listdir(extracted_dir), desc="Extracting faces"):
        person_dir = os.path.join(extracted_dir, person_name)
        if not os.path.isdir(person_dir):
            continue

        output_person_dir = os.path.join(processed_dir, person_name)
        os.makedirs(output_person_dir, exist_ok=True)

        for img_name in os.listdir(person_dir):
            img_path = os.path.join(person_dir, img_name)
            img = cv2.imread(img_path)
            if img is None:
                continue

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
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
