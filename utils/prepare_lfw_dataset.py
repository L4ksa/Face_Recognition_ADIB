import os
import shutil
import zipfile
import cv2
from tqdm import tqdm


def extract_uploaded_zip(zip_file, extract_to="uploaded_zips"):
    """
    Save and extract uploaded ZIP file, then return the path to the LFW root folder.
    """
    os.makedirs(extract_to, exist_ok=True)

    zip_path = os.path.join(extract_to, zip_file.name)
    with open(zip_path, "wb") as f:
        f.write(zip_file.getbuffer())

    # Extract ZIP file contents
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
    except zipfile.BadZipFile:
        raise ValueError("The uploaded file is not a valid ZIP archive.")

    os.remove(zip_path)  # Remove the zip after extraction

    # Locate the lfw-deepfunneled path inside the extracted files
    for root, dirs, files in os.walk(extract_to):
        if 'lfw-deepfunneled' in dirs:
            return os.path.join(root, 'lfw-deepfunneled', 'lfw-deepfunneled')

    raise FileNotFoundError("lfw-deepfunneled folder structure not found after extraction.")


def save_lfw_dataset(lfw_root, output_dir="dataset", face_cascade_path=None):
    """
    Detect faces in extracted images and save them under dataset/[person_name] folders.
    """
    if not os.path.exists(lfw_root):
        raise FileNotFoundError(f"Provided LFW root path '{lfw_root}' does not exist.")

    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    # Set the path for the default face cascade if not provided
    if face_cascade_path is None:
        face_cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

    face_cascade = cv2.CascadeClassifier(face_cascade_path)

    for person_name in tqdm(os.listdir(lfw_root), desc="Extracting faces"):
        person_dir = os.path.join(lfw_root, person_name)
        if not os.path.isdir(person_dir):
            continue

        output_person_dir = os.path.join(output_dir, person_name)
        os.makedirs(output_person_dir, exist_ok=True)

        # Process images for each person
        for img_name in os.listdir(person_dir):
            img_path = os.path.join(person_dir, img_name)
            img = cv2.imread(img_path)
            if img is None:
                continue

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 5)

            if len(faces) > 0:
                # Save only the first detected face per image
                (x, y, w, h) = faces[0]
                face = img[y:y+h, x:x+w]
                save_path = os.path.join(output_person_dir, img_name)
                cv2.imwrite(save_path, face)
            else:
                # Log the absence of faces in the image
                print(f"No face detected in {img_name} for person {person_name}")
