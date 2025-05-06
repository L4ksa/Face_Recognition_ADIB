import os
import cv2
from tqdm import tqdm

def save_lfw_dataset(zip_path, output_dir="dataset", face_cascade_path=None):
    extracted_dir = os.path.join(output_dir, "extracted")
    processed_dir = os.path.join(output_dir, "processed")

    # Ensure both directories exist
    os.makedirs(extracted_dir, exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)

    print("Extracting ZIP...")
    # Extract the ZIP (existing extraction code)

    print("Post-extraction folder structure:")
    for root, dirs, _ in os.walk(extracted_dir):
        print(f"{root} -> dirs: {dirs}")

    lfw_root = None
    for root, dirs, _ in os.walk(extracted_dir):
        if "lfw-deepfunneled" in dirs:
            lfw_root = os.path.join(root, "lfw-deepfunneled")
            break

    if lfw_root is None or not os.path.exists(lfw_root):
        raise FileNotFoundError("Could not find 'lfw-deepfunneled' folder after extraction.")

    # Prepare face detector
    if face_cascade_path is None:
        face_cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(face_cascade_path)

    if face_cascade.empty():
        print("Error loading face cascade!")
        return

    # Clean the processed output dir
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
                print(f"Error loading image: {img_path}")
                continue

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            print(f"Found {len(faces)} faces in {img_name} for person {person_name}")

            if len(faces) > 0:
                (x, y, w, h) = faces[0]
                face = img[y:y+h, x:x+w]
                save_path = os.path.join(output_person_dir, img_name)
                cv2.imwrite(save_path, face)
                print(f"Face detected and saved: {img_name} for person {person_name}")
            else:
                print(f"No face detected in {img_name} for person {person_name}, skipping this image.")

    print(f"Dataset processed and saved to: {processed_dir}")
