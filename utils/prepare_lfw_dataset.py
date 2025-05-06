import os
import shutil
import cv2
from tqdm import tqdm
import streamlit as st

def prepare_lfw_dataset(extracted_dir, processed_dir, face_cascade_path=None):
    """
    Prepare the LFW dataset for training by extracting faces from the raw images.

    :param extracted_dir: Path to the extracted dataset (e.g., 'dataset/extracted')
    :param processed_dir: Path to the processed dataset (e.g., 'dataset/processed')
    :param face_cascade_path: Optional path to the Haar Cascade for face detection.
    """
    st.markdown("🔧 **Preparing dataset...**")

    # Load the face cascade for face detection
    if face_cascade_path is None:
        face_cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(face_cascade_path)

    # Clean the processed output dir if it exists
    if os.path.exists(processed_dir):
        shutil.rmtree(processed_dir)
    os.makedirs(processed_dir)

    # Root of LFW dataset
    lfw_root = os.path.join(extracted_dir, "lfw-deepfunneled")
    st.info(f"📂 Processing faces from `{lfw_root}`")

    # Check if the dataset exists
    if not os.path.exists(lfw_root):
        st.error(f"❌ `{lfw_root}` does not exist!")
        return

    person_dirs = [d for d in os.listdir(lfw_root) if os.path.isdir(os.path.join(lfw_root, d))]
    st.write(f"Found `{len(person_dirs)}` person directories.")

    for person_name in tqdm(person_dirs, desc="Extracting faces"):
        person_dir = os.path.join(lfw_root, person_name)
        output_person_dir = os.path.join(processed_dir, person_name)
        os.makedirs(output_person_dir, exist_ok=True)

        for img_name in os.listdir(person_dir):
            img_path = os.path.join(person_dir, img_name)
            img = cv2.imread(img_path)

            if img is None:
                st.warning(f"⚠️ Failed to read image `{img_name}` for person `{person_name}`. Skipping.")
                continue

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

            if len(faces) > 0:
                x, y, w, h = faces[0]
                face = img[y:y+h, x:x+w]
                save_path = os.path.join(output_person_dir, img_name)
                cv2.imwrite(save_path, face)
                st.text(f"✅ Saved face: `{img_name}` under `{person_name}`")
            else:
                st.warning(f"🚫 No face detected in `{img_name}` for `{person_name}`. Skipping.")

    st.success(f"✅ Dataset processed and saved to: `{processed_dir}`")
