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
    # Load the face cascade for face detection
    if face_cascade_path is None:
        face_cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(face_cascade_path)

    # Clean the processed output dir if it exists
    if os.path.exists(processed_dir):
        shutil.rmtree(processed_dir)
    os.makedirs(processed_dir)

    # The lfw-deepfunneled folder should contain person directories
    lfw_root = os.path.join(extracted_dir, "lfw-deepfunneled")
    st.write(f"📂 Processing faces from {lfw_root}...")

    # Ensure the lfw-deepfunneled directory exists
    if not os.path.exists(lfw_root):
        st.error(f"❌ {lfw_root} does not exist!")
        return
    
    total_persons = 0
    total_images = 0
    
    # Iterate through each person's directory
    for person_name in tqdm(os.listdir(lfw_root), desc="Extracting faces"):
        person_dir = os.path.join(lfw_root, person_name)
        
        if not os.path.isdir(person_dir):
            st.warning(f"⚠️ Skipping non-directory: {person_dir}")
            continue
        
        # Create an output directory for each person
        output_person_dir = os.path.join(processed_dir, person_name)
        os.makedirs(output_person_dir, exist_ok=True)

        # Log how many images are being processed for this person
        img_files = [f for f in os.listdir(person_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        total_persons += 1
        total_images += len(img_files)

        st.write(f"🖼️ Found {len(img_files)} image(s) for {person_name}...")

        # Loop through each image file in the person's directory
        for img_name in img_files:
            img_path = os.path.join(person_dir, img_name)

            # Read the image
            img = cv2.imread(img_path)
            if img is None:
                st.warning(f"⚠️ Error reading {img_name} for person {person_name}, skipping this image.")
                continue

            # Convert the image to grayscale for face detection
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 5)

            # If a face is detected, save the cropped face image
            if len(faces) > 0:
                (x, y, w, h) = faces[0]
                face = img[y:y+h, x:x+w]
                save_path = os.path.join(output_person_dir, img_name)
                cv2.imwrite(save_path, face)
                st.write(f"✅ Face detected and saved: {img_name} for person {person_name}")
            else:
                st.warning(f"⚠️ No face detected in {img_name} for person {person_name}, skipping this image.")
    
    st.write(f"✅ Dataset processed and saved to: {processed_dir}")
    st.write(f"🧑‍🤝‍🧑 Total persons processed: {total_persons}")
    st.write(f"🖼️ Total images processed: {total_images}")
