import os
import zipfile
import shutil
import cv2
from tqdm import tqdm
import tempfile
import streamlit as st

def save_lfw_dataset(zip_file=None, output_dir="dataset", face_cascade_path=None):
    """
    Extracts the ZIP file containing the LFW dataset and saves the faces in the output directory.
    Works with Streamlit deployment.
    """
    if not zip_file:
        st.error("No ZIP file uploaded.")
        return

    # Use a temporary directory for extraction
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Create output_dir if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            st.write(f"Created output directory: {output_dir}")

        # Save the uploaded ZIP file to the temporary directory
        zip_file_path = os.path.join(tmp_dir, "lfw.zip")
        with open(zip_file_path, "wb") as f:
            f.write(zip_file.getbuffer())

        # Extract the ZIP file
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            zip_ref.extractall(tmp_dir)
        
        # Print directory structure after extraction
        st.write("Directory structure after extraction:")
        for root, dirs, files in os.walk(tmp_dir):
            st.write(f"Root: {root}, Directories: {dirs}, Files: {files}")

        # Adjust the path to correctly reference the "lfw-deepfunneled" folder
        lfw_root = os.path.join(tmp_dir, "lfw-deepfunneled", "lfw-deepfunneled")

        if not os.path.exists(lfw_root):
            # If we can't find it, check if it's one more level deep
            lfw_root = os.path.join(tmp_dir, "lfw-deepfunneled")

        # Debug: Print the final lfw_root path
        st.write(f"Final LFW root path: {lfw_root}")

        if not os.path.exists(lfw_root):
            st.error(f"LFW folder not found in '{tmp_dir}'. Ensure extraction succeeded.")
            return

        # Flatten the folder structure if necessary (move files out of nested folders)
        st.write("Flattening folder structure...")
        for subdir, dirs, files in os.walk(lfw_root):
            for dir_name in dirs:
                nested_dir = os.path.join(subdir, dir_name)
                if nested_dir.endswith('lfw-deepfunneled'):
                    for person_name in os.listdir(nested_dir):
                        person_dir = os.path.join(nested_dir, person_name)
                        if os.path.isdir(person_dir):
                            # Move all files to the correct location
                            target_person_dir = os.path.join(lfw_root, person_name)
                            if not os.path.exists(target_person_dir):
                                os.makedirs(target_person_dir)
                            
                            for img_name in os.listdir(person_dir):
                                img_path = os.path.join(person_dir, img_name)
                                target_img_path = os.path.join(target_person_dir, img_name)
                                shutil.move(img_path, target_img_path)
                            
                            # Remove the now empty directory
                            shutil.rmtree(person_dir)

        # Set the path for the default face cascade if not provided
        if face_cascade_path is None:
            face_cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

        face_cascade = cv2.CascadeClassifier(face_cascade_path)

        # Create the output directory for processed faces
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        os.makedirs(output_dir)

        # Process each person folder in the extracted LFW dataset
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
                    st.write(f"No face detected in {img_name} for person {person_name}")
    
    st.write("Face extraction completed successfully.")
