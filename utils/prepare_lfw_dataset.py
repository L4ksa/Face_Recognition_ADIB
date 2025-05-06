import os
import shutil
import streamlit as st
from tqdm import tqdm

def prepare_lfw_dataset(extracted_dir, processed_dir):
    """
    Copies each person's folder from 'lfw-deepfunneled' to 'processed_dir'.
    """

    lfw_root = os.path.join(extracted_dir, "lfw-deepfunneled", "lfw-deepfunneled")
    if os.path.exists(lfw_root):
        st.write(f"Files in lfw-deepfunneled: {os.listdir(lfw_root)}")
    else:
        st.error(f"The directory {lfw_root} does not exist.")

    # Check if lfw-deepfunneled exists
    if not os.path.exists(lfw_root):
        st.error(f"❌ Directory not found: {lfw_root}")
        return

    # Clear processed_dir if it exists and recreate it
    if os.path.exists(processed_dir):
        shutil.rmtree(processed_dir)
    os.makedirs(processed_dir)

    total_persons = 0
    total_images = 0

    # Iterate through subfolders (person directories)
    for person_name in tqdm(os.listdir(lfw_root), desc="Copying folders"):
        person_src_dir = os.path.join(lfw_root, person_name)

        if not os.path.isdir(person_src_dir):
            continue  # Skip if it's not a directory

        # Create a destination folder for each person in processed_dir
        person_dst_dir = os.path.join(processed_dir, person_name)
        shutil.copytree(person_src_dir, person_dst_dir)  # Copy the whole folder with images
        
        # Count the images copied
        img_files = [f for f in os.listdir(person_src_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        total_persons += 1
        total_images += len(img_files)

        st.write(f"✅ Copied {person_name} with {len(img_files)} images.")

    st.write(f"📦 Dataset prepared: {total_persons} persons, {total_images} images copied.")
