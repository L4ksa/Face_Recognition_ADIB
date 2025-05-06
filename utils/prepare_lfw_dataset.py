import os
import shutil
import streamlit as st
from tqdm import tqdm

def prepare_lfw_dataset(extracted_dir, processed_dir):
    """
    Prepare the LFW dataset by copying the person directories from the raw dataset.
    
    :param extracted_dir: Path to the extracted dataset (e.g., 'dataset/extracted')
    :param processed_dir: Path to the processed dataset (e.g., 'dataset/processed')
    """
    # Clean the processed output dir if it exists
    if os.path.exists(processed_dir):
        shutil.rmtree(processed_dir)
    os.makedirs(processed_dir)

    # The lfw-deepfunneled folder should contain person directories (i.e., no top-level folder copy)
    lfw_root = os.path.join(extracted_dir, "lfw-deepfunneled")
    st.write(f"📂 Processing faces from {lfw_root}...")

    # Ensure the lfw-deepfunneled directory exists
    if not os.path.exists(lfw_root):
        st.error(f"❌ {lfw_root} does not exist!")
        return

    total_persons = 0
    total_images = 0
    
    # Iterate through each person's directory inside lfw-deepfunneled
    person_names = [name for name in os.listdir(lfw_root) 
                    if os.path.isdir(os.path.join(lfw_root, name))]
    
    for person_name in tqdm(person_names, desc="Extracting faces"):
        person_dir = os.path.join(lfw_root, person_name)

        if not os.path.isdir(person_dir):
            st.warning(f"⚠️ Skipping non-directory: {person_dir}")
            continue
        
        # Create an output directory for each person directly in the processed directory
        output_person_dir = os.path.join(processed_dir, person_name)
        os.makedirs(output_person_dir, exist_ok=True)

        # Log how many images are being processed for this person
        img_files = [f for f in os.listdir(person_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        total_persons += 1
        total_images += len(img_files)

        st.write(f"🖼️ Found {len(img_files)} image(s) for {person_name}...")

        # Copy images to the new folder
        for img_name in img_files:
            img_path = os.path.join(person_dir, img_name)
            save_path = os.path.join(output_person_dir, img_name)
            shutil.copy(img_path, save_path)
            st.write(f"✅ Image copied: {img_name} for person {person_name}")
    
    st.write(f"✅ Dataset processed and saved to: {processed_dir}")
    st.write(f"🧑‍🤝‍🧑 Total persons processed: {total_persons}")
    st.write(f"🖼️ Total images copied: {total_images}")
