import os
import shutil
import streamlit as st
from tqdm import tqdm

def prepare_lfw_dataset(extracted_dir, processed_dir):
    """
    Copies person folders (each containing images) from extracted_dir to processed_dir.
    """
    st.write(f"📂 Processing faces from {extracted_dir}...")

    # Sanity check: what is actually in extracted_dir?
    st.write("📁 Contents of extracted_dir:", os.listdir(extracted_dir))

    if not os.path.exists(extracted_dir):
        st.error(f"❌ {extracted_dir} does not exist!")
        return

    # Clear processed_dir
    if os.path.exists(processed_dir):
        shutil.rmtree(processed_dir)
    os.makedirs(processed_dir)

    total_persons = 0
    total_images = 0

    for person_name in tqdm(os.listdir(extracted_dir), desc="Copying persons"):
        person_dir = os.path.join(extracted_dir, person_name)

        if not os.path.isdir(person_dir):
            continue

        output_person_dir = os.path.join(processed_dir, person_name)
        os.makedirs(output_person_dir, exist_ok=True)

        img_files = [f for f in os.listdir(person_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        total_persons += 1
        total_images += len(img_files)

        st.write(f"🖼️ Found {len(img_files)} image(s) for {person_name}...")

        for img_name in img_files:
            shutil.copy(os.path.join(person_dir, img_name), os.path.join(output_person_dir, img_name))
            st.write(f"✅ Copied: {img_name} for {person_name}")

    st.write(f"✅ Dataset processed and saved to: {processed_dir}")
    st.write(f"🧑‍🤝‍🧑 Total persons processed: {total_persons}")
    st.write(f"🖼️ Total images copied: {total_images}")
