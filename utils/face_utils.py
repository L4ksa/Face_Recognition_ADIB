import os
import cv2
import random
import numpy as np
from deepface import DeepFace
import streamlit as st

def get_face_embeddings(img, model_name="ArcFace"):
    """
    Extract face embeddings from the given image using DeepFace and ArcFace model.

    :param img: Input image (BGR format from OpenCV).
    :param model_name: DeepFace model name (default: "ArcFace").
    :return: The extracted face embedding as a NumPy array, or None if extraction fails.
    """
    try:
        # Convert BGR to RGB for DeepFace
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        print("🔍 Extracting face embeddings using DeepFace...")

        embeddings = DeepFace.represent(
            img_path=img_rgb,
            model_name=model_name,
            enforce_detection=True,  # set to False only if dataset is messy
            detector_backend="opencv",  # can be changed to mtcnn, retinaface, etc.
            align=True
        )

        if embeddings and isinstance(embeddings, list):
            return np.array(embeddings[0]['embedding'])
        else:
            print("⚠️ No embeddings found.")
            return None

    except Exception as e:
        print(f"❌ Error extracting embeddings with ArcFace: {e}")
        return None


def display_sample_faces(processed_dir):
    """
    Display a random sample face from the processed dataset using Streamlit.

    :param processed_dir: Path to the directory containing processed images.
    """
    print("📂 Loading sample face from the processed dataset...")

    # Pick a random person directory
    person_dirs = [d for d in os.listdir(processed_dir) if os.path.isdir(os.path.join(processed_dir, d))]
    if not person_dirs:
        st.error("❌ No processed faces found.")
        return

    # Pick a random person directory
    person_dir = random.choice(person_dirs)
    person_path = os.path.join(processed_dir, person_dir)
    
    # Ensure there are images in the directory
    img_files = [f for f in os.listdir(person_path) if f.endswith(('.jpg', '.png', '.jpeg'))]
    if not img_files:
        st.error(f"❌ No images found for person {person_dir}.")
        return
    
    # Pick a random image from the person's folder
    img_file = random.choice(img_files)
    img_path = os.path.join(person_path, img_file)

    img = cv2.imread(img_path)
    if img is None:
        st.error(f"❌ Error reading {img_file} for person {person_dir}")
    else:
        # Convert image to RGB before displaying with Streamlit
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        print(f"✅ Displaying sample face from {person_dir}")
        st.image(img_rgb, caption=f"Sample face from {person_dir}", use_column_width=True)
