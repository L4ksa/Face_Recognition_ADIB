import os
import cv2
import random
import numpy as np
from deepface import DeepFace
import streamlit as st

def get_face_embeddings(img):
    """
    Extract face embeddings from the given image using DeepFace.

    :param img: Input image (numpy array).
    :return: The extracted face embedding or None if extraction fails.
    """
    try:
        # DeepFace can detect and extract embeddings from faces
        embeddings = DeepFace.represent(img, model_name="VGG-Face", enforce_detection=False, verbose=False)
        
        # If embeddings are returned, extract the first embedding
        if embeddings:
            return embeddings[0]['embedding']
        else:
            return None
    except Exception as e:
        print(f"Error extracting embeddings: {e}")
        return None

def display_sample_faces(processed_dir):
    """
    Display a random sample face from the processed dataset using Streamlit.

    :param processed_dir: Path to the directory containing processed images.
    """
    # Pick a random person directory
    person_dirs = [d for d in os.listdir(processed_dir) if os.path.isdir(os.path.join(processed_dir, d))]
    if not person_dirs:
        st.error("No processed faces found.")
        return

    # Pick a random person directory
    person_dir = random.choice(person_dirs)
    person_path = os.path.join(processed_dir, person_dir)
    
    # Ensure there are images in the directory
    img_files = [f for f in os.listdir(person_path) if f.endswith(('.jpg', '.png', '.jpeg'))]
    if not img_files:
        st.error(f"No images found for person {person_dir}.")
        return
    
    # Pick a random image from the person's folder
    img_file = random.choice(img_files)
    img_path = os.path.join(person_path, img_file)

    img = cv2.imread(img_path)
    if img is None:
        st.error(f"Error reading {img_file} for person {person_dir}")
    else:
        # Convert image to RGB before displaying with Streamlit
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        st.image(img_rgb, caption=f"Sample face from {person_dir}", use_column_width=True)
