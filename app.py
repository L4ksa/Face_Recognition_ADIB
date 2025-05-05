import os
import pandas as pd
import numpy as np
import tempfile
import zipfile
import streamlit as st
from deepface import DeepFace

# Step 1: Handle ZIP Upload and Extraction
if uploaded_zip is not None:
    # Save the uploaded ZIP to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".zip") as tmp:
        tmp.write(uploaded_zip.read())
        zip_path = tmp.name
        st.write(f"ZIP file saved to: {zip_path}")
    
    # Extract the ZIP file
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        extract_dir = os.path.join(tempfile.gettempdir(), "lfw_dataset_extracted")
        zip_ref.extractall(extract_dir)
        st.write(f"ZIP file extracted to: {extract_dir}")
    
    # Verify the structure of extracted files
    lfw_base_dir = os.path.join(extract_dir, "lfw-deepfunneled")
    st.write(f"Checking if directory exists: {lfw_base_dir}")
    
    # Check if the directory exists
    if os.path.exists(lfw_base_dir):
        st.write("lfw-deepfunneled directory found.")
    else:
        st.write(f"Directory {lfw_base_dir} does not exist. Please check the ZIP contents.")
    
    # Path for images (check if 'lfw' directory exists inside the base directory)
    LFW_DIR = os.path.join(lfw_base_dir, "lfw-deepfunneled")  # Update to match extracted path
    st.write(f"Checking image directory: {LFW_DIR}")
    
    if os.path.exists(LFW_DIR):
        st.write("Image directory found!")
    else:
        st.write(f"Image directory {LFW_DIR} not found.")
    
    PEOPLE_CSV_PATH = os.path.join(extract_dir, 'people.csv')  # Metadata for people
    
    # Load metadata (people.csv) to manage training/test splits
    if os.path.exists(PEOPLE_CSV_PATH):
        people_df = pd.read_csv(PEOPLE_CSV_PATH)
        st.write("Loaded people.csv:")
        st.write(people_df.head())
    else:
        st.write(f"people.csv not found at {PEOPLE_CSV_PATH}")
    
    # Split metadata into training and testing sets (8 sets for training, 2 sets for testing)
    train_people = people_df.iloc[:8]
    test_people = people_df.iloc[8:]
    
    st.write("Training People:", train_people)
    st.write("Testing People:", test_people)
    
    # Step 2: Load Dataset and Prepare Embeddings for Training
    def load_known_faces(base_dir, people_list):
        embeddings = []
        labels = []
        
        # Ensure that the 'name' column is correctly referenced from people_list
        for person in people_list['name']:
            person_dir = os.path.join(base_dir, person)
            st.write(f"Checking directory for {person}: {person_dir}")
            
            if os.path.isdir(person_dir):  # Ensure the directory exists
                st.write(f"Directory for {person} exists.")
                for image in os.listdir(person_dir):
                    img_path = os.path.join(person_dir, image)
                    st.write(f"Processing image: {img_path}")
                    try:
                        result = DeepFace.represent(img_path=img_path, model_name="Facenet", enforce_detection=False)
                        if result and isinstance(result, list) and "embedding" in result[0]:
                            embedding = result[0]["embedding"]
                            embeddings.append(embedding)
                            labels.append(person)
                    except Exception as e:
                        st.write(f"Error processing {img_path}: {e}")
            else:
                st.write(f"Directory for {person} does not exist.")
        return np.array(embeddings), np.array(labels)
    
    # Load embeddings for known faces (based on the people in the training set)
    known_embeddings, known_labels = load_known_faces(LFW_DIR, train_people)
    st.write(f"Loaded {len(known_embeddings)} embeddings.")
