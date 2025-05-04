import streamlit as st
import os
import numpy as np
import cv2
from deepface import DeepFace
import matplotlib.pyplot as plt

# Set the path to the LFW dataset
lfw_path = "./known_faces"  # Update with your dataset path

# Load the images and labels from the LFW dataset
def preprocess_lfw_data():
    images = []
    labels = []
    for folder_name in os.listdir(lfw_path):
        folder_path = os.path.join(lfw_path, folder_name)
        if os.path.isdir(folder_path):
            for image_name in os.listdir(folder_path):
                image_path = os.path.join(folder_path, image_name)
                try:
                    img = DeepFace.detectFace(image_path, enforce_detection=False)
                    images.append(img)
                    labels.append(folder_name)
                except Exception as e:
                    st.error(f"Error processing image {image_path}: {e}")

    st.write(f"Loaded {len(images)} images.")
    return np.array(images), np.array(labels)

# Preprocess LFW data
images, labels = preprocess_lfw_data()

# Use the pre-trained model for face recognition (Facenet, VGG-Face, or DeepFace)
model_name = "Facenet"  # You can change to "VGG-Face" or "DeepFace" for comparison

# Perform face recognition on a test image from the LFW dataset
def perform_face_recognition(uploaded_image, model_name):
    # Read the image file as a byte array
    image_bytes = uploaded_image.read()
    
    # Convert byte array to numpy array
    nparr = np.frombuffer(image_bytes, np.uint8)
    
    # Decode the numpy array into an image
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # Check if the image was successfully decoded
    if img is None:
        st.error("Error decoding image. Please upload a valid image file.")
        return None
    
    # Perform face recognition using the pre-trained model
    result = DeepFace.find(img_path=img, db_path=lfw_path, model_name=model_name)
    return result

# Streamlit Interface
st.title('Face Recognition App')

# Allow user to upload a test image
uploaded_image = st.file_uploader("Choose an image for face recognition", type=["jpg", "png", "jpeg"])

if uploaded_image is not None:
    # Display the uploaded image
    img = np.array(bytearray(uploaded_image.read()), dtype=np.uint8)
    img = cv2.imdecode(img, cv2.IMREAD_COLOR)
    st.image(img, channels="BGR", use_column_width=True)

    # Perform face recognition using the pre-trained model
    result = perform_face_recognition(uploaded_image, model_name)

    # Show results
    if result is not None:
        st.write(f"Recognition results using {model_name}:")
        st.write(result)
