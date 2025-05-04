import streamlit as st
import os
import cv2
import numpy as np
from deepface import DeepFace
import pandas as pd
import matplotlib.pyplot as plt

# Title for the web app
st.title('Face Recognition App')

# Option to trigger model training
train_model = st.button("Train Model")

if train_model:
    # Path to LFW dataset
    lfw_path = r"C:\Users\adibs\Downloads\face_recognition_app\known_faces"
    
    # Function to load and preprocess the dataset
    def preprocess_lfw_data():
        images = []
        labels = []
        for folder_name in os.listdir(lfw_path):
            folder_path = os.path.join(lfw_path, folder_name)
            if os.path.isdir(folder_path):
                for image_name in os.listdir(folder_path):
                    image_path = os.path.join(folder_path, image_name)
                    # Preprocess the face
                    img = DeepFace.detectFace(image_path, enforce_detection=False)
                    images.append(img)
                    labels.append(folder_name)

        return np.array(images), np.array(labels)

    # Preprocessing data
    images, labels = preprocess_lfw_data()

    # Split into train and test sets
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

    # Train the model here (use a pre-trained model like Facenet from DeepFace)
    model = DeepFace.build_model("Facenet")

    # Example training or fine-tuning process (if needed, this is optional)
    # You can implement custom training steps if needed, but for now, we assume we use the pretrained model directly.

    # Display training completion
    st.write("Model training is complete!")

# File uploader for the user to upload an image
uploaded_image = st.file_uploader("Upload an Image", type=["jpg", "png", "jpeg"])

if uploaded_image is not None:
    # Convert the uploaded image to a format suitable for processing
    image_path = os.path.join("temp", uploaded_image.name)
    with open(image_path, "wb") as f:
        f.write(uploaded_image.getbuffer())

    # Display the uploaded image
    img = cv2.imread(image_path)
    st.image(img, channels="BGR", caption="Uploaded Image", use_column_width=True)

    # Perform face recognition and draw bounding boxes with names
    faces_df = DeepFace.detectFace(image_path, enforce_detection=False)

    # Draw bounding boxes and display the image with names
    result_image = draw_bbox(image_path, faces_df)
    st.image(result_image, channels="BGR", caption="Detected Faces", use_column_width=True)

    # Display face matching results
    st.subheader("Face Matching Results")
    img2_path = st.text_input("Enter path for second image for comparison:")

    if img2_path:
        match_result = DeepFace.verify(image_path, img2_path)
        st.write(match_result)

def draw_bbox(image_path, faces_df):
    """Function to draw bounding boxes around detected faces and annotate them with names"""
    img = cv2.imread(image_path)
    for face in faces_df:
        x, y, w, h = face['x'], face['y'], face['w'], face['h']
        name = os.path.basename(face['identity'])
        cv2.rectangle(img, (x, y), (x+w, y+h), (0,255,0), 2)
        cv2.putText(img, name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
    return img
