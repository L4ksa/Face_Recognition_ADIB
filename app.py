import streamlit as st
import cv2
import os
import pandas as pd
from deepface import DeepFace
from deepface.commons import functions
import matplotlib.pyplot as plt

# Title for the web app
st.title('Face Recognition App')

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
    for index, row in faces_df.iterrows():
        x, y, w, h = int(row['source_x']), int(row['source_y']), int(row['source_w']), int(row['source_h'])
        name = os.path.basename(row['identity'])
        cv2.rectangle(img, (x, y), (x+w, y+h), (0,255,0), 2)
        cv2.putText(img, name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
    return img
