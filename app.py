import streamlit as st
import os
import joblib
import zipfile
import io
from sklearn.decomposition import PCA
from utils.train_model import train_face_recognizer
from utils.prepare_lfw_dataset import prepare_lfw_dataset
from utils.face_utils import get_face_embeddings, display_sample_faces

# Streamlit UI setup
st.title("Face Recognition App")
st.sidebar.title("Options")

# Path for processed dataset and model
dataset_path = "dataset/processed"
model_path = "model/face_recognition_model.pkl"

# Option to upload a custom dataset (ZIP file)
uploaded_zip = st.sidebar.file_uploader("Upload ZIP of Dataset", type=["zip"])

if uploaded_zip is not None:
    # Extract ZIP file to dataset/extracted
    extract_dir = "dataset/extracted"
    if not os.path.exists(extract_dir):
        os.makedirs(extract_dir)
    
    with zipfile.ZipFile(uploaded_zip, "r") as zip_ref:
        zip_ref.extractall(extract_dir)
    
    st.success(f"Dataset uploaded and extracted to {extract_dir}. You can now train the model with this dataset.")

# Option to train the model on the uploaded dataset
if st.sidebar.button("Train Model"):
    if uploaded_zip is None:
        st.error("Please upload a ZIP dataset before training the model.")
    else:
        # Prepare the dataset for training
        st.write("Preparing the dataset for training...")
        try:
            prepare_lfw_dataset("dataset/extracted", dataset_path)
            st.write("Dataset preparation complete.")
        except Exception as e:
            st.error(f"Error in dataset preparation: {e}")
            st.stop()

        # Train the face recognition model
        st.write("Training the face recognition model...")
        try:
            train_face_recognizer(dataset_path, model_path, progress_callback=st.progress)
            st.success("Model trained successfully!")
        except Exception as e:
            st.error(f"Error during model training: {e}")

# Option to upload image for prediction
uploaded_image = st.sidebar.file_uploader("Upload Image for Prediction", type=["jpg", "png", "jpeg"])

if uploaded_image is not None:
    # Display the uploaded image
    st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)
    
    # Process the image and get the embedding
    img = uploaded_image.read()
    embedding = get_face_embeddings(img)

    if embedding is not None:
        st.write("Face embedding successfully extracted!")
        # Load the trained model and perform prediction
        if os.path.exists(model_path):
            model_data = joblib.load(model_path)
            pca = model_data['pca']
            clf = model_data['model']
            label_encoder = model_data['label_encoder']

            # Apply PCA and predict
            img_pca = pca.transform([embedding])
            prediction = clf.predict(img_pca)
            predicted_label = label_encoder.inverse_transform(prediction)[0]
            st.write(f"Predicted Label: {predicted_label}")
        else:
            st.error("Model not found. Please train the model first.")
    else:
        st.error("No face detected in the uploaded image.")

# Display sample faces from processed dataset
if st.sidebar.button("Show Sample Faces"):
    display_sample_faces("dataset/processed")
