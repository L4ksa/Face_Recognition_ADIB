import streamlit as st
import os
import joblib
from sklearn.decomposition import PCA
from train_model import train_face_recognizer
from prepare_lfw_dataset import prepare_lfw_dataset
from utils.face_utils import get_face_embeddings, display_sample_faces

# Streamlit UI setup
st.title("Face Recognition App")
st.sidebar.title("Options")

dataset_path = "dataset/processed"
model_path = "model/face_recognition_model.pkl"

# Option to train the model
if st.sidebar.button("Train Model"):
    # Prepare the dataset
    st.write("Preparing the LFW dataset...")
    prepare_lfw_dataset("dataset/extracted", dataset_path)
    
    # Train the model
    st.write("Training the face recognition model...")
    try:
        train_face_recognizer(dataset_path, model_path, progress_callback=st.progress)
        st.success("Model trained successfully!")
    except Exception as e:
        st.error(f"Error during model training: {e}")

# Option to upload image for prediction
uploaded_image = st.sidebar.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

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
