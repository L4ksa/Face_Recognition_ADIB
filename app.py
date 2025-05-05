import streamlit as st
import os
import pickle
import numpy as np
from PIL import Image
from deepface import DeepFace
from utils.prepare_lfw_dataset import save_lfw_dataset
from utils.train_model import train_face_recognizer

st.set_page_config(page_title="Face Recognition System", page_icon=":smiley:")
st.title("Face Recognition System")
st.write("Upload an image or train the model using the LFW dataset")

MODEL_PATH = "models/face_recognizer.pkl"
DATASET_PATH = "dataset"

# Sidebar with dataset preparation
st.sidebar.header("Step 1: Prepare Dataset")
if not os.path.exists(DATASET_PATH):
    if st.sidebar.button("Download LFW Dataset"):
        with st.spinner("Downloading and preparing dataset..."):
            save_lfw_dataset(output_dir=DATASET_PATH)
        st.sidebar.success("LFW dataset ready!")
else:
    st.sidebar.success("LFW dataset already available.")

# Step 2: Train model if not present
st.sidebar.header("Step 2: Train Model")
model_ready = os.path.exists(MODEL_PATH)
if not model_ready:
    if st.sidebar.button("Train Model"):
        with st.spinner("Training model..."):
            train_face_recognizer(DATASET_PATH, MODEL_PATH)
        st.sidebar.success("Model trained and saved.")
        model_ready = True
else:
    st.sidebar.success("Model already trained.")

# Load model function
@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as f:
        model_data = pickle.load(f)
    return model_data['classifier'], model_data['label_encoder']

if model_ready:
    classifier, label_encoder = load_model()

    # Main interface
    st.sidebar.header("Step 3: Face Recognition")
    option = st.sidebar.radio("Select input type:", ("Upload Image",))

    def recognize_faces(image):
        # Convert PIL Image to OpenCV format
        image_cv = np.array(image)
        image_cv = cv2.cvtColor(image_cv, cv2.COLOR_RGB2BGR)

        # Detect faces using DeepFace (it will automatically detect faces)
        faces = DeepFace.detectFace(image_cv, detector_backend='opencv', enforce_detection=False)

        if len(faces) == 0:
            st.warning("No faces detected in the image!")
            return image_cv

        # Get embeddings using DeepFace
        embeddings = []
        for face in faces:
            embedding = DeepFace.represent(face, model_name="VGG-Face", enforce_detection=False)[0]['embedding']
            embeddings.append(embedding)

        # Predict using classifier
        predictions = classifier.predict(embeddings)
        names = label_encoder.inverse_transform(predictions)

        # Draw bounding boxes and names
        for i, name in enumerate(names):
            x, y, w, h = faces[i]
            cv2.rectangle(image_cv, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(image_cv, name, (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        return image_cv

    if option == "Upload Image":
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)

            if st.button("Recognize Faces"):
                with st.spinner("Recognizing faces..."):
                    result_image = recognize_faces(image)
                    result_image = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)
                    st.image(result_image, caption="Result", use_column_width=True)
else:
    st.warning("Please prepare the dataset and train the model first.")
