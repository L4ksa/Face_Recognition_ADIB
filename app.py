import os
import numpy as np
import pandas as pd
from deepface import DeepFace
import tempfile
import zipfile
import shutil
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from PIL import Image

# Streamlit UI
st.title("Face Recognition System")
st.write("Upload an image, and the system will recognize known faces from the LFW dataset.")

# Step 1: Upload the ZIP of LFW Dataset
uploaded_zip = st.file_uploader("Upload the LFW dataset ZIP file", type="zip")

if uploaded_zip is not None:
    # Save the uploaded ZIP to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".zip") as tmp:
        tmp.write(uploaded_zip.read())
        zip_path = tmp.name
    
    # Extract the ZIP file
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        extract_dir = os.path.join(tempfile.gettempdir(), "lfw_dataset_extracted")
        zip_ref.extractall(extract_dir)

    # Path for extracted images and metadata
    LFW_DIR = os.path.join(extract_dir, "lfw")  # Directory containing images
    PEOPLE_CSV_PATH = os.path.join(extract_dir, 'people.csv')  # Metadata for people
    
    # Load metadata (people.csv) to manage training/test splits
    if os.path.exists(PEOPLE_CSV_PATH):
        people_df = pd.read_csv(PEOPLE_CSV_PATH)
        st.write("Loaded people.csv:")
        st.write(people_df.head())
    
    # Split metadata into training and testing sets (8 sets for training, 2 sets for testing)
    train_people = people_df.iloc[:8]
    test_people = people_df.iloc[8:]
    
    st.write("Training People:", train_people)
    st.write("Testing People:", test_people)
    
    # Step 2: Load Dataset and Prepare Embeddings for Training
    def load_known_faces(base_dir, people_list):
        embeddings = []
        labels = []
        for person in people_list['name']:
            person_dir = os.path.join(base_dir, person)
            if os.path.isdir(person_dir):
                for image in os.listdir(person_dir):
                    img_path = os.path.join(person_dir, image)
                    try:
                        result = DeepFace.represent(img_path=img_path, model_name="Facenet", enforce_detection=False)
                        if result and isinstance(result, list) and "embedding" in result[0]:
                            embedding = result[0]["embedding"]
                            embeddings.append(embedding)
                            labels.append(person)
                    except Exception as e:
                        print(f"Error processing {img_path}: {e}")
        return np.array(embeddings), np.array(labels)
    
    # Load embeddings for known faces (based on the people in the training set)
    known_embeddings, known_labels = load_known_faces(LFW_DIR, train_people)
    
    # Step 3: Train KNN Classifier
    if len(known_embeddings) > 0:
        X_train, X_test, y_train, y_test = train_test_split(known_embeddings, known_labels, test_size=0.2, random_state=42)
        knn = KNeighborsClassifier(n_neighbors=3)
        knn.fit(X_train, y_train)
        
        # Evaluate the model on the test set (optional)
        y_pred = knn.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        
        st.write("Model Evaluation:")
        st.write(f"Accuracy: {accuracy:.2f}")
        st.write(f"Precision: {precision:.2f}")
        st.write(f"Recall: {recall:.2f}")
        st.write(f"F1 Score: {f1:.2f}")
    else:
        knn = None
        st.write("Not enough data to train the model.")
    
    # Step 4: Face Recognition on Uploaded Image
    uploaded_file = st.file_uploader("Choose an image to recognize", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        img = Image.open(uploaded_file)
        st.image(img, caption='Uploaded Image', use_column_width=True)

        # Save uploaded image to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
            img.save(temp_file.name)
            temp_img_path = temp_file.name
        
        try:
            # Get embedding of the uploaded image
            result = DeepFace.represent(img_path=temp_img_path, model_name="Facenet", enforce_detection=False)
            if result and isinstance(result, list) and "embedding" in result[0]:
                embedding = np.array(result[0]["embedding"]).reshape(1, -1)
                
                if knn:
                    # Predict using KNN
                    distances, indices = knn.kneighbors(embedding)
                    threshold = 0.7  # Set a threshold to consider a match
                    
                    st.subheader("Recognition Result")
                    if distances[0][0] < threshold:
                        predicted_identity = knn.classes_[indices[0][0]]
                        st.write(f"Predicted identity: {predicted_identity} (distance: {distances[0][0]:.3f})")
                    else:
                        st.write("No match found (distance too high).")
                else:
                    st.write("KNN model is not trained due to insufficient data.")
            else:
                st.write("No face detected or embedding could not be extracted.")
        except Exception as e:
            st.error(f"Error: {e}")

else:
    st.write("Please upload the LFW dataset ZIP file to proceed.")
