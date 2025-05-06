import cv2
import numpy as np
from deepface import DeepFace

def detect_faces(image):
    """
    Detects faces in a BGR image using OpenCV's Haar cascade and returns bounding boxes.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    return faces  # List of bounding box coordinates (x, y, w, h)

def align_face(image):
    """
    Aligns the face using DeepFace's automatic alignment.
    Returns the cropped and aligned face image or None if no face is detected.
    """
    try:
        aligned_face = DeepFace.detectFace(image, detector_backend='opencv', enforce_detection=True)
        return aligned_face
    except Exception as e:
        return None  # Return None if no face is detected

def get_face_embeddings(face_image, model_name="ArcFace"):
    """
    Given a cropped face (160x160), return the 128-d face embedding using DeepFace.
    """
    try:
        embedding = DeepFace.represent(face_image, model_name=model_name, enforce_detection=False)
        if embedding:
            return embedding[0]['embedding']
        else:
            raise ValueError("No face embedding found.")
    except Exception as e:
        raise ValueError(f"Error extracting face embedding: {e}")
