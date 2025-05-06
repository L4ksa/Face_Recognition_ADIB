import cv2
import numpy as np
from deepface import DeepFace

def align_face(image):
    try:
        aligned_face = DeepFace.detectFace(image, detector_backend='opencv', enforce_detection=True)
        return aligned_face
    except Exception:
        return None

def get_face_embeddings(face_image, model_name="ArcFace"):
    try:
        embedding = DeepFace.represent(face_image, model_name=model_name, enforce_detection=False)
        if embedding:
            return embedding[0]['embedding']
        else:
            raise ValueError("No face embedding found.")
    except Exception as e:
        raise ValueError(f"Error extracting face embedding: {e}")
