import cv2
import numpy as np
from deepface import DeepFace

def detect_faces(image):
    """
    Detects faces in a BGR image using DeepFace's face detector.
    Returns a list of bounding box coordinates (x, y, w, h).
    """
    # DeepFace automatically detects faces
    detected_faces = DeepFace.detectFace(image, detector_backend='opencv', enforce_detection=False)
    return detected_faces

def align_face(image, face_rect):
    """
    Aligns the face using DeepFace's automatic alignment.
    Returns a cropped and aligned face image.
    """
    # DeepFace handles alignment internally
    face = DeepFace.detectFace(image, detector_backend='opencv', enforce_detection=False)
    return face

def get_face_embeddings(face_image):
    """
    Given a cropped face (160x160), return the 128-d face embedding using DeepFace.
    """
    # DeepFace generates embeddings
    embedding = DeepFace.represent(face_image, model_name="VGG-Face", enforce_detection=False)
    
    if embedding:
        return embedding[0]['embedding']
    else:
        raise ValueError("No face embedding could be extracted.")
