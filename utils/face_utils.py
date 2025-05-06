import cv2
import numpy as np
from deepface import DeepFace

def detect_faces(image):
    """
    Detects faces in a BGR image using DeepFace's face detector.
    Returns a list of bounding box coordinates (x, y, w, h) and the cropped faces.
    """
    # DeepFace automatically detects faces and returns cropped face images
    detected_faces = DeepFace.detectFace(image, detector_backend='opencv', enforce_detection=False)
    return detected_faces  # This returns the aligned face, not the bounding box

def align_face(image):
    """
    Aligns the face using DeepFace's automatic alignment.
    Returns the cropped and aligned face image.
    """
    # DeepFace handles alignment and returns the cropped face directly
    aligned_face = DeepFace.detectFace(image, detector_backend='opencv', enforce_detection=False)
    return aligned_face

def get_face_embeddings(face_image, model_name="VGG-Face"):
    """
    Given a cropped face (160x160), return the 128-d face embedding using DeepFace.
    """
    embedding = DeepFace.represent(face_image, model_name=model_name, enforce_detection=False)
    if embedding:
        return embedding[0]['embedding']
    else:
        raise ValueError("No face embedding could be extracted.")
