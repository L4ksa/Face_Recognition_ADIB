import face_recognition
import numpy as np
import cv2
import dlib

# Load dlib's face detector and shape predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(
    "utils/shape_predictor_68_face_landmarks.dat"  # Must be downloaded separately
)

def detect_faces(image):
    """
    Detects faces in a BGR image using dlib's frontal face detector.
    Returns a list of dlib rectangles.
    """
    return detector(image, 1)

def align_face(image, face_rect):
    """
    Aligns the face using dlib's 68 facial landmarks.
    Returns a cropped and aligned face image.
    """
    shape = predictor(image, face_rect)
    # Get facial landmarks: eyes and nose
    left_eye = (shape.part(36).x, shape.part(36).y)
    right_eye = (shape.part(45).x, shape.part(45).y)
    nose = (shape.part(30).x, shape.part(30).y)

    # Compute angle
    dy = right_eye[1] - left_eye[1]
    dx = right_eye[0] - left_eye[0]
    angle = np.degrees(np.arctan2(dy, dx))

    # Center between eyes
    eyes_center = ((left_eye[0] + right_eye[0]) // 2,
                   (left_eye[1] + right_eye[1]) // 2)

    # Rotate image
    rot_mat = cv2.getRotationMatrix2D(eyes_center, angle, 1.0)
    aligned = cv2.warpAffine(image, rot_mat, (image.shape[1], image.shape[0]),
                             flags=cv2.INTER_CUBIC)

    # Crop face region again after alignment
    aligned_face = aligned[
        face_rect.top():face_rect.bottom(),
        face_rect.left():face_rect.right()
    ]

    # Resize to fixed size
    return cv2.resize(aligned_face, (160, 160))

def get_face_embeddings(face_image):
    """
    Given a cropped face (160x160), return the 128-d face embedding.
    """
    rgb_face = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
    encodings = face_recognition.face_encodings(rgb_face)

    if encodings:
        return encodings[0]
    else:
        raise ValueError("No face embedding could be extracted.")
