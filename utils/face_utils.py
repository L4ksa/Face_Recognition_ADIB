import cv2
from deepface import DeepFace

def align_face(image, detector_backend="opencv"):
    """
    Aligns and crops a face from the input image using DeepFace's detector.
    """
    try:
        aligned_face = DeepFace.detectFace(
            img_path=image,
            detector_backend=detector_backend,
            enforce_detection=True
        )
        return aligned_face
    except Exception as e:
        print(f"[ERROR] Face alignment failed: {e}")
        return None

def get_face_embeddings(image, model_name="ArcFace"):
    """
    Returns a face embedding vector for a single face image.
    The face is first aligned using DeepFace's alignment.
    """
    try:
        aligned_face = align_face(image)
        if aligned_face is None:
            print("[WARN] No aligned face found.")
            return None

        embedding = DeepFace.represent(
            img_path=aligned_face,
            model_name=model_name,
            enforce_detection=False
        )

        if embedding and isinstance(embedding, list) and 'embedding' in embedding[0]:
            return embedding[0]['embedding']
        else:
            print("[INFO] No embedding extracted.")
            return None

    except Exception as e:
        print(f"[ERROR] Error extracting face embedding: {e}")
        return None
