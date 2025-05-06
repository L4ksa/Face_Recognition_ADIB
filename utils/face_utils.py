import cv2
import numpy as np
from deepface import DeepFace

def get_face_embeddings(img):
    try:
        # DeepFace can detect and extract embeddings from faces
        embeddings = DeepFace.represent(img, model_name="VGG-Face", enforce_detection=False, verbose=False)
        if embeddings:
            return embeddings[0]['embedding']
        else:
            return None
    except Exception as e:
        print(f"Error extracting embeddings: {e}")
        return None

def display_sample_faces(processed_dir):
    # Pick a random person directory
    person_dirs = [d for d in os.listdir(processed_dir) if os.path.isdir(os.path.join(processed_dir, d))]
    if not person_dirs:
        print("No processed faces found.")
        return

    # Pick a random image from the person's folder
    person_dir = random.choice(person_dirs)
    person_path = os.path.join(processed_dir, person_dir)
    img_file = random.choice(os.listdir(person_path))
    img_path = os.path.join(person_path, img_file)

    img = cv2.imread(img_path)
    if img is None:
        print(f"Error reading {img_file} for person {person_dir}")
    else:
        cv2.imshow(f"Sample face from {person_dir}", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
