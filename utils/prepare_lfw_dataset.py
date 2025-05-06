import os
import shutil
from sklearn.datasets import fetch_lfw_people
from PIL import Image
import numpy as np

def save_lfw_dataset(output_dir="dataset", min_faces_per_person=5, image_limit_per_person=10):
    """
    Downloads and processes the LFW dataset, saving the images in a structured directory
    for each person.
    """
    # Remove any existing dataset
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    print("Downloading and processing LFW dataset...")
    # Download and load the LFW dataset
    lfw_people = fetch_lfw_people(color=True, resize=1.0, funneled=True,
                                  min_faces_per_person=min_faces_per_person, download_if_missing=True)

    X = lfw_people.images  # Shape is (n_samples, height, width, 3)
    y = lfw_people.target
    target_names = lfw_people.target_names  # List of unique person names

    image_count = {}  # Track the number of images per person
    for idx, (img_array, label_idx) in enumerate(zip(X, y)):
        person_name = target_names[label_idx]
        person_dir = os.path.join(output_dir, person_name.replace(" ", "_"))  # Directory for each person

        # Limit images per person
        if image_count.get(person_name, 0) >= image_limit_per_person:
            continue

        os.makedirs(person_dir, exist_ok=True)
        img = Image.fromarray(np.uint8(img_array))

        # Save image
        img_path = os.path.join(person_dir, f"{person_name.replace(' ', '_')}_{image_count.get(person_name, 0)}.jpg")
        img.convert("RGB").save(img_path)
        image_count[person_name] = image_count.get(person_name, 0) + 1

    print(f"LFW dataset saved to '{output_dir}' with {len(image_count)} people.")
    
def load_dataset(dataset_path="dataset"):
    """
    Loads the dataset for training by reading images and their corresponding labels
    from the directory structure.
    """
    images = []
    labels = []
    target_names = []

    # Iterate through each person's folder
    for person_name in os.listdir(dataset_path):
        person_dir = os.path.join(dataset_path, person_name)
        if os.path.isdir(person_dir):
            for img_name in os.listdir(person_dir):
                img_path = os.path.join(person_dir, img_name)
                if img_path.endswith(".jpg"):  # Ensure we're processing only jpg images
                    img = Image.open(img_path)
                    img = np.array(img)  # Convert the image to a numpy array
                    images.append(img)
                    labels.append(person_name)
                    if person_name not in target_names:
                        target_names.append(person_name)

    return images, labels, target_names

if __name__ == "__main__":
    save_lfw_dataset()
