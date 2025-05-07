# 🧠 Face Recognition App

A user-friendly web application for training and running face recognition models using the [ArcFace](https://github.com/deepinsight/insightface) embedding method and a Support Vector Machine (SVM) classifier. Built with [Streamlit](https://streamlit.io/) and supports real-time face prediction and dataset visualization.

## 🚀 Features

- Upload a ZIP file of face images organized by class.
- Prepare and process datasets automatically.
- Train an SVM face recognizer with progress tracking and time estimation.
- Predict the identity of uploaded face images.
- Display sample images from the dataset.
- Handles imbalanced datasets using `class_weight='balanced'`.
- No unnecessary page refreshes during training using `st.session_state`.

## 📁 Dataset Structure

The uploaded ZIP file should be downloaded from the link below:
https://www.kaggle.com/datasets/jessicali9530/lfw-dataset/data

Each subfolder name will be used as the person's label.

## 🔧 Installation

1. **Clone the repo**:

git clone https://github.com/yourusername/face-recognition-app.git
cd face-recognition-app

2. **Install dependencies**:
pip install -r requirements.txt

3. **Run the app**:
streamlit run app.py

## 📦 Required Files and Folders
Make sure your project has the following structure:

face-recognition-app/
├── app.py
├── utils/
│   ├── face_utils.py
│   ├── train_model.py
│   └── prepare_lfw_dataset.py
├── dataset/
│   ├── extracted/         # Used after ZIP extraction
│   └── processed/         # Used for training
├── model/
│   └── face_recognition_model.pkl
├── requirements.txt
└── README.md


## 📝 Usage Guide
STEP 1 – Upload a ZIP file containing face images.

STEP 2 – Click “Prepare Dataset” to process and align the images.

STEP 3 – Click “Train Model” to start training (shows progress and estimated time).

STEP 4 – Upload a test image to identify using the trained model.

(Optional) Show random sample images from the dataset.


## 🙋 FAQ
Q: Will my files be deleted if I refresh the app?
No. Your uploaded dataset and trained model will remain in the dataset/ and model/ directories unless you explicitly delete them.

Q: What if my dataset is imbalanced?
This app uses class_weight='balanced' in the SVM classifier to help mitigate class imbalance automatically.

Q: Does it support real-time webcam recognition?
Not yet, but it's possible to extend this with OpenCV's VideoCapture and real-time face embedding.

## 📄 License
MIT License – feel free to use, modify, and distribute.
