def train_face_recognizer(dataset_path, model_path, features_path=FEATURES_PATH, progress_callback=None, streamlit_error_callback=None):
    try:
        # Step 1: Feature extraction in batches and save to disk
        print("📂 Extracting features in batches...")
        embeddings, labels = extract_features_in_batches(dataset_path, features_path)

        if not embeddings:
            raise ValueError("No embeddings extracted. Training aborted.")

        print(f"✅ Extracted {len(embeddings)} embeddings.")

        # Step 2: Prepare data for training
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(labels)
        X = np.array(embeddings)

        if len(np.unique(y_encoded)) < 2:
            raise ValueError(f"The number of classes has to be greater than one; got {len(np.unique(y_encoded))} class.")

        # Apply PCA if necessary
        pca = None
        if X.shape[0] > 100:
            pca = PCA(n_components=100)
            X = pca.fit_transform(X)
            print("🧬 PCA applied.")

        # Step 3: Train the SVM classifier
        clf = SVC(kernel='linear', probability=True, class_weight='balanced')
        clf.fit(X, y_encoded)
        print("🤖 Model training completed.")

        # Step 4: Save the trained model and PCA
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        joblib.dump({
            'model': clf,
            'pca': pca,
            'label_encoder': label_encoder
        }, model_path)
        print(f"✅ Model saved to: {model_path}")

        gc.collect()

    except Exception as e:
        error_message = f"❌ Training failed: {e}"
        print(error_message)
        if streamlit_error_callback:
            streamlit_error_callback(error_message)
