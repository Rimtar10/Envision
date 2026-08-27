import os
import pickle
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from embeddings import extract_embeddings_from_dataset

MODEL_PATH = "face_model.pkl"
ENCODER_PATH = "label_encoder.pkl"
DATASET_PATH = "dataset"
CONFIDENCE_THRESHOLD = 0.5


def train(dataset_path=DATASET_PATH):
    """Train SVM classifier on face embeddings from dataset."""
    print("[Classifier] Starting training...")

    X, y = extract_embeddings_from_dataset(dataset_path)

    if len(X) == 0:
        print("[Classifier] No embeddings found. Cannot train.")
        return False

    if len(set(y)) < 2:
        print("[Classifier] Need at least 2 people to train classifier.")
        return False

    # Encode labels
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)

    # Train SVM
    print(f"[Classifier] Training SVM on {len(X)} samples...")
    model = SVC(kernel="linear", probability=True, C=1.0)
    model.fit(X, y_encoded)

    # Save model and encoder
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    with open(ENCODER_PATH, "wb") as f:
        pickle.dump(encoder, f)

    print(f"[Classifier] Training complete! Saved to {MODEL_PATH}")
    print(f"[Classifier] People recognized: {list(encoder.classes_)}")
    return True


def load_model():
    """Load trained model and encoder."""
    if not os.path.exists(MODEL_PATH) or not os.path.exists(ENCODER_PATH):
        return None, None
    try:
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
        with open(ENCODER_PATH, "rb") as f:
            encoder = pickle.load(f)
        return model, encoder
    except Exception as e:
        print(f"[Classifier] Error loading model: {e}")
        return None, None


def predict(embedding, model, encoder):
    """
    Predict person from embedding.
    Returns (name, confidence) or ("Unknown", 0.0)
    """
    if model is None or embedding is None:
        return "Unknown", 0.0

    try:
        embedding_reshaped = embedding.reshape(1, -1)
        probabilities = model.predict_proba(embedding_reshaped)[0]
        best_idx = np.argmax(probabilities)
        confidence = probabilities[best_idx]

        if confidence < CONFIDENCE_THRESHOLD:
            return "Unknown", float(confidence)

        name = encoder.inverse_transform([best_idx])[0]
        return name, float(confidence)
    except Exception as e:
        print(f"[Classifier] Prediction error: {e}")
        return "Unknown", 0.0


def add_person_and_retrain(person_name, image_paths, dataset_path=DATASET_PATH):
    """
    Add new person's images to dataset and retrain the model.
    Called when registering from the app.
    """
    import shutil

    # Create person folder
    person_folder = os.path.join(dataset_path, person_name)
    os.makedirs(person_folder, exist_ok=True)

    # Copy images to dataset
    for i, img_path in enumerate(image_paths):
        dest = os.path.join(person_folder, f"img{i+1}.jpg")
        shutil.copy(img_path, dest)
        print(f"[Classifier] Saved {dest}")

    # Retrain
    print(f"[Classifier] Retraining after adding {person_name}...")
    return train(dataset_path)


if __name__ == "__main__":
    # Run this directly to train: python classifier.py
    success = train()
    if success:
        print("[Classifier] Model ready!")
    else:
        print("[Classifier] Training failed.")