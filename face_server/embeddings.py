import os
import numpy as np
from deepface import DeepFace

MODEL_NAME = "ArcFace"
DETECTOR_BACKEND = "opencv"


def extract_embedding(image_path):
    """Extract face embedding from a single image."""
    try:
        result = DeepFace.represent(
            img_path=image_path,
            model_name=MODEL_NAME,
            detector_backend=DETECTOR_BACKEND,
            enforce_detection=False,
        )
        if result:
            return np.array(result[0]["embedding"])
        return None
    except Exception as e:
        print(f"[Embeddings] Error extracting from {image_path}: {e}")
        return None


def extract_embeddings_from_dataset(dataset_path):
    """
    Walk through dataset folder and extract embeddings for all images.
    Returns X (embeddings) and y (labels).
    """
    X = []
    y = []

    if not os.path.exists(dataset_path):
        print(f"[Embeddings] Dataset path not found: {dataset_path}")
        return np.array(X), np.array(y)

    people = [
        d for d in os.listdir(dataset_path)
        if os.path.isdir(os.path.join(dataset_path, d))
    ]

    if not people:
        print("[Embeddings] No person folders found in dataset.")
        return np.array(X), np.array(y)

    print(f"[Embeddings] Found {len(people)} people: {people}")

    for person_name in people:
        person_folder = os.path.join(dataset_path, person_name)
        images = [
            f for f in os.listdir(person_folder)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]

        print(f"[Embeddings] Processing {person_name} ({len(images)} images)...")

        for img_file in images:
            img_path = os.path.join(person_folder, img_file)
            embedding = extract_embedding(img_path)
            if embedding is not None:
                X.append(embedding)
                y.append(person_name)
                print(f"  ✓ {img_file}")
            else:
                print(f"  ✗ {img_file} (no face detected)")

    print(f"[Embeddings] Total: {len(X)} embeddings for {len(set(y))} people")
    return np.array(X), np.array(y)


def extract_embedding_from_bytes(image_bytes):
    """Extract face embedding from image bytes (sent from Flutter app)."""
    import tempfile
    import cv2

    try:
        # Save bytes to temp file
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            tmp.write(image_bytes)
            tmp_path = tmp.name

        embedding = extract_embedding(tmp_path)
        os.unlink(tmp_path)
        return embedding
    except Exception as e:
        print(f"[Embeddings] Error extracting from bytes: {e}")
        return None