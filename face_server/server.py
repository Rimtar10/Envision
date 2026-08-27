import os
import tempfile
from flask import Flask, request, jsonify
from embeddings import extract_embedding_from_bytes, extract_embedding
from classifier import load_model, predict, add_person_and_retrain, train
import ocr

app = Flask(__name__)

DATASET_PATH = "dataset"

# Load model at startup
model, encoder = load_model()
if model is not None:
    print(f"[Server] Model loaded. Recognizes: {list(encoder.classes_)}")
else:
    print("[Server] No model found. Train first or register people via app.")


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/status", methods=["GET"])
def status():
    """Health check — Flutter app pings this to check if server is online."""
    people = []
    if encoder is not None:
        people = list(encoder.classes_)
    return jsonify({
        "status": "online",
        "model_loaded": model is not None,
        "people": people,
        "total": len(people)
    })


@app.route("/recognize", methods=["POST"])
def recognize():
    """
    Recognize a face from an image.
    Flutter sends: multipart/form-data with 'image' field
    Returns: { name, confidence, status }
    """
    global model, encoder

    if "image" not in request.files:
        return jsonify({"error": "No image provided"}), 400

    if model is None:
        return jsonify({
            "name": "Unknown",
            "confidence": 0.0,
            "status": "no_model"
        })

    image_file = request.files["image"]
    image_bytes = image_file.read()

    # Extract embedding
    embedding = extract_embedding_from_bytes(image_bytes)
    if embedding is None:
        return jsonify({
            "name": "Unknown",
            "confidence": 0.0,
            "status": "no_face_detected"
        })

    # Predict
    name, confidence = predict(embedding, model, encoder)
    print(f"[Server] Recognized: {name} ({confidence:.2f})")

    return jsonify({
        "name": name,
        "confidence": round(confidence, 3),
        "status": "success"
    })


@app.route("/ocr", methods=["POST"])
def read_text():
    """
    Extract text (Arabic + English) from an image.
    Flutter sends: multipart/form-data with 'image' field
    Returns: { text, lines, status }
    """
    if "image" not in request.files:
        return jsonify({"error": "No image provided"}), 400

    image_file = request.files["image"]
    image_bytes = image_file.read()

    result = ocr.extract_text_from_bytes(image_bytes)

    if result["status"] == "success":
        print(f"[Server] OCR extracted {len(result['lines'])} line(s)")
    elif result["status"] == "error":
        return jsonify(result), 500

    return jsonify(result)


@app.route("/register", methods=["POST"])
def register():
    """
    Register a new person from the app.
    Flutter sends: multipart/form-data with:
      - 'name': person name
      - 'images': multiple image files
    Returns: { status, message, people }
    """
    global model, encoder

    if "name" not in request.form:
        return jsonify({"error": "No name provided"}), 400

    if "images" not in request.files:
        return jsonify({"error": "No images provided"}), 400

    person_name = request.form["name"].strip().lower().replace(" ", "_")
    image_files = request.files.getlist("images")

    if len(image_files) == 0:
        return jsonify({"error": "No images received"}), 400

    print(f"[Server] Registering {person_name} with {len(image_files)} images...")

    # Save images to temp folder
    temp_paths = []
    try:
        for i, img_file in enumerate(image_files):
            with tempfile.NamedTemporaryFile(
                suffix=".jpg", delete=False, dir="."
            ) as tmp:
                img_file.save(tmp.name)
                temp_paths.append(tmp.name)

        # Add to dataset and retrain
        success = add_person_and_retrain(person_name, temp_paths, DATASET_PATH)

        if success:
            # Reload model
            model, encoder = load_model()
            people = list(encoder.classes_) if encoder else []
            return jsonify({
                "status": "success",
                "message": f"{person_name} registered successfully",
                "people": people
            })
        else:
            return jsonify({
                "status": "error",
                "message": "Training failed. Need at least 2 people registered."
            })

    except Exception as e:
        print(f"[Server] Registration error: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

    finally:
        # Clean up temp files
        for path in temp_paths:
            try:
                os.unlink(path)
            except:
                pass


@app.route("/people", methods=["GET"])
def get_people():
    """Get list of all registered people."""
    people = []
    if os.path.exists(DATASET_PATH):
        people = [
            d for d in os.listdir(DATASET_PATH)
            if os.path.isdir(os.path.join(DATASET_PATH, d))
        ]
    return jsonify({"people": people, "total": len(people)})


@app.route("/delete/<name>", methods=["DELETE"])
def delete_person(name):
    """Delete a person from the dataset and retrain."""
    global model, encoder

    import shutil
    person_folder = os.path.join(DATASET_PATH, name)

    if not os.path.exists(person_folder):
        return jsonify({"error": f"{name} not found"}), 404

    shutil.rmtree(person_folder)
    print(f"[Server] Deleted {name}")

    # Retrain
    train(DATASET_PATH)
    model, encoder = load_model()

    return jsonify({"status": "success", "message": f"{name} deleted"})


@app.route("/retrain", methods=["POST"])
def retrain():
    """Manually trigger retraining."""
    global model, encoder
    success = train(DATASET_PATH)
    if success:
        model, encoder = load_model()
        people = list(encoder.classes_) if encoder else []
        return jsonify({"status": "success", "people": people})
    else:
        return jsonify({"status": "error", "message": "Training failed"})


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import socket

    # Get local IP so you can see what to put in Flutter app
    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    print(f"\n{'='*50}")
    print(f"[Server] Starting Envision Face Server")
    print(f"[Server] Local IP: {local_ip}")
    print(f"[Server] URL: http://{local_ip}:5000")
    print(f"[Server] Put this IP in your Flutter app!")
    print(f"{'='*50}\n")

    app.run(host="0.0.0.0", port=5000, debug=False)