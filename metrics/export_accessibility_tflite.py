#!/usr/bin/env python3
# export_accessibility_tflite.py
# Export-only: loads the already-trained accessibility_v2 best.pt and writes a
# float32 TFLite to assets/. No retraining, no validation. Run AFTER:
#     python -m pip install tf_keras onnx_graphsurgeon ai-edge-litert
#
#     python metrics\export_accessibility_tflite.py

import os, glob, shutil

IMGSZ  = 640
ASSETS = "assets"
NAMES  = ["Door", "Stair", "Window"]


def find_best():
    cands = glob.glob("runs/**/accessibility_v2*/weights/best.pt", recursive=True)
    if not cands:
        raise FileNotFoundError("No accessibility_v2*/weights/best.pt under runs/.")
    return max(cands, key=os.path.getmtime)


def main():
    from ultralytics import YOLO
    best = find_best()
    print(f"[export] loading {best}")
    model = YOLO(best)

    exported = model.export(format="tflite", half=False, imgsz=IMGSZ)
    src = str(exported)
    if not src.endswith(".tflite") or not os.path.exists(src):
        hits = glob.glob(os.path.join(os.path.dirname(best),
                                      "best_saved_model", "*float32.tflite"))
        src = hits[0] if hits else None

    if src and os.path.exists(src):
        os.makedirs(ASSETS, exist_ok=True)
        dst = os.path.join(ASSETS, "accessibility_v2_float32.tflite")
        shutil.copy(src, dst)
        with open(os.path.join(ASSETS, "accessibility_v2_labels.txt"), "w") as f:
            f.write("\n".join(NAMES) + "\n")
        print(f"OK -> {dst}")
        print(f"OK -> {os.path.join(ASSETS, 'accessibility_v2_labels.txt')}")
        print("Output tensor shape is [1, 7, 8400]  (4 box + 3 classes).")
    else:
        print("Export ran but .tflite not found; check the *_saved_model/ folder.")


if __name__ == "__main__":
    main()
