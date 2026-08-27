#!/usr/bin/env python3
# export_coco_models.py
# Export the STOCK pretrained COCO models to float32 TFLite so we can swap the
# app's general-object detector. No training — these already know all 80 COCO
# classes from Ultralytics' full-COCO training.
#
#   yolo11n : 39.5 mAP50-95, tiny/fast  -> speed+accuracy upgrade over yolov8n (37.3)
#   yolo11s : 47.0 mAP50-95, ~2-3x slower -> max accuracy
#
# Run AFTER (in a clean terminal, so no locked DLLs):
#     python -m pip install tf_keras onnx_graphsurgeon ai-edge-litert
#     python metrics\export_coco_models.py

import os, glob, shutil

IMGSZ  = 640
ASSETS = "assets"
MODELS = ["yolo11n.pt", "yolo11s.pt"]   # both; app picks via app_constants flip


def main():
    from ultralytics import YOLO
    os.makedirs(ASSETS, exist_ok=True)
    for name in MODELS:
        stem = name.replace(".pt", "")
        print(f"\n=== exporting {name} -> float32 TFLite ===")
        model = YOLO(name)                      # auto-downloads stock weights
        exported = model.export(format="tflite", half=False, imgsz=IMGSZ)
        src = str(exported)
        if not src.endswith(".tflite") or not os.path.exists(src):
            hits = glob.glob(f"{stem}_saved_model/*float32.tflite")
            src = hits[0] if hits else None
        if src and os.path.exists(src):
            dst = os.path.join(ASSETS, f"{stem}_float32.tflite")
            shutil.copy(src, dst)
            print(f"OK -> {dst}")
        else:
            print(f"!! {name}: .tflite not found; check {stem}_saved_model/")
    print("\nCOCO labels are unchanged (assets/coco_labels.txt, 80 classes).")
    print("Flip the active model in lib/values/app_constants.dart.")


if __name__ == "__main__":
    main()
