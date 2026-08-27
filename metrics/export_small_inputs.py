#!/usr/bin/env python3
"""Re-export both detectors at smaller input sizes.

WHY
---
Measured on a Galaxy S23 Ultra (SM-S918B), profile build:

    total 2951 ms | preprocessing 61 | INFERENCE 2872 | postprocess 18
    -> 0.4 FPS

Both models cost ~1.9 s per call despite a ~4x difference in parameter count
(yolo11n 2.6M vs the yolo11s-based accessibility model 9.4M). A per-call cost
that ignores model size is overhead, not arithmetic -- and the one thing
identical between them is the 640x640x3 input tensor.

Work scales with inputSize^2, so:

    640 -> 448   0.49x the values   (~2x faster)
    640 -> 320   0.25x the values   (~4x faster)

This exports BOTH sizes for BOTH models so you can A/B on-device without
re-running this script. No Dart changes are needed: ModelGeometry reads the
input size and anchor count from the model's own tensors at load time.

ACCURACY NOTE
-------------
Doors and stairs are large in frame by definition, which is the regime where
dropping resolution costs least. People and cars vary hugely in scale and will
lose distant detections first. Measure before you commit -- and measure STAIR
RECALL specifically, not mAP, which averages away exactly the failure that
matters (a missed stair is a fall).

USAGE
-----
    python metrics/export_small_inputs.py
"""

import glob
import os
import shutil

SIZES = [320, 448]
ASSETS = "assets"

# (label, weights path, output basename)
MODELS = [
    ("COCO (80 classes)", "yolo11n.pt", "yolo11n"),
    ("Accessibility (Door/Stair/Window)",
     "runs/detect/runs/accessibility_v2/weights/best.pt", "accessibility_v2"),
]


def find_export(stem_dir, size):
    """Locate the float32 tflite Ultralytics just wrote."""
    pats = [
        os.path.join(stem_dir, "*_saved_model", "*float32.tflite"),
        os.path.join(stem_dir, "**", "*float32.tflite"),
    ]
    hits = []
    for p in pats:
        hits += glob.glob(p, recursive=True)
    if not hits:
        return None
    return max(hits, key=os.path.getmtime)


def main():
    from ultralytics import YOLO

    os.makedirs(ASSETS, exist_ok=True)
    written = []

    for label, weights, base in MODELS:
        if not os.path.exists(weights):
            print(f"!! SKIP {label}: {weights} not found")
            continue

        for size in SIZES:
            print(f"\n[export] {label} @ {size}x{size} ...")
            try:
                model = YOLO(weights)
                model.export(format="tflite", half=False, imgsz=size)
            except Exception as e:
                print(f"   FAILED: {e}")
                print("   Fix: pip install tf_keras onnx_graphsurgeon ai-edge-litert")
                continue

            src = find_export(os.path.dirname(os.path.abspath(weights)), size)
            if not src:
                src = find_export(".", size)
            if not src or not os.path.exists(src):
                print("   Export ran but no .tflite found; look in *_saved_model/")
                continue

            dst = os.path.join(ASSETS, f"{base}_{size}_float32.tflite")
            shutil.copy(src, dst)
            mb = os.path.getsize(dst) / 1048576
            print(f"   OK -> {dst}  ({mb:.1f} MB)")
            written.append(dst)

    if not written:
        print("\nNothing exported.")
        return

    print("\n" + "=" * 66)
    print("EXPORTED:")
    for w in written:
        print(f"  {w}")
    print("""
TO SHIP THE 320 PAIR (start here -- biggest speed win):

1. pubspec.yaml -- under `assets:` replace the two model lines with:
     - assets/yolo11n_320_float32.tflite
     - assets/accessibility_v2_320_float32.tflite
   (keep the two label .txt lines and assets/vectors/ as they are)

2. lib/values/app_constants.dart:
     cocoModelPath          = 'assets/yolo11n_320_float32.tflite'
     accessibilityModelPath = 'assets/accessibility_v2_320_float32.tflite'
     accessibilityLabelPath = 'assets/accessibility_v2_labels.txt'   (unchanged)

3. flutter build apk --profile && flutter install --profile

4. Capture and compare. If detection quality is too low at 320, repeat with
   the _448_ files -- that is why both were exported.

CHANGE ONE THING PER BUILD. Stacking changes is how the last regression
became hard to attribute.
""")


if __name__ == "__main__":
    main()
