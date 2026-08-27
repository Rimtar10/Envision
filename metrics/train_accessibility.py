#!/usr/bin/env python3
# train_accessibility.py
# Dedicated Door/Stair/Window detector for max accuracy on the accessibility
# classes. Reuses the custom images ALREADY on disk in unified_dataset/ (label
# indices 18=Door, 19=Stair, 20=Window) -> no re-download. Trains yolo11s on
# just these 3 classes, then exports a clean float32 TFLite for the app.
#
#   python metrics\train_accessibility.py

import os, glob, shutil, yaml

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

SRC          = "unified_dataset"          # where the merged data lives
DST          = "accessibility_dataset"    # new 3-class dataset we build
CUSTOM_IDX   = {18: 0, 19: 1, 20: 2}      # unified idx -> new idx
NAMES        = ["Door", "Stair", "Window"]
# YOLO11n, not 11s. The v2 (11s) model measured mAP50 0.706 / mAP50-95 0.488
# but is 38 MB float32 and could only be afforded every 6th frame. For a blind
# pedestrian, refreshing Stair EVERY frame is worth more than the 2-4 mAP points
# the nano backbone costs: a staircase you are told about 500 ms late is a fall.
# Compare the two runs before committing to it — results.csv is written for both.
BASE_MODEL   = "yolo11n.pt"

# 120 epochs was already past convergence for 11s: epochs 90->120 gained
# +0.006 mAP50-95 while the train/val box-loss gap widened 0.14 -> 0.31 (mild
# overfitting). 11n is smaller so it can use the budget, but patience will stop
# it early if it plateaus. This model is DATA-limited, not epoch-limited: more
# and better Door/Stair images will move mAP; more epochs will not.
EPOCHS       = 150
IMGSZ        = 640
BATCH        = 8          # 11n is small enough for batch 8 on an RTX 3050 4GB
                          # (11s needed 2). Bigger batches also train faster.
ASSETS       = "assets"


def build_dataset():
    """Copy only images/labels that contain Door/Stair/Window, remapping the
    class indices to 0/1/2."""
    made = {}
    for split in ["train", "val"]:
        di = os.path.join(DST, "images", split)
        dl = os.path.join(DST, "labels", split)
        os.makedirs(di, exist_ok=True)
        os.makedirs(dl, exist_ok=True)
        kept = 0
        for lf in glob.glob(os.path.join(SRC, "labels", split, "*.txt")):
            out = []
            with open(lf) as f:
                for line in f:
                    p = line.split()
                    if p and int(p[0]) in CUSTOM_IDX:
                        p[0] = str(CUSTOM_IDX[int(p[0])])
                        out.append(" ".join(p))
            if not out:
                continue
            stem = os.path.splitext(os.path.basename(lf))[0]
            # find the matching image (jpg/png/jpeg)
            img = None
            for ext in (".jpg", ".jpeg", ".png"):
                cand = os.path.join(SRC, "images", split, stem + ext)
                if os.path.exists(cand):
                    img = cand
                    break
            if img is None:
                continue
            shutil.copy(img, os.path.join(di, os.path.basename(img)))
            with open(os.path.join(dl, stem + ".txt"), "w") as f:
                f.write("\n".join(out))
            kept += 1
        made[split] = kept
        print(f"[data] {split}: {kept} images with Door/Stair/Window")

    path = os.path.join(DST, "accessibility.yaml")
    with open(path, "w") as f:
        yaml.dump({"path": os.path.abspath(DST),
                   "train": "images/train", "val": "images/val",
                   "nc": len(NAMES), "names": NAMES}, f, sort_keys=False)
    return path, made


def main():
    from ultralytics import YOLO
    data_yaml, made = build_dataset()
    if made["train"] == 0:
        raise SystemExit("No custom images found in unified_dataset/. Did the "
                         "unified build run? (needs indices 18/19/20).")

    print(f"\n[train] {BASE_MODEL} on 3 classes {NAMES} ...")
    model = YOLO(BASE_MODEL)
    model.train(data=data_yaml, epochs=EPOCHS, imgsz=IMGSZ, batch=BATCH,
                name="accessibility_v3", project="runs", patience=30)

    print("\n[val] per-class ...")
    model.val(data=data_yaml, plots=True, verbose=True)

    print("\n[export] TFLite (float16 is what the app loads) ...")
    try:
        # Ultralytics writes BOTH best_float32.tflite and best_float16.tflite
        # into <weights>/best_saved_model/. float16 is ~half the size and about
        # 2x the CPU throughput with no measurable accuracy loss at this scale.
        model.export(format="tflite", half=False, imgsz=IMGSZ)

        hits = glob.glob(
            "runs/**/accessibility_v3*/weights/best_saved_model/*float16.tflite",
            recursive=True)
        if not hits:
            hits = glob.glob(
                "runs/**/accessibility_v3*/weights/best_saved_model/*float32.tflite",
                recursive=True)
        if not hits:
            print("Export ran but no .tflite found; look in *_saved_model/.")
            return

        src = max(hits, key=os.path.getmtime)
        os.makedirs(ASSETS, exist_ok=True)
        dst = os.path.join(ASSETS, "accessibility_v3_float16.tflite")
        shutil.copy(src, dst)
        with open(os.path.join(ASSETS, "accessibility_v3_labels.txt"), "w") as f:
            f.write("\n".join(NAMES) + "\n")
        print(f"OK -> {dst}")
        print(f"OK -> {os.path.join(ASSETS, 'accessibility_v3_labels.txt')}")
    except Exception as e:
        print(f"TFLite export failed: {e}")
        print("Fix: pip install tf_keras onnx_graphsurgeon ai-edge-litert")
        return

    print("""
Done. To ship it:
  1. pubspec.yaml  -> add   - assets/accessibility_v3_float16.tflite
                            - assets/accessibility_v3_labels.txt
                     remove the accessibility_v2_* lines
  2. app_constants.dart -> point accessibilityModelPath / accessibilityLabelPath
                           at the v3 files
  3. detector.dart -> set _accEveryNFrames = 1 (nano is cheap enough to run
                      every frame; that is the whole point of this retrain)
  4. Compare runs/detect/runs/accessibility_v3*/results.csv against
     accessibility_v2/results.csv before committing to the swap.
""")


if __name__ == "__main__":
    main()
