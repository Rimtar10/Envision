#!/usr/bin/env python3
"""
compare_coco_models.py
======================
Local YOLOv8n vs YOLO11n comparison on COCO — no Colab, no manual dataset.

You do NOT need to download COCO. Ultralytics auto-downloads everything it needs
on first run:
  - the model weights  (yolov8n.pt, yolo11n.pt)
  - the coco128 dataset (a 128-image COCO subset, ~7 MB)
into a local `datasets/` folder. (You already have coco128 from your earlier run.)

Run from your project ROOT, with your venv active:
    pip install ultralytics matplotlib
    python metrics/compare_coco_models.py

Outputs -> metrics/output/coco_compare/:
    comparison.csv              side-by-side metrics table
    map_comparison.png          mAP50 / mAP50-95 bar chart (both models)
    speed_size_comparison.png   inference-ms + params bar chart
    yolov8n/ , yolo11n/         each model's Ultralytics curves + confusion matrix

It also exports yolo11n -> float32 TFLite and copies it to
assets/yolo11n_float32.tflite (the file the app now expects). Set
EXPORT_TFLITE = False to skip the export step.
"""

from __future__ import annotations

import csv
import os
import shutil

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ── CONFIG ───────────────────────────────────────────────────────────────────
MODELS = ["yolov8n.pt", "yolo11n.pt"]

# coco.yaml = COCO val2017, a proper HELD-OUT set (~1 GB, auto-downloads once).
#
# This was "coco128.yaml" — the first 128 images of COCO *train2017*. Both sets
# of weights were trained on those images, so the resulting mAP50 figures
# (0.607 for yolov8n, 0.671 for yolo11n) were train-set scores. They are
# inflated and cannot support a claim that one model is better than the other.
# Anything already published from the coco128 run should be withdrawn.
DATA = "coco.yaml"
OUT = os.path.join("metrics", "output", "coco_compare")
EXPORT_TFLITE = True
ASSETS_DIR = "assets"

BLUE, GREEN, AMBER = "#2563eb", "#16a34a", "#f59e0b"


def main() -> None:
    from ultralytics import YOLO  # imported here so a missing install fails loudly

    os.makedirs(OUT, exist_ok=True)
    rows = []

    for name in MODELS:
        print(f"\n=== validating {name} on {DATA} ===")
        m = YOLO(name)                      # auto-downloads the weights
        r = m.val(data=DATA, plots=True)    # auto-downloads coco128 if missing
        params_m = sum(p.numel() for p in m.model.parameters()) / 1e6
        rows.append({
            "model": name.replace(".pt", ""),
            "params_M": round(params_m, 2),
            "mAP50": round(float(r.box.map50), 3),
            "mAP50_95": round(float(r.box.map), 3),
            "precision": round(float(r.box.mp), 3),
            "recall": round(float(r.box.mr), 3),
            "infer_ms": round(float(r.speed["inference"]), 1),
        })
        # copy this model's own plots (PR/F1 curves, confusion matrix)
        sub = os.path.join(OUT, name.replace(".pt", ""))
        os.makedirs(sub, exist_ok=True)
        try:
            for f in os.listdir(r.save_dir):
                if f.endswith((".png", ".jpg")):
                    shutil.copy(os.path.join(r.save_dir, f), os.path.join(sub, f))
        except Exception:
            pass

    # ── table -> CSV ─────────────────────────────────────────────────────────
    keys = list(rows[0].keys())
    with open(os.path.join(OUT, "comparison.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)

    labels = [r["model"] for r in rows]
    x = np.arange(len(labels))
    width = 0.35

    # ── accuracy chart ───────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 4.5))
    b1 = ax.bar(x - width / 2, [r["mAP50"] for r in rows], width, label="mAP50", color=BLUE)
    b2 = ax.bar(x + width / 2, [r["mAP50_95"] for r in rows], width, label="mAP50-95", color=GREEN)
    for bars in (b1, b2):
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.01,
                    f"{b.get_height():.3f}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(labels); ax.set_ylim(0, 1)
    ax.set_ylabel("score"); ax.set_title(f"Accuracy on {DATA}")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "map_comparison.png"), dpi=150); plt.close(fig)

    # ── speed + size chart (dual axis) ───────────────────────────────────────
    fig, ax1 = plt.subplots(figsize=(7, 4.5))
    ax1.bar(x - width / 2, [r["infer_ms"] for r in rows], width, label="inference ms", color=BLUE)
    ax1.set_ylabel("inference ms (GPU/CPU of this machine)")
    ax1.set_xticks(x); ax1.set_xticklabels(labels)
    ax2 = ax1.twinx()
    ax2.bar(x + width / 2, [r["params_M"] for r in rows], width, label="params (M)", color=AMBER)
    ax2.set_ylabel("params (M)")
    ax1.set_title("Speed & size")
    ax1.grid(axis="y", alpha=0.3)
    lines = ax1.get_legend_handles_labels()[0] + ax2.get_legend_handles_labels()[0]
    labs = ax1.get_legend_handles_labels()[1] + ax2.get_legend_handles_labels()[1]
    ax1.legend(lines, labs, loc="upper right")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "speed_size_comparison.png"), dpi=150); plt.close(fig)

    # ── print the table ──────────────────────────────────────────────────────
    print("\n=== COMPARISON (COCO) ===")
    header = " | ".join(f"{k:>10}" for k in keys)
    print(header); print("-" * len(header))
    for r in rows:
        print(" | ".join(f"{str(r[k]):>10}" for k in keys))
    print(f"\nCharts + CSV -> {OUT}/")

    # ── export yolo11n -> float32 tflite for the app ─────────────────────────
    if EXPORT_TFLITE:
        print("\nExporting yolo11n -> float32 TFLite ...")
        try:
            exported = YOLO("yolo11n.pt").export(format="tflite", imgsz=640, half=False)
            src = str(exported) if str(exported).endswith(".tflite") else None
            if src is None or not os.path.exists(src):
                for root, _, files in os.walk("yolo11n_saved_model"):
                    for f in files:
                        if f.endswith("float32.tflite"):
                            src = os.path.join(root, f)
            if src and os.path.exists(src):
                os.makedirs(ASSETS_DIR, exist_ok=True)
                dst = os.path.join(ASSETS_DIR, "yolo11n_float32.tflite")
                shutil.copy(src, dst)
                print(f"OK -> {dst}  (the app is already pointed at this file)")
            else:
                print("Export ran but I couldn't find the .tflite; look in yolo11n_saved_model/")
        except Exception as e:
            print(f"TFLite export failed: {e}")
            print("Fix: pip install onnx onnx2tf tensorflow   (then re-run)")
            print("Or export it in Colab and drop the file into assets/ manually.")


if __name__ == "__main__":
    main()
