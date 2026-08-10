#!/usr/bin/env python3
"""
accuracy_report.py
==================

Accuracy report for the FULL Envision detection stack — BOTH models the app
runs:
  1. accessibility model  (YOLOv8n, 5 classes: Door, Gate, Stair, Window, Building)
  2. COCO model           (stock YOLOv8n, 80 everyday classes: person, chair, car, ...)

For each model it runs real Ultralytics validation and writes, into a per-model
subfolder of ./output/:
  - per_class_map.png        per-class mAP@50-95 (adapts to 5 or 80 classes)
  - overall_metrics.png      precision / recall / mAP50 / mAP50-95
  - class_distribution.png   validation instances per class
  - confusion_matrix.png     normalized confusion matrix
  - per_class_metrics.csv    machine-readable table
  - metrics_summary.txt      human-readable summary
  - ultralytics_*.png        Ultralytics' own PR/F1 curves, copied in

Modes
-----
RUN_VALIDATION = True  (default): real numbers — runs model.val() on each model.
  Needs `ultralytics`, the model file, and the dataset. The COCO model uses
  coco128.yaml (a tiny 128-image set that auto-downloads in seconds) for a quick
  REAL evaluation; switch its `data` to "coco.yaml" for the full COCO val2017.

If validation can't run for a model (ultralytics missing, file not found, no
network), the script falls back to embedded numbers where it has them
(accessibility model) and otherwise skips that model with a clear message — so
it never crashes.

Install
-------
    pip install matplotlib numpy
    pip install ultralytics            # needed for real validation

Usage
-----
    python accuracy_report.py

On Windows PowerShell, run the script with `python` or `py -3` rather than
`/usr/bin/env python3`.
"""

from __future__ import annotations

import csv
import os
import shutil

import matplotlib

matplotlib.use("Agg")  # headless: write files, no GUI needed
import matplotlib.pyplot as plt
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
RUN_VALIDATION = True  # real numbers only — there is no fallback any more.
OUTPUT_DIR = os.path.join("metrics", "output")

# One entry per model the app runs. `model` may be a .pt OR an exported .tflite
# (validate the .tflite to measure exactly what's deployed on-device).
MODELS = [
    {
        "name": "accessibility",
        # The SHIPPING model. The old path here (accessibility_detector) no
        # longer exists, so every run silently fell back to the hard-coded
        # EMBEDDED numbers below — which described the RETIRED 5-class v1 model.
        # metrics_summary.txt was reporting a model the app had stopped using.
        "model": "runs/detect/runs/accessibility_v2/weights/best.pt",
        "data": "accessibility_dataset/accessibility.yaml",
        "untrained": [],
    },
    {
        "name": "coco",
        # yolo11n is what the app ships now (assets/yolo11n_float32.tflite).
        "model": "yolo11n.pt",
        # coco.yaml = val2017, a proper HELD-OUT set (~1 GB, auto-downloads).
        #
        # This was "coco128.yaml", which is the first 128 images of COCO
        # *train2017* — images these weights were TRAINED on. Those numbers
        # (mAP50 0.607 / 0.671) are train-set scores: inflated, and not a valid
        # basis for comparing models. Do not put them in a report.
        "data": "coco.yaml",
        "untrained": [],
    },
]

PRIMARY = "#2563eb"   # blue
ACCENT = "#16a34a"    # green
MUTED = "#9ca3af"     # gray (untrained / empty classes)

# ─────────────────────────────────────────────────────────────────────────────
# NO EMBEDDED FALLBACK.
#
# There used to be a dict of hard-coded numbers here that the script fell back
# to whenever live validation failed. Because the model path had gone stale, it
# fell back EVERY time — and wrote those constants into metrics_summary.txt as
# though they had just been measured. They described the retired 5-class model.
#
# A metrics script that silently invents numbers is worse than one that
# crashes. If validation cannot run, this now fails loudly and says why.
# ─────────────────────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────────────────────
def run_validation(cfg: dict) -> dict:
    """Run Ultralytics validation for one model; return a normalized results dict."""
    from ultralytics import YOLO  # lazy import so offline mode needs no deps

    print(f"[live] {cfg['name']}: loading {cfg['model']}")
    model = YOLO(cfg["model"])
    print(f"[live] {cfg['name']}: validating on {cfg['data']} ...")
    metrics = model.val(data=cfg["data"], save_json=False, plots=True)

    names = metrics.names                       # {idx: name}
    maps = metrics.box.maps                      # mAP@50-95 per class
    ap_idx = list(metrics.box.ap_class_index)    # classes actually scored
    nt = getattr(metrics.box, "nt_per_class", None)

    ordered = [names[i] for i in sorted(names)]
    per_class, instances = {}, {}
    for i in sorted(names):
        name = names[i]
        per_class[name] = float(maps[i]) if i in ap_idx else None
        if nt is not None and i < len(nt):
            instances[name] = int(nt[i])

    cm = None
    try:
        candidate = np.array(metrics.confusion_matrix.matrix)
        expected = len(ordered) + 1
        if candidate.ndim == 2 and candidate.shape == (expected, expected):
            cm = candidate
    except Exception:
        pass

    # Copy Ultralytics' own plots (PR/F1 curves, its confusion matrix) into ours.
    out = os.path.join(OUTPUT_DIR, cfg["name"])
    os.makedirs(out, exist_ok=True)
    try:
        save_dir = str(metrics.save_dir)
        for fname in os.listdir(save_dir):
            if fname.endswith((".png", ".jpg")):
                shutil.copy(os.path.join(save_dir, fname),
                            os.path.join(out, f"ultralytics_{fname}"))
    except Exception:
        pass

    return {
        "name": cfg["name"],
        "class_names": ordered,
        "overall": {
            "precision": float(metrics.box.mp),
            "recall": float(metrics.box.mr),
            "mAP50": float(metrics.box.map50),
            "mAP50_95": float(metrics.box.map),
        },
        "per_class_map50_95": per_class,
        "val_instances": instances,
        "confusion_matrix": cm,
    }


# ─────────────────────────────────────────────────────────────────────────────
# PLOTS (adapt to 5 or 80 classes)
# ─────────────────────────────────────────────────────────────────────────────
def plot_per_class_map(results: dict, path: str) -> bool:
    data = results["per_class_map50_95"]
    pairs = [(k, data[k]) for k in results["class_names"] if data.get(k) is not None]
    if not pairs:
        return False
    pairs.sort(key=lambda kv: kv[1], reverse=True)
    labels = [k for k, _ in pairs]
    values = [v for _, v in pairs]
    mean_v = float(np.mean(values))

    if len(labels) > 12:  # many classes (COCO) → horizontal, sorted
        fig, ax = plt.subplots(figsize=(8, max(4.5, len(labels) * 0.26)))
        ax.barh(labels[::-1], values[::-1], color=PRIMARY)
        ax.axvline(mean_v, color=ACCENT, linestyle="--", linewidth=1.2,
                   label=f"mean {mean_v:.3f}")
        ax.set_xlim(0, 1.0)
        ax.set_xlabel("mAP@50-95")
        ax.legend(loc="lower right")
    else:                 # few classes → vertical with labels
        fig, ax = plt.subplots(figsize=(7, 4.5))
        bars = ax.bar(labels, values, color=PRIMARY)
        bars[int(np.argmax(values))].set_color(ACCENT)  # highlight best
        for b, v in zip(bars, values):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.3f}",
                    ha="center", va="bottom", fontsize=10, fontweight="bold")
        ax.set_ylim(0, 1.0)
        ax.set_ylabel("mAP@50-95")
    ax.set_title(f"{results['name']} — per-class mAP@50-95 "
                 f"({len(labels)} evaluated classes)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return True


def plot_overall(results: dict, path: str) -> None:
    o = results["overall"]
    labels = ["Precision", "Recall", "mAP@50", "mAP@50-95"]
    values = [o["precision"], o["recall"], o["mAP50"], o["mAP50_95"]]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(labels, values, color=[PRIMARY, PRIMARY, ACCENT, ACCENT])
    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.3f}",
                ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("score")
    ax.set_title(f"{results['name']} — overall detection metrics")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_distribution(results: dict, path: str) -> None:
    inst = results["val_instances"]
    names = results["class_names"]
    if not inst:
        return
    if len(names) > 12:  # many classes → horizontal, sorted by count
        items = sorted(inst.items(), key=lambda kv: kv[1], reverse=True)
        labels = [k for k, _ in items]
        values = [v for _, v in items]
        fig, ax = plt.subplots(figsize=(8, max(4.5, len(labels) * 0.26)))
        colors = [MUTED if v == 0 else PRIMARY for v in values]
        ax.barh(labels[::-1], values[::-1], color=colors[::-1])
        ax.set_xlabel("validation instances")
    else:
        labels = list(names)
        values = [inst.get(k, 0) for k in labels]
        colors = [MUTED if v == 0 else PRIMARY for v in values]
        fig, ax = plt.subplots(figsize=(7, 4.5))
        bars = ax.bar(labels, values, color=colors)
        top = max(values) if values else 1
        for b, v in zip(bars, values):
            ax.text(b.get_x() + b.get_width() / 2, v + top * 0.01, str(v),
                    ha="center", va="bottom", fontsize=10)
        ax.set_ylabel("validation instances")
    ax.set_title(f"{results['name']} — class distribution (validation set)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_confusion(results: dict, path: str) -> bool:
    cm = results.get("confusion_matrix")
    if cm is None:
        return False
    labels = list(results["class_names"]) + ["background"]
    cm = np.array(cm, dtype=float)
    if cm.ndim != 2 or cm.shape != (len(labels), len(labels)):
        print(f"  .. {results['name']}: skipping confusion matrix with shape {cm.shape}")
        return False
    col_sums = cm.sum(axis=0, keepdims=True)
    norm = np.divide(cm, col_sums, out=np.zeros_like(cm), where=col_sums != 0)

    n = len(labels)
    fig, ax = plt.subplots(figsize=(max(6, n * 0.5), max(5, n * 0.5)))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("True")
    ax.set_ylabel("Predicted")
    ax.set_title(f"{results['name']} — confusion matrix (column-normalized)")
    if n <= 12:  # annotate only when readable
        for i in range(n):
            for j in range(n):
                if norm[i, j] > 0.01:
                    ax.text(j, i, f"{norm[i, j]:.2f}", ha="center", va="center",
                            color="white" if norm[i, j] > 0.5 else "black",
                            fontsize=8)
    fig.colorbar(im, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return True


def write_csv(results: dict, path: str) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["class", "val_instances", "mAP50_95"])
        for name in results["class_names"]:
            m = results["per_class_map50_95"].get(name)
            w.writerow([name, results["val_instances"].get(name, 0),
                        "" if m is None else f"{m:.4f}"])
        o = results["overall"]
        w.writerow([])
        w.writerow(["overall_precision", f"{o['precision']:.4f}"])
        w.writerow(["overall_recall", f"{o['recall']:.4f}"])
        w.writerow(["overall_mAP50", f"{o['mAP50']:.4f}"])
        w.writerow(["overall_mAP50_95", f"{o['mAP50_95']:.4f}"])


def write_summary(results: dict, path: str) -> None:
    o = results["overall"]
    lines = [
        f"ENVISION — {results['name'].upper()} MODEL ACCURACY",
        "=" * 48,
        "",
        f"Overall precision : {o['precision']:.3f}",
        f"Overall recall    : {o['recall']:.3f}",
        f"mAP@50            : {o['mAP50']:.3f}",
        f"mAP@50-95         : {o['mAP50_95']:.3f}",
        "",
        "Per-class mAP@50-95 (classes with validation instances):",
    ]
    scored = [(n, results["per_class_map50_95"].get(n),
               results["val_instances"].get(n, 0))
              for n in results["class_names"]
              if results["per_class_map50_95"].get(n) is not None]
    scored.sort(key=lambda t: t[1], reverse=True)
    for name, m, inst in scored:
        lines.append(f"  {name:<14}: {m:.3f} ({inst} instances)")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


# ─────────────────────────────────────────────────────────────────────────────
def generate(results: dict) -> None:
    out = os.path.join(OUTPUT_DIR, results["name"])
    os.makedirs(out, exist_ok=True)
    plot_per_class_map(results, os.path.join(out, "per_class_map.png"))
    plot_overall(results, os.path.join(out, "overall_metrics.png"))
    plot_distribution(results, os.path.join(out, "class_distribution.png"))
    has_cm = plot_confusion(results, os.path.join(out, "confusion_matrix.png"))
    write_csv(results, os.path.join(out, "per_class_metrics.csv"))
    write_summary(results, os.path.join(out, "metrics_summary.txt"))
    o = results["overall"]
    print(f"  -> {results['name']}: mAP50={o['mAP50']:.3f} "
          f"mAP50-95={o['mAP50_95']:.3f}  ({out}/)"
          + ("" if has_cm else "  [confusion matrix needs live validation]"))


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"[mode] {'LIVE validation' if RUN_VALIDATION else 'OFFLINE (embedded)'}")

    failures = []
    for cfg in MODELS:
        if not RUN_VALIDATION:
            raise SystemExit(
                "RUN_VALIDATION is False and there is no fallback data. "
                "Set it to True and install ultralytics."
            )
        if not os.path.exists(cfg["model"]):
            failures.append(
                f"{cfg['name']}: model not found at {cfg['model']}"
            )
            continue
        try:
            results = run_validation(cfg)
        except Exception as e:
            failures.append(f"{cfg['name']}: validation failed -> {e}")
            continue
        generate(results)

    if failures:
        raise SystemExit(
            "\n!! REPORT INCOMPLETE — these models were NOT evaluated:\n  "
            + "\n  ".join(failures)
            + "\n\nNothing was written for them. Fix the paths and re-run;"
              "\ndo not quote older numbers from output/ as if they were fresh."
        )

    print(f"\nDone. See ./{OUTPUT_DIR}/<model>/ for each model's charts + tables.")


if __name__ == "__main__":
    main()
