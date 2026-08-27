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
  Needs `ultralytics`, the model file, and the dataset. The accessibility
  model validates against accessibility_dataset/ (local, ~1 min). The COCO
  model is OPT-IN via INCLUDE_COCO and uses metrics/coco_val_only.yaml — do
  NOT use Ultralytics' bundled coco.yaml, whose download block pulls ~27 GB
  including the unlabeled test2017 set that cannot be validated against.

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

# Validate the COCO model too? The accessibility model is the one that matters
# for the report and takes ~1 minute on 655 local images. COCO val2017 is 5,000
# images and takes 15-30 min. Set this False to get the accessibility numbers
# immediately and come back to COCO later.
INCLUDE_COCO = False

# Input sizes to validate the accessibility model at, so the speed win from
# dropping to 320 can be reported with its accuracy cost attached.
# Set to [] to skip the sweep.
IMGSZ_SWEEP = [320, 448, 640]

# NEVER point this at Ultralytics' bundled "coco.yaml": its download block
# fetches ~27 GB (train2017 19 GB + val2017 1 GB + test2017 7 GB), and
# test2017 is the UNLABELED competition set which cannot be validated against.
# coco_val_only.yaml has no download block and uses only val2017.
COCO_DATA = os.path.join("metrics", "coco_val_only.yaml")
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
        # val2017 — a proper HELD-OUT set, and ONLY val2017 (see COCO_DATA).
        #
        # This was "coco128.yaml", which is the first 128 images of COCO
        # *train2017* — images these weights were TRAINED on. Those numbers
        # (mAP50 0.607 / 0.671) are train-set scores: inflated, and not a valid
        # basis for comparing models. Do not put them in a report.
        "data": COCO_DATA,
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
def count_val_instances(cfg: dict, class_names: list) -> dict:
    """Count annotated boxes per class in the validation split.

    Reads the dataset yaml, resolves its val images dir, swaps /images/ for
    /labels/, and tallies the leading class index of every line. Returns
    {class_name: count}; falls back to an empty dict if anything is missing, so
    a counting problem can never take down the whole report.
    """
    try:
        import yaml
    except ImportError:
        return {}
    try:
        with open(cfg["data"], encoding="utf-8") as f:
            spec = yaml.safe_load(f)
    except Exception:
        return {}

    root = spec.get("path") or os.path.dirname(os.path.abspath(cfg["data"]))
    val = str(spec.get("val", ""))
    val_dir = val if os.path.isabs(val) else os.path.join(root, val)
    if not os.path.isdir(val_dir):
        return {}

    label_dir = val_dir.replace(os.sep + "images", os.sep + "labels")
    label_dir = label_dir.replace("/images", "/labels")
    if not os.path.isdir(label_dir):
        return {}

    counts = {}
    for fname in os.listdir(label_dir):
        if not fname.endswith(".txt"):
            continue
        try:
            with open(os.path.join(label_dir, fname), encoding="utf-8") as f:
                for line in f:
                    parts = line.split()
                    if not parts:
                        continue
                    idx = int(parts[0])
                    if 0 <= idx < len(class_names):
                        counts[class_names[idx]] = counts.get(
                            class_names[idx], 0) + 1
        except (OSError, ValueError):
            continue
    return counts


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

    ordered = [names[i] for i in sorted(names)]
    per_class = {}
    for i in sorted(names):
        per_class[names[i]] = float(maps[i]) if i in ap_idx else None

    # Instance counts: box.nt_per_class does not exist on every Ultralytics
    # version, and when it is missing the old code silently reported "0
    # instances" for every class — which reads in a report as "this class was
    # never evaluated". Counting the label files is version-proof and exact.
    instances = count_val_instances(cfg, ordered)

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
    with open(path, "w", newline="", encoding="utf-8") as f:
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
    with open(path, "w", encoding="utf-8") as f:
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


# ─────────────────────────────────────────────────────────────────────────────
def resolution_sweep():
    """Validate the shipping accessibility model at several input sizes.

    The app now runs at 320x320 instead of 640x640, which made it 4.8x faster
    on-device. This measures what that cost in accuracy, so the speed claim can
    be stated with its price attached instead of on its own.

    Watch STAIR recall specifically. mAP averages across classes and hides the
    one failure that matters -- a missed stair is a fall.
    """
    from ultralytics import YOLO

    cfg = MODELS[0]  # the accessibility model
    if not os.path.exists(cfg["model"]):
        print(f"!! sweep skipped: {cfg['model']} not found")
        return

    out_dir = os.path.join(OUTPUT_DIR, "accessibility")
    os.makedirs(out_dir, exist_ok=True)
    rows = []

    for size in IMGSZ_SWEEP:
        print(f"\n[sweep] validating at {size}x{size} ...")
        try:
            m = YOLO(cfg["model"]).val(
                data=cfg["data"], imgsz=size, plots=False, verbose=False)
        except Exception as e:
            print(f"   FAILED at {size}: {e}")
            continue

        names = m.names
        ap_idx = list(m.box.ap_class_index)
        per_map, per_recall = {}, {}
        for i in sorted(names):
            per_map[names[i]] = float(m.box.maps[i]) if i in ap_idx else None
            # box.r is indexed by POSITION IN ap_class_index, not by class id.
            if i in ap_idx:
                try:
                    per_recall[names[i]] = float(m.box.r[ap_idx.index(i)])
                except (IndexError, TypeError):
                    per_recall[names[i]] = None
            else:
                per_recall[names[i]] = None

        rows.append({
            "imgsz": size,
            "precision": round(float(m.box.mp), 4),
            "recall": round(float(m.box.mr), 4),
            "mAP50": round(float(m.box.map50), 4),
            "mAP50_95": round(float(m.box.map), 4),
            **{f"{n}_mAP50_95": (round(v, 4) if v is not None else "")
               for n, v in per_map.items()},
            # Per-class RECALL is the safety metric: it is the fraction of real
            # stairs the model actually found. mAP averages precision across
            # thresholds and classes and hides exactly this.
            **{f"{n}_recall": (round(v, 4) if v is not None else "")
               for n, v in per_recall.items()},
        })

    if not rows:
        print("!! sweep produced nothing")
        return

    path = os.path.join(out_dir, "resolution_tradeoff.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print("\n" + "=" * 70)
    print("ACCURACY vs INPUT RESOLUTION  (accessibility model)")
    print("=" * 70)
    hdr = list(rows[0].keys())
    print(" | ".join(f"{h:>14}" for h in hdr))
    print("-" * (17 * len(hdr)))
    for r in rows:
        print(" | ".join(f"{str(r[h]):>14}" for h in hdr))

    base = next((r for r in rows if r["imgsz"] == 640), None)
    ship = next((r for r in rows if r["imgsz"] == 320), None)
    if base and ship:
        print(f"""
WHAT 320 COSTS versus 640:
  mAP50-95 : {base['mAP50_95']:.4f} -> {ship['mAP50_95']:.4f}   """
              f"""({(ship['mAP50_95'] - base['mAP50_95']):+.4f})
  recall   : {base['recall']:.4f} -> {ship['recall']:.4f}   """
              f"""({(ship['recall'] - base['recall']):+.4f})""")
        for sk, lbl in (("Stair_mAP50_95", "STAIR mAP"),
                        ("Stair_recall", "STAIR RECALL")):
            if sk in base and sk in ship and base[sk] != "" and ship[sk] != "":
                mark = "   <- THE safety metric" if "recall" in sk else ""
                print(f"  {lbl:<12}: {base[sk]:.4f} -> {ship[sk]:.4f}   "
                      f"({(ship[sk] - base[sk]):+.4f}){mark}")
        print(f"""
  Measured on-device speed at these two sizes (Galaxy S23 Ultra, profile):
    640 -> 2951 ms/frame (0.4 FPS)
    320 ->  619 ms/frame (1.6 FPS)   = 4.8x faster
  That is the trade. Quote both halves together, never the speed alone.""")
    print(f"\nSaved -> {path}")


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"[mode] {'LIVE validation' if RUN_VALIDATION else 'OFFLINE (embedded)'}")

    failures = []
    models = MODELS if INCLUDE_COCO else [
        m for m in MODELS if m["name"] != "coco"]
    if not INCLUDE_COCO:
        print("[skip] COCO validation disabled (INCLUDE_COCO = False)")
    for cfg in models:
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

    if IMGSZ_SWEEP:
        resolution_sweep()

    print(f"\nDone. See ./{OUTPUT_DIR}/<model>/ for each model's charts + tables.")


if __name__ == "__main__":
    main()
