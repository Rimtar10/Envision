#!/usr/bin/env python3
"""
train_rtdetr.py
===============
End-to-end RT-DETR (transformer detector) experiment for Envision, on the SAME
doors/windows/stairs dataset the YOLOv8n accessibility model was trained on.

It reproduces your 4 Roboflow data cells (download -> keep 5 classes -> download
stairs -> merge stairs at index 2), then fine-tunes RT-DETR, validates it, writes
the results next to your YOLO numbers, and draws a CNN-vs-transformer bar chart.

Run (Colab with a GPU runtime is strongly recommended):
    pip install ultralytics roboflow
    python train_rtdetr.py

You do NOT need the COCO dataset: rtdetr-l.pt is already COCO-pretrained and is
only fine-tuned on your doors/windows/stairs data here.
"""

import os, shutil, csv, yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ── SETTINGS (tune these) ─────────────────────────────────────────────────────
EPOCHS   = 10          # limited-compute run on a 4GB laptop GPU
IMGSZ    = 512          # smaller -> fits in 4GB VRAM (avoids slow shared-memory spill)
BATCH    = 2           # if GPU_mem still >4G & slow, set 1
FRACTION = 0.5         # use half the data to keep the run feasible; set 1.0 for full
# Key is read from metrics/roboflow_key.txt (gitignored) or the
# ROBOFLOW_API_KEY env var. It used to be hard-coded on this line.
import sys as _sys
_sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _secrets import roboflow_api_key
OUT      = os.path.join("metrics", "output", "rtdetr")
KEEP     = ['Door', 'Gate', 'Stair', 'Window', 'Building']

# ── MATCHED CNN BASELINE ──────────────────────────────────────────────────────
# This script used to hard-code YOLO numbers copied from compare_coco_models.py
# and put them in the same table as RT-DETR. That comparison was invalid three
# times over:
#   1. DIFFERENT DATA   — the YOLO rows were measured on coco128, the RT-DETR
#                         row on this doors/stairs set. Different label spaces.
#   2. DIFFERENT BUDGET — RT-DETR got 10 epochs at 512px on 50% of the data;
#                         the YOLO weights had full COCO pretraining behind them.
#   3. DIFFERENT HARDWARE — the infer_ms column mixed a GPU timing with CPU
#                         timings, making the transformer look ~6x FASTER than
#                         the CNNs, which is the opposite of the truth on a phone.
#
# The baseline is now TRAINED HERE, on the same yaml, for the same epochs, at
# the same imgsz, with the same data fraction, and timed in the same process.
# That is the only way the bar chart means anything.
CNN_BASELINES = ["yolo11n.pt", "yolo11s.pt"]
BLUE, GREEN, RED = "#2563eb", "#16a34a", "#dc2626"


def prepare_dataset():
    """Cells 1-4: download doors dataset, keep 5 classes, download+merge stairs."""
    from roboflow import Roboflow
    rf = Roboflow(api_key=roboflow_api_key())

    print("[data] downloading doors-dataset-updated ...")
    doors = rf.workspace("rims-workspace-spb1p") \
              .project("doors-dataset-updated-fqlay-q7qtb").version(1).download("yolov8")
    loc = doors.location

    print("[data] remapping to 5 classes ...")
    with open(f"{loc}/data.yaml") as f:
        cfg = yaml.safe_load(f)
    old = cfg['names']
    keep_idx = {old.index(c): i for i, c in enumerate(KEEP) if c in old}
    for split in ['train', 'valid', 'test']:
        lp = f"{loc}/{split}/labels"
        if not os.path.isdir(lp):
            continue
        for lf in os.listdir(lp):
            fp = os.path.join(lp, lf); out_lines = []
            with open(fp) as f:
                for line in f:
                    p = line.split()
                    if p and int(p[0]) in keep_idx:
                        p[0] = str(keep_idx[int(p[0])]); out_lines.append(' '.join(p))
            with open(fp, 'w') as f:
                f.write('\n'.join(out_lines))
    cfg['names'] = KEEP; cfg['nc'] = len(KEEP)
    with open(f"{loc}/data.yaml", 'w') as f:
        yaml.dump(cfg, f)

    print("[data] downloading stairs dataset ...")
    stairs = rf.workspace("yolo-datasets-f9og9") \
               .project("stairs_detection-9av4i-lyswf-orcim-duw1m").version(3).download("yolov8")

    print("[data] merging stairs at index 2 ...")
    with open(f"{stairs.location}/data.yaml") as f:
        s_cfg = yaml.safe_load(f)
    src_idx = next((i for i, n in enumerate(s_cfg['names']) if 'stair' in n.lower()), 0)
    TARGET = 2
    for split in ['train', 'valid', 'test']:
        si = f"{stairs.location}/{split}/images"; sl = f"{stairs.location}/{split}/labels"
        di = f"{loc}/{split}/images";            dl = f"{loc}/{split}/labels"
        if not os.path.isdir(si):
            continue
        os.makedirs(di, exist_ok=True); os.makedirs(dl, exist_ok=True)
        for img in os.listdir(si):
            shutil.copy(os.path.join(si, img), os.path.join(di, f"stairs_{img}"))
        for lb in os.listdir(sl):
            out_lines = []
            with open(os.path.join(sl, lb)) as f:
                for line in f:
                    p = line.split()
                    if p and int(p[0]) == src_idx:
                        p[0] = str(TARGET); out_lines.append(' '.join(p))
            with open(os.path.join(dl, f"stairs_{lb}"), 'w') as f:
                f.write('\n'.join(out_lines))
    print(f"[data] ready -> {loc}/data.yaml")
    return f"{loc}/data.yaml"


def plot_comparison(rows, path):
    """Grouped bar chart of mAP50 / mAP50-95 for all models; RT-DETR highlighted."""
    labels = [r["model"] for r in rows]
    m50    = [r["mAP50"] for r in rows]
    m5095  = [r["mAP50_95"] for r in rows]
    x = np.arange(len(labels)); w = 0.38
    fig, ax = plt.subplots(figsize=(8, 4.8))
    b1 = ax.bar(x - w/2, m50,   w, label="mAP@50",    color=BLUE)
    b2 = ax.bar(x + w/2, m5095, w, label="mAP@50-95", color=GREEN)
    # highlight the transformer (last row)
    b1[-1].set_color(RED); b2[-1].set_color("#f59e0b")
    for bars in (b1, b2):
        for b in bars:
            ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.01,
                    f"{b.get_height():.3f}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim(0, 1.0); ax.set_ylabel("score")
    ax.set_title("CNN vs Transformer detectors on the Envision dataset")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def _measure(model, data_yaml, display_name):
    """Validate one already-trained model and return a normalized row."""
    m = model.val(data=data_yaml, plots=True)
    params = sum(p.numel() for p in model.model.parameters()) / 1e6
    return {
        "model": display_name,
        "params_M": round(params, 1),
        "mAP50": round(float(m.box.map50), 3),
        "mAP50_95": round(float(m.box.map), 3),
        "precision": round(float(m.box.mp), 3),
        "recall": round(float(m.box.mr), 3),
        # Ultralytics validation speed on THIS machine. Identical conditions for
        # every row, but still not a phone: see the note printed at the end.
        "infer_ms": round(float(m.speed["inference"]), 1),
    }, m


def main():
    from ultralytics import RTDETR, YOLO
    os.makedirs(OUT, exist_ok=True)
    data_yaml = prepare_dataset()

    rows = []

    # ── CNN baselines, trained under EXACTLY the RT-DETR budget ──────────────
    for base in CNN_BASELINES:
        tag = base.replace(".pt", "")
        print(f"\n[train] {tag}: epochs={EPOCHS} imgsz={IMGSZ} "
              f"batch={BATCH} fraction={FRACTION}")
        cnn = YOLO(base)
        cnn.train(data=data_yaml, epochs=EPOCHS, imgsz=IMGSZ, batch=BATCH,
                  name=f"matched_{tag}", project="runs", fraction=FRACTION)
        print(f"[val] {tag} ...")
        row, _ = _measure(cnn, data_yaml, f"{tag} (CNN)")
        rows.append(row)

    # ── Transformer, same budget ─────────────────────────────────────────────
    print(f"\n[train] RT-DETR-L: epochs={EPOCHS} imgsz={IMGSZ} "
          f"batch={BATCH} fraction={FRACTION}")
    model = RTDETR("rtdetr-l.pt")
    model.train(data=data_yaml, epochs=EPOCHS, imgsz=IMGSZ, batch=BATCH,
                name="rtdetr_accessibility", project="runs", fraction=FRACTION)

    print("\n[val] validating RT-DETR ...")
    rt, m = _measure(model, data_yaml, "RT-DETR-L (transformer)")
    rows.append(rt)

    fields = list(rt)

    with open(os.path.join(OUT, "rtdetr_metrics.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerow(rt)
    with open(os.path.join(OUT, "cnn_vs_transformer.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
        for r in rows: w.writerow(r)

    plot_comparison(rows, os.path.join(OUT, "cnn_vs_transformer.png"))

    try:
        for fn in os.listdir(m.save_dir):
            if fn.endswith((".png", ".jpg")):
                shutil.copy(os.path.join(m.save_dir, fn), os.path.join(OUT, fn))
    except Exception:
        pass

    print("\n=== RESULTS (all rows: same data, same epochs, same imgsz) ===")
    header = " | ".join(f"{k:>22}" for k in fields)
    print(header)
    print("-" * len(header))
    for r in rows:
        print(" | ".join(f"{str(r[k]):>22}" for k in fields))

    print(f"\nSaved -> {OUT}/")
    print(f"""
CAVEATS TO CARRY INTO THE REPORT
  * {EPOCHS} epochs at {IMGSZ}px on {int(FRACTION * 100)}% of the data is a
    LIMITED-COMPUTE run. Every model here is undertrained; treat the ordering
    as indicative, not as a ceiling for either architecture.
  * infer_ms is Ultralytics validation speed on THIS machine, not on a phone.
    RT-DETR's published "real-time" numbers are GPU numbers. The app runs
    TFLite on a mobile CPU, where a 32M-parameter transformer is not viable
    regardless of how this table looks.
  * The only on-device numbers that count come from metrics/runtime_report.py.
""")


if __name__ == "__main__":
    main()
