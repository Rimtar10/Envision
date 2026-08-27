#!/usr/bin/env python3
"""
benchmark_all.py
================
ONE command that measures every model this project uses and draws the figures
for the report.

    python metrics/benchmark_all.py

WHAT IT DOES
------------
Re-validates each model live, on the correct dataset, with a stated protocol,
and writes figures + a machine-readable CSV + a README explaining each figure.

THE INTEGRITY RULE THIS SCRIPT ENFORCES
---------------------------------------
Models are only ever plotted against models validated on the SAME dataset with
the SAME protocol. That sounds obvious; it is exactly what went wrong before.
The old cnn_vs_transformer chart put YOLO numbers measured on coco128 in the
same bars as an RT-DETR number measured on the doors/stairs set, with a speed
column mixing GPU and CPU timings. Every number in it was real and the chart
was still meaningless.

So: comparisons live in GROUPS. A group names its dataset once, and every model
in it is measured against that dataset. Nothing crosses a group boundary. If a
model cannot be measured, the script says so loudly and omits it, rather than
falling back to a stale number from an earlier run.

WHAT IT WILL NOT DO
-------------------
- Read accuracy numbers out of old CSVs. Everything is re-measured.
- Chart a model whose weights it cannot find.
- Put two datasets in one figure.
"""

from __future__ import annotations

import csv
import glob
import json
import os
import re
import sys
from datetime import datetime

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG — edit these, nothing else
# ─────────────────────────────────────────────────────────────────────────────

# Named on every figure. Latency claims are meaningless without it.
DEVICE_NAME = "Galaxy S23 Ultra (SM-S918B, Snapdragon 8 Gen 2)"
BUILD_MODE = "Flutter profile build, TFLite CPU, 4 threads"

OUT = os.path.join("metrics", "output", "benchmark")

# Input sizes to sweep the accessibility model over.
IMGSZ_SWEEP = [320, 448, 640]

# The size the app actually ships. Highlighted in the figures.
SHIPPING_IMGSZ = 320

# ── Comparison groups ────────────────────────────────────────────────────────
# Each group = one dataset + the models measured against it. Nothing is ever
# compared across groups.
GROUPS = [
    {
        "id": "accessibility",
        "title": "Accessibility detector (Door / Stair / Window)",
        "data": "accessibility_dataset/accessibility.yaml",
        "models": [
            ("accessibility_v2 (YOLO11s)",
             "runs/detect/runs/accessibility_v2/weights/best.pt"),
        ],
        "sweep": True,          # also sweep IMGSZ_SWEEP for this group
    },
    {
        "id": "coco_backbones",
        "title": "COCO backbones (80 classes)",
        # NOT Ultralytics' bundled coco.yaml -- that downloads ~27 GB including
        # the unlabeled test2017 split. coco_val_only.yaml points at val2017
        # already on disk and has no download block.
        "data": os.path.join("metrics", "coco_val_only.yaml"),
        "models": [
            ("YOLOv8n", "yolov8n.pt"),
            ("YOLO11n", "yolo11n.pt"),
            ("YOLO11s", "yolo11s.pt"),
            ("YOLO26n", "yolo26n.pt"),
        ],
        "sweep": False,
    },
    {
        "id": "unified",
        "title": "Unified 21-class model (ablation)",
        "data": "unified_dataset/unified.yaml",
        "models": [
            ("unified_envision (YOLO11s, 21 cls)",
             "runs/detect/runs/unified_envision-3/weights/best.pt"),
        ],
        "sweep": False,
    },
]

# On-device capture logs. Label -> file. Add rows as you capture more.
# Any perf_log*.txt not listed here is auto-discovered and labelled by filename.
PERF_LOGS = {
    "640 input": "perf_log_gpu_delegate.txt",
    "320 input": "perf_log.txt",
}

# ─────────────────────────────────────────────────────────────────────────────
# PALETTE — validated with the dataviz palette validator.
#   3 slots, all-pairs:  CVD dE 9.2, normal-vision dE 24.0  -> PASS
#   5 slots, adjacent:   CVD dE 9.1, normal-vision dE 19.6  -> PASS
# Contrast warns below 3:1, so every chart ships visible direct labels.
# ─────────────────────────────────────────────────────────────────────────────
BLUE, ORANGE, AQUA, YELLOW, MAGENTA = (
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4")

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#8a8a85"
GRID = "#e3e3df"

# Door / Stair / Window keep the SAME colour in every figure. Colour follows the
# entity, never its rank -- so a reader can carry identity between charts.
CLASS_COLOR = {"Door": BLUE, "Stair": ORANGE, "Window": AQUA}

# Pipeline stages, fixed order, fixed colours.
STAGES = ["convert", "rotate", "letterbox", "infer", "parse"]
STAGE_COLOR = {
    "convert": BLUE, "rotate": ORANGE, "letterbox": AQUA,
    "infer": YELLOW, "parse": MAGENTA,
}
STAGE_LABEL = {
    "convert": "YUV→RGB",
    "rotate": "rotate",
    "letterbox": "letterbox",
    "infer": "inference",
    "parse": "decode + NMS",
}


def style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID,
        "axes.labelcolor": INK_2,
        "axes.titlecolor": INK,
        "text.color": INK,
        "xtick.color": INK_2,
        "ytick.color": INK_2,
        "grid.color": GRID,
        "font.size": 10,
        "axes.titlesize": 12.5,
        "axes.titleweight": "600",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 150,
    })


def finish(ax, fig, path, note=None):
    """Recessive grid, provenance footer, save."""
    ax.grid(axis="y", alpha=0.5, linewidth=0.7)
    ax.set_axisbelow(True)
    footer = note or ""
    fig.text(0.01, 0.005, footer, fontsize=7, color=MUTED, ha="left", va="bottom")
    fig.tight_layout(rect=[0, 0.035, 1, 1])
    fig.savefig(path)
    plt.close(fig)
    print(f"   -> {path}")


# ─────────────────────────────────────────────────────────────────────────────
# MEASUREMENT
# ─────────────────────────────────────────────────────────────────────────────
def measure(weights: str, data: str, imgsz: int) -> dict | None:
    """Validate one model at one input size. Returns None on any failure."""
    from ultralytics import YOLO

    try:
        m = YOLO(weights).val(data=data, imgsz=imgsz, plots=False, verbose=False)
    except Exception as e:
        print(f"   !! FAILED ({e})")
        return None

    names = m.names
    ap_idx = list(m.box.ap_class_index)
    per = {}
    for i in sorted(names):
        if i not in ap_idx:
            per[names[i]] = None
            continue
        j = ap_idx.index(i)
        try:
            per[names[i]] = {
                "precision": float(m.box.p[j]),
                "recall": float(m.box.r[j]),
                "mAP50": float(m.box.ap50[j]),
                "mAP50_95": float(m.box.maps[i]),
            }
        except (IndexError, TypeError):
            per[names[i]] = None

    params = None
    try:
        params = sum(p.numel() for p in YOLO(weights).model.parameters()) / 1e6
    except Exception:
        pass

    return {
        "imgsz": imgsz,
        "params_M": round(params, 2) if params else None,
        "precision": float(m.box.mp),
        "recall": float(m.box.mr),
        "mAP50": float(m.box.map50),
        "mAP50_95": float(m.box.map),
        "val_ms_per_image": float(m.speed.get("inference", 0.0)),
        "per_class": per,
        # Label counts live on the DetMetrics object in Ultralytics 8.3, NOT on
        # m.box -- reading m.box.nt_per_class silently yields nothing, which is
        # why F7 was skipped and F2 printed "? instances".
        "instances": _instances(m, names),
    }


def _instances(m, names):
    """{class name: labelled instances in the val split}, or {} if unavailable."""
    nt = getattr(m, "nt_per_class", None)
    if nt is None:
        nt = getattr(m.box, "nt_per_class", None)
    if nt is None:
        return {}
    try:
        return {names[i]: int(nt[i]) for i in sorted(names) if i < len(nt)}
    except (TypeError, ValueError, IndexError):
        return {}


def run_groups():
    results = {}
    for g in GROUPS:
        print(f"\n=== {g['title']} ===")
        print(f"    dataset: {g['data']}")
        if not os.path.exists(g["data"]):
            print("    !! dataset yaml not found -- GROUP SKIPPED")
            results[g["id"]] = {"skipped": "dataset yaml missing", "runs": []}
            continue

        runs = []
        for label, weights in g["models"]:
            if not os.path.exists(weights):
                print(f"  -- {label}: weights not found at {weights} -- SKIPPED")
                continue
            sizes = IMGSZ_SWEEP if g.get("sweep") else [SHIPPING_IMGSZ if
                                                        g["id"] == "x" else 640]
            for size in sizes:
                print(f"  .. {label} @ {size}")
                r = measure(weights, g["data"], size)
                if r:
                    r["model"] = label
                    r["weights"] = weights
                    runs.append(r)
        results[g["id"]] = {"title": g["title"], "data": g["data"], "runs": runs}
    return results


# ─────────────────────────────────────────────────────────────────────────────
# ON-DEVICE LOGS
# ─────────────────────────────────────────────────────────────────────────────
def read_log(path):
    raw = open(path, "rb").read()
    for enc in ("utf-16", "utf-8-sig", "utf-8", "cp1252"):
        try:
            txt = raw.decode(enc)
            if "PERF_CSV" in txt:
                return txt
        except Exception:
            continue
    return ""


def parse_perf(path):
    txt = read_log(path)
    rows, geom = [], []
    for line in txt.splitlines():
        if "MODEL_GEOMETRY" in line:
            geom.append(line.split("MODEL_GEOMETRY,", 1)[-1].strip())
        if "PERF_CSV" not in line:
            continue
        p = line[line.index("PERF_CSV"):].split(",")
        if len(p) < 10 or p[1] == "timestamp_ms":
            continue
        try:
            rows.append({
                "total": float(p[2]), "fps": float(p[3]), "det": int(float(p[4])),
                **{s: float(p[5 + i]) for i, s in enumerate(STAGES)},
            })
        except ValueError:
            continue
    return rows, geom


def collect_perf():
    found = dict(PERF_LOGS)
    for f in sorted(glob.glob("perf_log*.txt")):
        if f not in found.values():
            found[os.path.splitext(os.path.basename(f))[0]] = f

    out = {}
    for label, path in found.items():
        if not os.path.exists(path):
            print(f"  -- perf log '{label}': {path} not found -- SKIPPED")
            continue
        rows, geom = parse_perf(path)
        if not rows:
            print(f"  -- perf log '{label}': no usable PERF_CSV rows -- SKIPPED")
            continue
        out[label] = {"rows": rows, "geometry": geom, "path": path}
        print(f"  .. perf log '{label}': {len(rows)} frames from {path}")
    return out


# ─────────────────────────────────────────────────────────────────────────────
# FIGURES
# ─────────────────────────────────────────────────────────────────────────────
def fig_per_class(runs, path):
    """F1 — per-class accuracy of the SHIPPING configuration."""
    run = next((r for r in runs if r["imgsz"] == SHIPPING_IMGSZ), None)
    if not run:
        return False
    classes = [c for c in run["per_class"] if run["per_class"][c]]
    if not classes:
        return False

    metrics = [("precision", "Precision"), ("recall", "Recall"),
               ("mAP50", "mAP@50"), ("mAP50_95", "mAP@50-95")]
    x = np.arange(len(metrics))
    w = 0.8 / len(classes)

    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    for i, c in enumerate(classes):
        vals = [run["per_class"][c][k] for k, _ in metrics]
        off = (i - (len(classes) - 1) / 2) * w
        # 2px surface gap between adjacent bars
        bars = ax.bar(x + off, vals, w * 0.9, label=c,
                      color=CLASS_COLOR.get(c, BLUE), edgecolor=SURFACE,
                      linewidth=1.4)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.015, f"{v:.2f}",
                    ha="center", va="bottom", fontsize=8.5, color=INK_2)

    ax.set_xticks(x)
    ax.set_xticklabels([lbl for _, lbl in metrics])
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("score")
    ax.set_title(f"Per-class accuracy — shipping model @ {SHIPPING_IMGSZ}px")
    ax.legend(frameon=False, ncol=len(classes), loc="upper right")
    n = run.get("instances", {})
    note = (f"accessibility_dataset val split · "
            + " · ".join(f"{c}: {n.get(c, '?')} instances" for c in classes))
    finish(ax, fig, path, note)
    return True


def fig_recall_vs_res(runs, path):
    """F2 — the trade-off curve, on RECALL (the safety metric)."""
    runs = sorted([r for r in runs if r["imgsz"] in IMGSZ_SWEEP],
                  key=lambda r: r["imgsz"])
    if len(runs) < 2:
        return False
    sizes = [r["imgsz"] for r in runs]
    classes = [c for c in runs[0]["per_class"] if runs[0]["per_class"][c]]

    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    for c in classes:
        ys = [r["per_class"][c]["recall"] for r in runs]
        ax.plot(sizes, ys, marker="o", markersize=8, linewidth=2,
                color=CLASS_COLOR.get(c, BLUE), label=c,
                markeredgecolor=SURFACE, markeredgewidth=1.6)
        # direct label at the right end -- identity never by colour alone
        ax.annotate(f" {c}", (sizes[-1], ys[-1]), fontsize=9.5,
                    color=CLASS_COLOR.get(c, BLUE), va="center", weight="600")
        # Selective labels: the shipping size and the baseline it is judged
        # against. A number on every point is noise, not information.
        for sx, sy in zip(sizes, ys):
            if sx not in (SHIPPING_IMGSZ, sizes[-1]):
                continue
            ax.text(sx, sy + 0.014, f"{sy:.3f}", ha="center", fontsize=8.5,
                    color=INK_2, weight="600")

    # Headroom BEFORE the guide line, so labels never reach the title.
    allv = [r["per_class"][c]["recall"] for r in runs for c in classes]
    span = max(allv) - min(allv)
    ax.set_ylim(min(allv) - span * 0.14, max(allv) + span * 0.22)

    ax.axvline(SHIPPING_IMGSZ, color=MUTED, linestyle=":", linewidth=1.2,
               zorder=0)
    ax.text(SHIPPING_IMGSZ, ax.get_ylim()[0], " shipping", fontsize=8,
            color=MUTED, va="bottom")

    ax.set_xticks(sizes)
    ax.set_xticklabels([f"{s}px" for s in sizes])
    ax.set_xlim(sizes[0] - 30, sizes[-1] + 90)
    ax.set_ylabel("recall  (fraction of real objects found)")
    ax.set_title("Recall vs input resolution — per class")
    ax.legend(frameon=False, loc="lower right")
    finish(ax, fig, path,
           "Recall, not mAP: for a navigation aid a MISS is the dangerous "
           "failure. accessibility_dataset val split.")
    return True


def fig_speed_vs_safety(runs, perf, path):
    """F3 — the money chart: on-device latency against Stair recall."""
    dev = {}
    for label, d in perf.items():
        m = re.search(r"(\d{3,4})", label)
        if m:
            dev[int(m.group(1))] = float(np.mean([r["total"] for r in d["rows"]]))
    if not dev:
        return False

    pts = []
    for r in sorted(runs, key=lambda r: r["imgsz"]):
        if r["imgsz"] in dev and r["per_class"].get("Stair"):
            pts.append((dev[r["imgsz"]], r["per_class"]["Stair"]["recall"],
                        r["imgsz"]))
    if len(pts) < 2:
        return False

    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    xs, ys, labels = zip(*pts)
    ax.plot(xs, ys, linewidth=2, color=MUTED, alpha=0.5, zorder=1)
    ymid = (min(ys) + max(ys)) / 2
    for x, y, s in pts:
        c = ORANGE if s == SHIPPING_IMGSZ else BLUE
        ax.scatter([x], [y], s=190, color=c, zorder=3,
                   edgecolor=SURFACE, linewidth=2.2)
        # Offset in POINTS, away from the marker and away from the line: text
        # above when the point sits low, below when it sits high.
        dy = 16 if y <= ymid else -16
        va = "bottom" if dy > 0 else "top"
        ax.annotate(f"{s}px\n{x:.0f} ms/frame · recall {y:.3f}",
                    (x, y), textcoords="offset points", xytext=(0, dy),
                    ha="center", va=va, fontsize=9, color=INK_2,
                    linespacing=1.45)

    ax.set_xlabel(f"on-device latency, ms per frame  ({DEVICE_NAME})")
    ax.set_ylabel("Stair recall")
    ax.set_title("Speed vs safety — lower-left is worse on both axes")
    pad = (max(ys) - min(ys)) or 0.02
    ax.set_ylim(min(ys) - pad * 1.5, max(ys) + pad * 1.5)
    ax.set_xlim(-max(xs) * 0.08, max(xs) * 1.20)
    finish(ax, fig, path, BUILD_MODE + " · orange = shipping configuration")
    return True


def fig_backbones(runs, path, title, dataset_note):
    """F4 — models measured on ONE dataset with ONE protocol."""
    runs = [r for r in runs if r]
    if len(runs) < 2:
        return False
    runs = sorted(runs, key=lambda r: r["mAP50_95"])
    names = [r["model"] for r in runs]
    y = np.arange(len(runs))

    fig, ax = plt.subplots(figsize=(8.4, 0.72 * len(runs) + 2.4))
    bars = ax.barh(y, [r["mAP50_95"] for r in runs], 0.62, color=BLUE,
                   edgecolor=SURFACE, linewidth=1.4)
    for b, r in zip(bars, runs):
        extra = f"   {r['params_M']}M params" if r.get("params_M") else ""
        ax.text(r["mAP50_95"] + 0.006,
                b.get_y() + b.get_height() / 2,
                f"{r['mAP50_95']:.3f}{extra}",
                va="center", fontsize=9, color=INK_2)
    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.set_xlim(0, max(r["mAP50_95"] for r in runs) * 1.45)
    ax.set_xlabel("mAP@50-95")
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.5, linewidth=0.7)
    ax.set_axisbelow(True)
    fig.text(0.01, 0.005, dataset_note, fontsize=7, color=MUTED)
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    fig.savefig(path)
    plt.close(fig)
    print(f"   -> {path}")
    return True


def fig_frame_budget(perf, path):
    """F5 — where the frame budget goes, per capture."""
    if not perf:
        return False
    labels = list(perf.keys())
    fig, ax = plt.subplots(figsize=(8.6, 0.9 * len(labels) + 2.6))
    y = np.arange(len(labels))

    # Row totals first: the in-bar label threshold must be a share of the ROW,
    # not of whatever has been stacked so far.
    means = {st: np.array([np.mean([r[st] for r in perf[l]["rows"]])
                           for l in labels]) for st in STAGES}
    totals = np.sum([means[st] for st in STAGES], axis=0)

    left = np.zeros(len(labels))
    for st in STAGES:
        vals = means[st]
        ax.barh(y, vals, 0.55, left=left, color=STAGE_COLOR[st],
                label=STAGE_LABEL[st], edgecolor=SURFACE, linewidth=1.8)
        for i, v in enumerate(vals):
            # Only label a segment wide enough to hold the text. Slivers get
            # their value from the legend and on_device.csv instead of an
            # unreadable smear -- and a sliver IS the finding here.
            if v >= totals[i] * 0.10:
                ax.text(left[i] + v / 2, y[i], f"{v:.0f} ms", ha="center",
                        va="center", fontsize=9, color="white", weight="600")
        left += vals

    for i, tot in enumerate(left):
        ax.text(tot * 1.015, y[i], f"  {tot:.0f} ms   ({1000/tot:.1f} FPS)",
                va="center", fontsize=9.5, color=INK, weight="600")

    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlim(0, left.max() * 1.32)
    ax.set_xlabel("mean milliseconds per frame")
    ax.set_title("Where the frame budget goes", pad=30)
    # Legend ABOVE the axes -- inside it lands on top of the longest bar.
    ax.legend(frameon=False, ncol=len(STAGES), fontsize=8.5,
              loc="lower left", bbox_to_anchor=(0, 1.008))
    ax.grid(axis="x", alpha=0.5, linewidth=0.7)
    ax.set_axisbelow(True)
    fig.text(0.01, 0.005, f"{DEVICE_NAME} · {BUILD_MODE}", fontsize=7,
             color=MUTED)
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    fig.savefig(path)
    plt.close(fig)
    print(f"   -> {path}")
    return True


def fig_latency_spread(perf, path):
    """F6 — the tail, not the mean. p95 is what a safety claim rests on."""
    if not perf:
        return False
    labels = list(perf.keys())
    data = [[r["total"] for r in perf[l]["rows"]] for l in labels]

    fig, ax = plt.subplots(figsize=(8.0, 0.85 * len(labels) + 2.6))
    bp = ax.boxplot(data, vert=False, widths=0.5, patch_artist=True,
                    showfliers=False,
                    medianprops=dict(color=INK, linewidth=2))
    for i, box in enumerate(bp["boxes"]):
        box.set(facecolor=[BLUE, ORANGE, AQUA][i % 3], alpha=0.85,
                edgecolor=SURFACE, linewidth=1.4)
    for i, vals in enumerate(data):
        p95 = float(np.percentile(vals, 95))
        ax.scatter([p95], [i + 1], marker="D", s=48, color=INK, zorder=4,
                   edgecolor=SURFACE, linewidth=1.2)
        ax.text(p95, i + 1.33, f"p95 {p95:.0f} ms", fontsize=8.5,
                color=INK_2, ha="center")

    ax.set_yticklabels(labels)
    ax.set_xlabel("milliseconds per frame")
    ax.set_title("Per-frame latency distribution (box = IQR, diamond = p95)")
    ax.grid(axis="x", alpha=0.5, linewidth=0.7)
    ax.set_axisbelow(True)
    fig.text(0.01, 0.005,
             f"{DEVICE_NAME} · a safety claim rests on the tail, not the mean",
             fontsize=7, color=MUTED)
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    fig.savefig(path)
    plt.close(fig)
    print(f"   -> {path}")
    return True


def fig_dataset_balance(runs, path):
    """F7 — how much evidence backs each class."""
    run = next((r for r in runs if r.get("instances")), None)
    if not run or not run["instances"]:
        return False
    items = sorted(run["instances"].items(), key=lambda kv: kv[1])
    names = [k for k, _ in items]
    vals = [v for _, v in items]

    fig, ax = plt.subplots(figsize=(7.6, 0.62 * len(names) + 2.4))
    y = np.arange(len(names))
    bars = ax.barh(y, vals, 0.6,
                   color=[CLASS_COLOR.get(n, BLUE) for n in names],
                   edgecolor=SURFACE, linewidth=1.4)
    for b, v in zip(bars, vals):
        ax.text(v + max(vals) * 0.012, b.get_y() + b.get_height() / 2, f"{v:,}",
                va="center", fontsize=9, color=INK_2)
    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.set_xlim(0, max(vals) * 1.22)
    ax.set_xlabel("labelled instances in the validation split")
    ax.set_title("Evidence behind each class")
    ax.grid(axis="x", alpha=0.5, linewidth=0.7)
    ax.set_axisbelow(True)
    fig.text(0.01, 0.005,
             "A per-class score is only as trustworthy as the count behind it.",
             fontsize=7, color=MUTED)
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    fig.savefig(path)
    plt.close(fig)
    print(f"   -> {path}")
    return True


# ─────────────────────────────────────────────────────────────────────────────
def write_tables(results, perf, out):
    rows = []
    for gid, g in results.items():
        for r in g.get("runs", []):
            base = {
                "group": gid, "model": r["model"], "imgsz": r["imgsz"],
                "params_M": r.get("params_M"),
                "precision": round(r["precision"], 4),
                "recall": round(r["recall"], 4),
                "mAP50": round(r["mAP50"], 4),
                "mAP50_95": round(r["mAP50_95"], 4),
                "desktop_val_ms_per_image": round(r["val_ms_per_image"], 2),
            }
            for c, m in r["per_class"].items():
                if m:
                    base[f"{c}_recall"] = round(m["recall"], 4)
                    base[f"{c}_mAP50_95"] = round(m["mAP50_95"], 4)
            rows.append(base)

    if rows:
        keys = sorted({k for r in rows for k in r})
        keys = (["group", "model", "imgsz"] +
                [k for k in keys if k not in ("group", "model", "imgsz")])
        p = os.path.join(out, "all_models.csv")
        with open(p, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
        print(f"   -> {p}")

    if perf:
        p = os.path.join(out, "on_device.csv")
        with open(p, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["capture", "frames", "mean_ms", "median_ms", "p95_ms",
                        "mean_fps"] + [f"{s}_ms" for s in STAGES])
            for label, d in perf.items():
                t = [r["total"] for r in d["rows"]]
                w.writerow([
                    label, len(t), round(float(np.mean(t)), 1),
                    round(float(np.median(t)), 1),
                    round(float(np.percentile(t, 95)), 1),
                    round(1000 / float(np.mean(t)), 2),
                ] + [round(float(np.mean([r[s] for r in d["rows"]])), 2)
                     for s in STAGES])
        print(f"   -> {p}")


README = """FIGURES — what each one shows, and on what data
================================================================
Generated {when}
Device for all latency figures: {device}
Build: {build}

Every accuracy number here was re-measured by this script. Nothing was read
from a previous run's CSV.

THE RULE: models are only compared against models validated on the SAME
dataset with the SAME protocol. Figures never mix datasets. If a model is
absent from a figure, its weights or its dataset could not be found -- the
script omits it rather than substituting an older number.

F1_per_class_accuracy.png
    Precision / recall / mAP for each class of the SHIPPING model at {ship}px.
    Dataset: accessibility_dataset val split.

F2_recall_vs_resolution.png
    Recall per class at each input size. Recall, not mAP, because for a
    navigation aid a MISS is the dangerous failure -- mAP averages that away.

F3_speed_vs_safety.png
    On-device ms/frame against Stair recall. The single most important figure:
    it shows what the speed was bought with. Latency is measured on the device
    named above; recall on the validation split.

F4_coco_backbones.png
    COCO backbones on held-out val2017. NOT coco128 -- coco128 is a slice of
    COCO *train*, so scoring pretrained weights on it measures memorisation.

F5_frame_budget.png
    Mean ms per pipeline stage, per capture. Shows whether time goes to the
    models or to moving pixels around before they reach a model.

F6_latency_spread.png
    Distribution with p95 marked. A safety claim rests on the tail: the mean
    frame is not the one that misses the staircase.

F7_dataset_balance.png
    Labelled instances per class. A per-class score is only as trustworthy as
    the count behind it.

all_models.csv    every measurement, machine-readable
on_device.csv     per-capture latency summary incl. p95

WHAT IS DELIBERATELY NOT HERE
-----------------------------
RT-DETR is not charted against the YOLO models. It was trained on a 5-class
label space (Door, Gate, Stair, Window, Building) while the shipping detector
uses 3, and its run used different epochs, image size and data fraction. Those
numbers cannot share an axis with these. To compare them, train a matched
baseline under identical settings -- metrics/train_rtdetr.py now does that.
"""


def main():
    style()
    os.makedirs(OUT, exist_ok=True)
    print(f"[out] {OUT}")

    print("\n### measuring models")
    results = run_groups()

    print("\n### reading on-device captures")
    perf = collect_perf()

    print("\n### figures")
    acc = results.get("accessibility", {}).get("runs", [])
    made = []

    if acc:
        if fig_per_class(acc, os.path.join(OUT, "F1_per_class_accuracy.png")):
            made.append("F1")
        if fig_recall_vs_res(acc, os.path.join(OUT, "F2_recall_vs_resolution.png")):
            made.append("F2")
        if fig_speed_vs_safety(acc, perf,
                               os.path.join(OUT, "F3_speed_vs_safety.png")):
            made.append("F3")
        if fig_dataset_balance(acc, os.path.join(OUT, "F7_dataset_balance.png")):
            made.append("F7")
    else:
        print("   -- accessibility group empty; F1/F2/F3/F7 skipped")

    coco = results.get("coco_backbones", {})
    if coco.get("runs"):
        if fig_backbones(coco["runs"], os.path.join(OUT, "F4_coco_backbones.png"),
                         "COCO backbones — held-out val2017",
                         f"dataset: {coco['data']} · all models, one protocol"):
            made.append("F4")
    else:
        print("   -- COCO group empty; F4 skipped "
              "(check metrics/coco_val_only.yaml paths)")

    if perf:
        if fig_frame_budget(perf, os.path.join(OUT, "F5_frame_budget.png")):
            made.append("F5")
        if fig_latency_spread(perf, os.path.join(OUT, "F6_latency_spread.png")):
            made.append("F6")
    else:
        print("   -- no on-device captures; F5/F6 skipped")

    print("\n### tables")
    write_tables(results, perf, OUT)

    with open(os.path.join(OUT, "README.txt"), "w", encoding="utf-8") as f:
        f.write(README.format(
            when=datetime.now().strftime("%Y-%m-%d %H:%M"),
            device=DEVICE_NAME, build=BUILD_MODE, ship=SHIPPING_IMGSZ))
    print(f"   -> {os.path.join(OUT, 'README.txt')}")

    print(f"\nDone. Figures produced: {', '.join(made) if made else 'NONE'}")
    print(f"See {OUT}/README.txt for what each figure shows.")


if __name__ == "__main__":
    main()
