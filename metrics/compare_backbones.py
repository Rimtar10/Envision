#!/usr/bin/env python3
"""
compare_backbones.py
====================
Which backbone is actually best at detecting doors and stairs?

    python metrics/compare_backbones.py

WHY THIS EXISTS
---------------
The shipping accessibility detector is YOLO11s, and nobody ever checked whether
it needed to be. YOLO11s is 9.4M parameters and carries most of the ~567 ms
per-frame inference cost measured on device. If a nano backbone gets within a
point or two on Door/Stair/Window, that is another large speed win with
evidence behind it rather than a guess.

Comparing the STOCK pretrained models on COCO does not answer this. Those
weights have never seen a door. The only way to answer it is to fine-tune each
backbone on the same data, with the same settings, and measure them the same
way. That is all this script does.

WHAT IT TOUCHES
---------------
Nothing the app builds from. It writes only into runs/ and
metrics/output/backbones/. No Dart, no pubspec, no app_constants.

PROTOCOL (identical for every backbone -- this is the whole point)
------------------------------------------------------------------
  data     accessibility_dataset/accessibility.yaml   (Door / Stair / Window)
  epochs   120          matched to the existing YOLO11s run
  imgsz    640          matched to the existing YOLO11s run
  seed     0            matched; deterministic=True
  patience 30           matched

  batch    the largest that fits each model (see BATCH below). The existing
           YOLO11s run used batch 2 because 11s at 640 would not fit a 4GB
           card at anything larger. Ultralytics normalises the learning rate
           to a nominal batch of 64, so this is a documented, compensated
           difference -- not a silent one. It is the one setting that is not
           bit-identical across models, and it is recorded in the output CSV.

TIME
----
Roughly 1.5-2.5 h per nano backbone on an RTX 3050 -- so about 5-7 h for the
three, unattended. It is RESUME-SAFE: a backbone whose run already exists is
skipped, so if it dies overnight just run it again.

The existing YOLO11s run is REUSED, not retrained.
"""

from __future__ import annotations

import csv
import glob
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
DATA = "accessibility_dataset/accessibility.yaml"
OUT = os.path.join("metrics", "output", "backbones")
PROJECT = "runs"

EPOCHS = 120
IMGSZ_TRAIN = 640
PATIENCE = 30
SEED = 0

# Input sizes every finished model is validated at.
IMGSZ_EVAL = [320, 448, 640]
SHIPPING_IMGSZ = 320

# (label, weights to fine-tune from, run name, batch)
# Drop a row to skip that backbone.
BACKBONES = [
    ("YOLO11n", "yolo11n.pt", "bb_yolo11n", 8),
    ("YOLOv8n", "yolov8n.pt", "bb_yolov8n", 8),
    ("YOLO26n", "yolo26n.pt", "bb_yolo26n", 8),
]

# Already trained -- reused, never retrained.
INCUMBENT = ("YOLO11s (shipping)",
             "runs/detect/runs/accessibility_v2/weights/best.pt", 2)

# ── palette (validated: all-pairs CVD dE 9.2, normal-vision dE 24.0) ─────────
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
SURFACE, INK, INK_2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8a85", "#e3e3df"
CLASS_COLOR = {"Door": BLUE, "Stair": ORANGE, "Window": AQUA}


def style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE, "axes.edgecolor": GRID,
        "axes.labelcolor": INK_2, "axes.titlecolor": INK, "text.color": INK,
        "xtick.color": INK_2, "ytick.color": INK_2, "grid.color": GRID,
        "font.size": 10, "axes.titlesize": 12.5, "axes.titleweight": "600",
        "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150,
    })


# ─────────────────────────────────────────────────────────────────────────────
def existing_best(run_name: str) -> str | None:
    hits = glob.glob(f"{PROJECT}/**/{run_name}/weights/best.pt", recursive=True)
    return max(hits, key=os.path.getmtime) if hits else None


def run_epochs(weights: str) -> int:
    """How many epochs the run behind these weights actually completed.

    A run that died overnight still leaves a best.pt, and reusing it silently
    would break the one claim this whole script rests on -- that every
    backbone got the same training. So we count the rows in results.csv and
    say so out loud.
    """
    csv_path = os.path.join(os.path.dirname(os.path.dirname(weights)),
                            "results.csv")
    if not os.path.exists(csv_path):
        return -1
    try:
        with open(csv_path, encoding="utf-8", errors="ignore") as f:
            return max(0, sum(1 for _ in f) - 1)
    except OSError:
        return -1


def warn_if_short(label, weights):
    n = run_epochs(weights)
    if n < 0:
        print(f"     .. {label}: no results.csv -- epoch count unknown")
    elif n < EPOCHS - PATIENCE:
        print(f"     !! {label}: only {n}/{EPOCHS} epochs completed. Either it "
              f"was interrupted or it early-stopped. Delete the run and "
              f"re-run to make the comparison fair.")
    elif n < EPOCHS:
        print(f"     .. {label}: {n}/{EPOCHS} epochs (early stop, patience "
              f"{PATIENCE}) -- fine")
    return n


def train_one(label, base, run_name, batch):
    """Fine-tune one backbone. Returns path to best.pt, or None."""
    from ultralytics import YOLO

    done = existing_best(run_name)
    if done:
        print(f"  == {label}: already trained ({done}) -- SKIPPED")
        return done

    if not os.path.exists(base):
        print(f"  !! {label}: base weights {base} not found -- SKIPPED")
        return None

    print(f"\n  >> training {label} from {base}  "
          f"(epochs={EPOCHS} imgsz={IMGSZ_TRAIN} batch={batch} seed={SEED})")
    try:
        YOLO(base).train(
            data=DATA, epochs=EPOCHS, imgsz=IMGSZ_TRAIN, batch=batch,
            patience=PATIENCE, seed=SEED, deterministic=True,
            name=run_name, project=PROJECT, verbose=False,
        )
    except Exception as e:
        print(f"  !! {label}: training failed -- {e}")
        return None

    got = existing_best(run_name)
    if not got:
        print(f"  !! {label}: training finished but no best.pt found")
    return got


def evaluate(label, weights, batch_used):
    """Validate one finished model at every eval resolution."""
    from ultralytics import YOLO

    rows = []
    for size in IMGSZ_EVAL:
        try:
            m = YOLO(weights).val(data=DATA, imgsz=size, plots=False,
                                  verbose=False)
        except Exception as e:
            print(f"  !! {label} @ {size}: {e}")
            continue

        names, ap_idx = m.names, list(m.box.ap_class_index)
        row = {
            "backbone": label, "weights": weights, "imgsz": size,
            "train_batch": batch_used,
            "params_M": round(sum(p.numel() for p in YOLO(weights).model.parameters())
                              / 1e6, 2),
            "precision": round(float(m.box.mp), 4),
            "recall": round(float(m.box.mr), 4),
            "mAP50": round(float(m.box.map50), 4),
            "mAP50_95": round(float(m.box.map), 4),
            "desktop_val_ms": round(float(m.speed.get("inference", 0.0)), 2),
            "epochs_done": run_epochs(weights),
        }
        for i in sorted(names):
            if i in ap_idx:
                j = ap_idx.index(i)
                row[f"{names[i]}_recall"] = round(float(m.box.r[j]), 4)
                row[f"{names[i]}_mAP50_95"] = round(float(m.box.maps[i]), 4)
        rows.append(row)
        print(f"     {label} @ {size}: mAP50-95 {row['mAP50_95']}  "
              f"Stair recall {row.get('Stair_recall', '?')}")
    return rows


# ─────────────────────────────────────────────────────────────────────────────
def fig_stair_recall(rows, path):
    """The safety metric, per backbone, at the shipping resolution."""
    sel = [r for r in rows
           if r["imgsz"] == SHIPPING_IMGSZ and "Stair_recall" in r]
    if len(sel) < 2:
        return False
    sel.sort(key=lambda r: r["Stair_recall"])

    names = [r["backbone"] for r in sel]
    vals = [r["Stair_recall"] for r in sel]
    y = np.arange(len(sel))

    fig, ax = plt.subplots(figsize=(8.6, 0.72 * len(sel) + 2.6))
    colors = [ORANGE if "shipping" in n else BLUE for n in names]
    bars = ax.barh(y, vals, 0.6, color=colors, edgecolor=SURFACE, linewidth=1.6)
    for b, r in zip(bars, sel):
        ax.text(r["Stair_recall"] + max(vals) * 0.015,
                b.get_y() + b.get_height() / 2,
                f"{r['Stair_recall']:.3f}    {r['params_M']:.2f}M params",
                va="center", fontsize=9.5, color=INK_2)
    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.set_xlim(0, max(vals) * 1.45)
    ax.set_xlabel("Stair recall  (fraction of real staircases found)")
    ax.set_title(f"Which backbone finds the most stairs?  @ {SHIPPING_IMGSZ}px")
    ax.grid(axis="x", alpha=0.5, linewidth=0.7)
    ax.set_axisbelow(True)
    fig.text(0.012, 0.018,
             f"Identical protocol: {DATA} · {EPOCHS} epochs · train imgsz "
             f"{IMGSZ_TRAIN} · seed {SEED} · orange = current shipping model",
             fontsize=7, color=MUTED)
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    fig.savefig(path)
    plt.close(fig)
    print(f"   -> {path}")
    return True


# A hairline leader from each marker to its label, so attribution stays
# unambiguous when the solver has to push a label sideways.
LEADER = dict(arrowstyle="-", color=MUTED, linewidth=0.7,
              shrinkA=3, shrinkB=9)


def _place_labels(fig, ax, points, texts, marker_pt=15.0):
    """Greedy, collision-avoiding point labels.

    points  [(x, y), ...] in data coords
    texts   matching label strings

    Tries a ring of candidate offsets per label and keeps the first that
    overlaps neither an already-placed label nor any marker; when a point is
    boxed in on every side it keeps the LEAST-overlapping candidate rather
    than falling back to a fixed one. Without this a cluster of similar models
    (which is exactly what nano backbones are) prints its labels on top of
    each other.
    """
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()

    # Every marker becomes a keep-out box in display coords.
    blockers = []
    for x, y in points:
        px, py = ax.transData.transform((x, y))
        r = marker_pt * fig.dpi / 72.0 / 2.0 + 2
        blockers.append((px - r, py - r, px + r, py + r))

    # Labels must also stay inside the plot area, or they collide with the
    # tick labels and get clipped at the figure edge.
    ab = ax.get_window_extent(renderer=rend)

    def penalty(box):
        """0 means the slot is clean; larger means worse."""
        x0, y0, x1, y1 = box
        out = (max(0.0, ab.x0 - x0) + max(0.0, x1 - ab.x1)
               + max(0.0, ab.y0 - y0) + max(0.0, y1 - ab.y1))
        p = out * 400.0  # leaving the plot area is worse than any overlap
        for bx0, by0, bx1, by1 in blockers:
            ox = min(x1, bx1) - max(x0, bx0)
            oy = min(y1, by1) - max(y0, by0)
            if ox > 0 and oy > 0:
                p += ox * oy
        return p

    # (dx, dy, ha, va) in points: near ring, then a wider ring, so a boxed-in
    # point can escape outwards instead of landing on its neighbour.
    cands = []
    for d in (17, 30, 46):
        cands += [(0, d, "center", "bottom"), (0, -d, "center", "top"),
                  (d, 0, "left", "center"), (-d, 0, "right", "center"),
                  (d * 0.8, d * 0.8, "left", "bottom"),
                  (-d * 0.8, d * 0.8, "right", "bottom"),
                  (d * 0.8, -d * 0.8, "left", "top"),
                  (-d * 0.8, -d * 0.8, "right", "top")]

    # Label the most cramped points first: they have the fewest escape routes.
    order = sorted(range(len(points)),
                   key=lambda i: min((abs(points[i][0] - points[j][0]) for j
                                      in range(len(points)) if j != i),
                                     default=0.0))

    for i in order:
        best = None  # (penalty, annotation, box)
        for dx, dy, ha, va in cands:
            ann = ax.annotate(texts[i], points[i], textcoords="offset points",
                              xytext=(dx, dy), ha=ha, va=va, fontsize=9,
                              color=INK_2, linespacing=1.4, zorder=4,
                              arrowprops=LEADER)
            bb = ann.get_window_extent(renderer=rend)
            box = (bb.x0 - 3, bb.y0 - 2, bb.x1 + 3, bb.y1 + 2)
            p = penalty(box)
            if p <= 0:
                best = (0.0, ann, box)
                break
            if best is None or p < best[0]:
                if best is not None:
                    best[1].remove()
                best = (p, ann, box)
            else:
                ann.remove()
        blockers.append(best[2])


def fig_accuracy_vs_size(rows, path):
    """mAP against parameter count -- is the extra capacity buying anything?"""
    sel = [r for r in rows if r["imgsz"] == SHIPPING_IMGSZ]
    if len(sel) < 2:
        return False

    fig, ax = plt.subplots(figsize=(8.6, 5.2))
    for r in sel:
        c = ORANGE if "shipping" in r["backbone"] else BLUE
        ax.scatter([r["params_M"]], [r["mAP50_95"]], s=190, color=c, zorder=3,
                   edgecolor=SURFACE, linewidth=2.2)
    ys = [r["mAP50_95"] for r in sel]
    xs = [r["params_M"] for r in sel]

    ax.set_xlabel("parameters (millions) — proxy for on-device inference cost")
    ax.set_ylabel("mAP@50-95")
    ax.set_title("Does the bigger backbone earn its cost?")
    padx = (max(xs) - min(xs)) or max(xs) or 1
    pady = (max(ys) - min(ys)) or 0.02
    # Parameters cannot be negative, so never let the axis run below zero.
    ax.set_xlim(max(0.0, min(xs) - padx * 0.30), max(xs) + padx * 0.30)
    ax.set_ylim(min(ys) - pady * 0.75, max(ys) + pady * 0.75)
    ax.grid(alpha=0.5, linewidth=0.7)
    ax.set_axisbelow(True)

    fig.text(0.012, 0.015,
             f"Fine-tuned on {DATA}, identical protocol. Validated @ "
             f"{SHIPPING_IMGSZ}px, the resolution the app ships.",
             fontsize=7, color=MUTED)
    # Lock the geometry BEFORE solving label placement -- the solver works in
    # display coordinates, so moving the axes afterwards would invalidate it.
    fig.subplots_adjust(left=0.10, right=0.97, top=0.91, bottom=0.16)
    _place_labels(fig, ax,
                  [(r["params_M"], r["mAP50_95"]) for r in sel],
                  [f"{r['backbone']}\n{r['mAP50_95']:.3f}" for r in sel])
    fig.savefig(path)
    plt.close(fig)
    print(f"   -> {path}")
    return True


def fig_per_class(rows, path):
    """Per-class recall, every backbone, at the shipping resolution."""
    sel = [r for r in rows if r["imgsz"] == SHIPPING_IMGSZ]
    if len(sel) < 2:
        return False
    classes = [c for c in ("Door", "Stair", "Window")
               if f"{c}_recall" in sel[0]]
    if not classes:
        return False

    sel.sort(key=lambda r: r["params_M"])
    names = [r["backbone"] for r in sel]
    x = np.arange(len(names))
    w = 0.8 / len(classes)

    fig, ax = plt.subplots(figsize=(9.0, 4.9))
    for i, c in enumerate(classes):
        vals = [r.get(f"{c}_recall", 0) for r in sel]
        off = (i - (len(classes) - 1) / 2) * w
        bars = ax.bar(x + off, vals, w * 0.9, label=c, color=CLASS_COLOR[c],
                      edgecolor=SURFACE, linewidth=1.5)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.012, f"{v:.2f}",
                    ha="center", va="bottom", fontsize=8.5, color=INK_2)

    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("recall")
    ax.set_title(f"Per-class recall by backbone  @ {SHIPPING_IMGSZ}px",
                 pad=26)
    # Legend above the axes: inside, it collides with any bar close to 1.0.
    ax.legend(frameon=False, ncol=len(classes), loc="lower left",
              bbox_to_anchor=(0, 1.005), handlelength=1.2, columnspacing=1.6)
    ax.grid(axis="y", alpha=0.5, linewidth=0.7)
    ax.set_axisbelow(True)
    fig.text(0.012, 0.018,
             "Recall, not mAP: a miss is the dangerous failure for a "
             "navigation aid.", fontsize=7, color=MUTED)
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    fig.savefig(path)
    plt.close(fig)
    print(f"   -> {path}")
    return True


# ─────────────────────────────────────────────────────────────────────────────
def main():
    style()
    os.makedirs(OUT, exist_ok=True)

    if not os.path.exists(DATA):
        raise SystemExit(f"dataset yaml not found: {DATA}")

    print("=" * 70)
    print("BACKBONE COMPARISON on Door / Stair / Window")
    print(f"  {EPOCHS} epochs · imgsz {IMGSZ_TRAIN} · seed {SEED} · "
          f"patience {PATIENCE}")
    print("  Resume-safe: an already-trained backbone is skipped.")
    print("=" * 70)

    finished = []

    inc_label, inc_weights, inc_batch = INCUMBENT
    if os.path.exists(inc_weights):
        print(f"\n  == {inc_label}: reusing existing run (not retrained)")
        warn_if_short(inc_label, inc_weights)
        finished.append((inc_label, inc_weights, inc_batch))
    else:
        print(f"\n  !! incumbent weights not found at {inc_weights}")

    print("\n### training")
    for label, base, run_name, batch in BACKBONES:
        got = train_one(label, base, run_name, batch)
        if got:
            warn_if_short(label, got)
            finished.append((label, got, batch))

    if len(finished) < 2:
        raise SystemExit("\nFewer than two models available -- nothing to "
                         "compare. Check the messages above.")

    print("\n### evaluating (identical protocol for every model)")
    rows = []
    for label, weights, batch in finished:
        rows += evaluate(label, weights, batch)

    if not rows:
        raise SystemExit("No evaluations succeeded.")

    keys = sorted({k for r in rows for k in r})
    keys = (["backbone", "imgsz", "params_M"] +
            [k for k in keys if k not in ("backbone", "imgsz", "params_M")])
    csv_path = os.path.join(OUT, "backbone_comparison.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"\n   -> {csv_path}")

    print("\n### figures")
    fig_stair_recall(rows, os.path.join(OUT, "B1_stair_recall.png"))
    fig_accuracy_vs_size(rows, os.path.join(OUT, "B2_accuracy_vs_params.png"))
    fig_per_class(rows, os.path.join(OUT, "B3_per_class_recall.png"))

    # ── verdict ──────────────────────────────────────────────────────────────
    ship = [r for r in rows if r["imgsz"] == SHIPPING_IMGSZ
            and "Stair_recall" in r]
    if ship:
        ship.sort(key=lambda r: -r["Stair_recall"])
        inc = next((r for r in ship if "shipping" in r["backbone"]), None)
        best = ship[0]
        print("\n" + "=" * 70)
        print(f"STAIR RECALL @ {SHIPPING_IMGSZ}px  (the safety metric)")
        print("=" * 70)
        for r in ship:
            print(f"  {r['backbone']:<22} {r['Stair_recall']:.4f}   "
                  f"{r['params_M']:>5.2f}M params   "
                  f"mAP50-95 {r['mAP50_95']:.4f}")
        if inc and best["backbone"] != inc["backbone"]:
            shrink = inc["params_M"] / best["params_M"]
            print(f"""
  {best['backbone']} beats the shipping model on the metric that matters,
  with {shrink:.1f}x fewer parameters. On-device inference is roughly
  proportional to that, so this is a speed win AND an accuracy win.
  Verify on-device before switching: export it, swap app_constants, rebuild,
  and re-capture PERF_CSV.""")
        elif inc:
            gap = inc["Stair_recall"] - best["Stair_recall"] if best is not inc \
                else 0.0
            print(f"""
  The shipping YOLO11s is still the best on Stair recall (by {gap:.4f}).
  It has earned its parameters. Worth checking whether a nano backbone is
  CLOSE enough that the speed is worth the trade -- see B1 and B2.""")
    print(f"\nAll outputs in {OUT}/")


if __name__ == "__main__":
    main()
