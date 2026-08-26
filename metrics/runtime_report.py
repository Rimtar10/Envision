#!/usr/bin/env python3
"""
runtime_report.py
=================

Turns the app's live performance log into graphs + stats — the on-device
counterpart to accuracy_report.py.

How the data is produced
------------------------
The Dart detector prints one CSV line per processed frame, tagged `PERF_CSV`:

    PERF_CSV,timestamp_ms,inference_ms,fps,detections

Capture a session while running the app:

    flutter run | tee perf_log.txt        # macOS / Linux
    flutter run | Tee-Object perf_log.txt # Windows PowerShell

(You can also use `adb logcat | findstr PERF_CSV > perf_log.txt`.)

Then:

    pip install matplotlib numpy
    python runtime_report.py perf_log.txt

On Windows PowerShell, use `python` or `py -3` instead of the Unix-style
`/usr/bin/env python3` launcher.

Outputs (into ./runtime_output/):
  - latency_over_time.png    per-frame processing time across the session
  - fps_over_time.png        rolling FPS across the session
  - latency_histogram.png    distribution of per-frame latency
  - runtime_summary.txt      mean / median / p95 latency, mean FPS, etc.
  - runtime_samples.csv      cleaned per-frame samples
"""

from __future__ import annotations

import csv
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUTPUT_DIR = os.path.join("metrics", "runtime_output")
PRIMARY = "#2563eb"
ACCENT = "#16a34a"


def read_log_lines(path: str):
    """Read a runtime log using the common encodings produced by PowerShell and Flutter."""
    for encoding in ("utf-8-sig", "utf-16", "cp1252"):
        try:
            with open(path, "r", encoding=encoding) as f:
                lines = f.readlines()
            if any("PERF_CSV" in line for line in lines):
                return lines
        except UnicodeError:
            continue
    with open(path, "r", errors="ignore") as f:
        return f.readlines()


# Per-stage column order emitted by Detector._updatePerf, after the original
# four fields. Absent in logs captured before stage profiling was added.
STAGES = ["convert", "rotate", "letterbox", "infer", "parse"]

STAGE_BLURB = {
    "convert":   "YUV/NV21 -> RGB       (Dart, ImageUtils)",
    "rotate":    "rotate to display     (Dart, copyRotate)",
    "letterbox": "letterbox + normalise (Dart, Letterbox.fill)",
    "infer":     "interpreter.run       (native TFLite)",
    "parse":     "decode boxes + NMS    (Dart, _parseYoloOutput)",
}


def parse_log(path: str):
    """Extract PERF_CSV rows from a console/logcat dump.

    Handles both formats:
      old (5 fields):  PERF_CSV,timestamp_ms,inference_ms,fps,detections
      new (10 fields): ...,convert_ms,rotate_ms,letterbox_ms,infer_ms,parse_ms

    Returns (ts, lat, fps, det, stages); `stages` is {} for an old-format log.
    """
    ts, lat, fps, det = [], [], [], []
    stage_rows = []
    for line in read_log_lines(path):
        if "PERF_CSV" not in line:
            continue
        frag = line[line.index("PERF_CSV"):].strip()
        parts = frag.split(",")
        if len(parts) < 5:
            continue
        if parts[1] == "timestamp_ms":
            continue
        try:
            ts.append(int(parts[1]))
            lat.append(float(parts[2]))
            fps.append(float(parts[3]))
            det.append(int(parts[4]))
        except ValueError:
            continue
        if len(parts) >= 10:
            try:
                stage_rows.append([float(x) for x in parts[5:10]])
            except ValueError:
                stage_rows.append([0.0] * 5)
        else:
            stage_rows.append(None)

    stages = {}
    if stage_rows and all(r is not None for r in stage_rows):
        arr = np.array(stage_rows)
        stages = {name: arr[:, i] for i, name in enumerate(STAGES)}

    return np.array(ts), np.array(lat), np.array(fps), np.array(det), stages


def line_plot(x, y, title, ylabel, color, path, ref=None, ref_label=None):
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(x, y, color=color, linewidth=1.2)
    if ref is not None:
        ax.axhline(ref, color="#ef4444", linestyle="--", linewidth=1,
                   label=ref_label)
        ax.legend()
    ax.set_xlabel("seconds since start")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def histogram(values, title, xlabel, path):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(values, bins=30, color=PRIMARY, alpha=0.85)
    p50, p95 = np.percentile(values, [50, 95])
    ax.axvline(p50, color=ACCENT, linestyle="--", linewidth=1.2,
               label=f"median {p50:.0f} ms")
    ax.axvline(p95, color="#ef4444", linestyle="--", linewidth=1.2,
               label=f"p95 {p95:.0f} ms")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("frames")
    ax.set_title(title)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def stage_breakdown(stages, path):
    """Mean ms per pipeline stage.

    This is the chart that answers the only question worth asking before any
    optimisation work: is the frame budget going to the models, or to moving
    pixels around in Dart before they ever reach a model?
    """
    means = {k: float(np.mean(v)) for k, v in stages.items()}
    pre = means["convert"] + means["rotate"] + means["letterbox"]
    model = means["infer"]
    post = means["parse"]

    labels = STAGES
    values = [means[k] for k in labels]
    # The three preprocessing stages share a colour so the group reads as one.
    colors = [PRIMARY, PRIMARY, PRIMARY, ACCENT, "#9ca3af"]

    fig, ax = plt.subplots(figsize=(8, 4.2))
    bars = ax.barh(labels[::-1], values[::-1], color=colors[::-1])
    total = sum(values)
    for b, v in zip(bars, values[::-1]):
        pct = (v / total * 100) if total else 0
        ax.text(v + total * 0.012, b.get_y() + b.get_height() / 2,
                f"{v:.1f} ms  ({pct:.0f}%)", va="center", fontsize=9)
    ax.set_xlim(0, total * 1.28 if total else 1)
    ax.set_xlabel("mean milliseconds per frame")
    ax.set_title("Where the frame budget goes")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return pre, model, post


def write_summary(lat, fps, det, path, stages=None):
    lines = [
        "ENVISION — LIVE RUNTIME PERFORMANCE",
        "=" * 38,
        "",
        f"Frames analysed     : {len(lat)}",
        "",
        "Per-frame latency (ms):",
        f"  mean    : {lat.mean():.1f}",
        f"  median  : {np.median(lat):.1f}",
        f"  p95     : {np.percentile(lat, 95):.1f}",
        f"  min/max : {lat.min():.1f} / {lat.max():.1f}",
        "",
        "Throughput (FPS):",
        f"  mean    : {fps.mean():.1f}",
        f"  median  : {np.median(fps):.1f}",
        "",
        f"Avg objects / frame : {det.mean():.1f}",
    ]

    if stages:
        means = {k: float(np.mean(v)) for k, v in stages.items()}
        pre = means["convert"] + means["rotate"] + means["letterbox"]
        model = means["infer"]
        post = means["parse"]
        total = pre + model + post

        lines += ["", "Per-stage mean (ms):"]
        for k in STAGES:
            share = (means[k] / total * 100) if total else 0
            lines.append(
                f"  {k:<10} {means[k]:7.2f}  {share:5.1f}%   {STAGE_BLURB[k]}")
        lines += [
            "",
            f"  preprocessing (convert+rotate+letterbox) : {pre:7.2f} ms"
            f"  {(pre / total * 100) if total else 0:5.1f}%",
            f"  model inference                          : {model:7.2f} ms"
            f"  {(model / total * 100) if total else 0:5.1f}%",
            f"  post-processing                          : {post:7.2f} ms"
            f"  {(post / total * 100) if total else 0:5.1f}%",
            "",
            "VERDICT:",
        ]
        if total and pre > model:
            factor = pre / model if model else float("inf")
            lines += [
                f"  PREPROCESSING IS THE BOTTLENECK ({factor:.1f}x the model time).",
                "  Moving convert/rotate/letterbox off pure Dart is worth more",
                "  than any model swap. A published benchmark on this same",
                "  pipeline puts native OpenCV at 5.4 ms against 82.5 ms for",
                "  the pure-Dart `image` package.",
                "",
                "  Next steps, in order:",
                "   1. flutter test test/letterbox_parity_test.dart, then set",
                "      TensorflowHelper.useFusedRotationLetterbox = true --",
                "      that deletes the rotate stage outright.",
                "   2. Migrate tflite_flutter -> flutter_litert and move",
                "      convert/letterbox to opencv_dart.",
            ]
        elif total:
            lines += [
                "  The models dominate; preprocessing is not the problem.",
                "  Next: w8a32 quantisation, GPU delegate, and drop the",
                "  accessibility model to 448 input.",
                "  Guard Stair recall through every one of those changes.",
            ]
    else:
        lines += [
            "",
            "No per-stage data in this log -- it was captured before stage",
            "profiling was added. Re-capture with the current build to find",
            "out where the time actually goes.",
        ]

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def write_csv(ts, lat, fps, det, path, stages=None):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        header = ["timestamp_ms", "inference_ms", "fps", "detections"]
        if stages:
            header += [f"{k}_ms" for k in STAGES]
        w.writerow(header)
        for i, row in enumerate(zip(ts, lat, fps, det)):
            if stages:
                row = tuple(row) + tuple(
                    round(float(stages[k][i]), 2) for k in STAGES)
            w.writerow(row)


def main():
    if len(sys.argv) < 2:
        print("usage: python runtime_report.py <perf_log.txt>")
        sys.exit(1)
    log_path = sys.argv[1]
    if not os.path.exists(log_path):
        print(f"error: {log_path} not found")
        sys.exit(1)

    ts, lat, fps, det, stages = parse_log(log_path)
    if len(lat) == 0:
        print("error: no PERF_CSV samples found in the log.")
        print("Make sure you captured `flutter run` output while the camera ran.")
        sys.exit(1)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    t0 = ts.min()
    secs = (ts - t0) / 1000.0

    line_plot(secs, lat, "Per-frame processing latency", "latency (ms)",
              PRIMARY, os.path.join(OUTPUT_DIR, "latency_over_time.png"),
              ref=lat.mean(), ref_label=f"mean {lat.mean():.0f} ms")
    line_plot(secs, fps, "Throughput over time", "FPS",
              ACCENT, os.path.join(OUTPUT_DIR, "fps_over_time.png"),
              ref=fps.mean(), ref_label=f"mean {fps.mean():.1f} FPS")
    histogram(lat, "Latency distribution", "latency (ms)",
              os.path.join(OUTPUT_DIR, "latency_histogram.png"))

    pre = model = post = None
    if stages:
        pre, model, post = stage_breakdown(
            stages, os.path.join(OUTPUT_DIR, "stage_breakdown.png"))

    write_summary(lat, fps, det,
                  os.path.join(OUTPUT_DIR, "runtime_summary.txt"), stages)
    write_csv(ts, lat, fps, det,
              os.path.join(OUTPUT_DIR, "runtime_samples.csv"), stages)

    print(f"Parsed {len(lat)} frames.")
    print(f"  mean latency : {lat.mean():.1f} ms")
    print(f"  mean FPS     : {fps.mean():.1f}")
    if stages:
        total = pre + model + post
        pct = lambda v: (v / total * 100) if total else 0
        print(f"  preprocessing: {pre:6.1f} ms  ({pct(pre):.0f}%)")
        print(f"  model        : {model:6.1f} ms  ({pct(model):.0f}%)")
        print(f"  postprocess  : {post:6.1f} ms  ({pct(post):.0f}%)")
        if pre > model:
            print("  --> PREPROCESSING IS THE BOTTLENECK. "
                  "See runtime_summary.txt")
    else:
        print("  (no per-stage data -- log predates stage profiling)")
    print(f"Files written to ./{OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
