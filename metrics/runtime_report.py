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

OUTPUT_DIR = "runtime_output"
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


def parse_log(path: str):
    """Extract PERF_CSV rows from a console/logcat dump. Robust to log prefixes."""
    ts, lat, fps, det = [], [], [], []
    for line in read_log_lines(path):
        if "PERF_CSV" not in line:
            continue
        # Keep everything from the tag onward, in case logcat prepended text.
        frag = line[line.index("PERF_CSV"):].strip()
        parts = frag.split(",")
        if len(parts) < 5:
            continue
        # parts[0] == 'PERF_CSV'; skip the header row.
        if parts[1] == "timestamp_ms":
            continue
        try:
            ts.append(int(parts[1]))
            lat.append(float(parts[2]))
            fps.append(float(parts[3]))
            det.append(int(parts[4]))
        except ValueError:
            continue
    return np.array(ts), np.array(lat), np.array(fps), np.array(det)


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


def write_summary(lat, fps, det, path):
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
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def write_csv(ts, lat, fps, det, path):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["timestamp_ms", "inference_ms", "fps", "detections"])
        for row in zip(ts, lat, fps, det):
            w.writerow(row)


def main():
    if len(sys.argv) < 2:
        print("usage: python runtime_report.py <perf_log.txt>")
        sys.exit(1)
    log_path = sys.argv[1]
    if not os.path.exists(log_path):
        print(f"error: {log_path} not found")
        sys.exit(1)

    ts, lat, fps, det = parse_log(log_path)
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

    write_summary(lat, fps, det, os.path.join(OUTPUT_DIR, "runtime_summary.txt"))
    write_csv(ts, lat, fps, det, os.path.join(OUTPUT_DIR, "runtime_samples.csv"))

    print(f"Parsed {len(lat)} frames.")
    print(f"  mean latency : {lat.mean():.1f} ms")
    print(f"  mean FPS     : {fps.mean():.1f}")
    print(f"Files written to ./{OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
