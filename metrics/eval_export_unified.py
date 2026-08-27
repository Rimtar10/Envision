#!/usr/bin/env python3
"""
eval_export_unified.py
======================
Run AFTER build_unified_model.py has trained (best.pt exists). Does two things
the training script's export step choked on:

  1. Per-class validation -> prints a table AND writes per_class_metrics.csv
     so we can see exactly which of the 21 classes are strong/weak.
  2. A clean *float32* TFLite export (no INT8 -> no 6.7GB RAM blow-up), copied
     to assets/unified_float32.tflite for the app.

    python metrics\eval_export_unified.py
"""
import os, csv, glob, shutil

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

DATA   = os.path.join("unified_dataset", "unified.yaml")
IMGSZ  = 640
ASSETS = "assets"
EXPORT_TFLITE = True


def find_best():
    cands = glob.glob(os.path.join("runs", "**", "unified_envision*", "weights", "best.pt"),
                      recursive=True)
    if not cands:
        raise FileNotFoundError("No unified_envision*/weights/best.pt found under runs/.")
    return max(cands, key=os.path.getmtime)   # newest


def main():
    from ultralytics import YOLO
    best = find_best()
    print(f"[eval] using {best}")
    model = YOLO(best)

    # per-class validation
    r = model.val(data=DATA, imgsz=IMGSZ, plots=True, verbose=True)
    names = model.names   # {idx: name}

    rows = []
    # r.box.maps = mAP50-95 per class ; per-class p/r/ap50 via ap_class_index
    ap50   = getattr(r.box, "ap50", None)
    p      = getattr(r.box, "p", None)
    rec    = getattr(r.box, "r", None)
    idxs   = getattr(r.box, "ap_class_index", None)
    if idxs is not None:
        for j, ci in enumerate(idxs):
            rows.append({
                "class": names[int(ci)],
                "precision": round(float(p[j]), 3) if p is not None else "",
                "recall":    round(float(rec[j]), 3) if rec is not None else "",
                "mAP50":     round(float(ap50[j]), 3) if ap50 is not None else "",
                "mAP50_95":  round(float(r.box.maps[int(ci)]), 3),
            })
    rows.sort(key=lambda x: x["mAP50"] if x["mAP50"] != "" else 0)

    out = os.path.join("metrics", "output", "unified")
    os.makedirs(out, exist_ok=True)
    csv_path = os.path.join(out, "per_class_metrics.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["class", "precision", "recall", "mAP50", "mAP50_95"])
        w.writeheader(); w.writerows(rows)

    print("\n=== PER-CLASS (worst -> best by mAP50) ===")
    print(f"{'class':<16}{'P':>7}{'R':>7}{'mAP50':>8}{'mAP50-95':>10}")
    for r_ in rows:
        print(f"{r_['class']:<16}{str(r_['precision']):>7}{str(r_['recall']):>7}"
              f"{str(r_['mAP50']):>8}{str(r_['mAP50_95']):>10}")
    print(f"\noverall  mAP50={float(r.box.map50):.3f}  mAP50-95={float(r.box.map):.3f}")
    print(f"table -> {csv_path}")

    if EXPORT_TFLITE:
        print("\n[export] float32 TFLite (this can take a few minutes) ...")
        try:
            exported = model.export(format="tflite", half=False, imgsz=IMGSZ)
            src = str(exported)
            if not src.endswith(".tflite") or not os.path.exists(src):
                hits = glob.glob(os.path.join(os.path.dirname(best),
                                              "best_saved_model", "*float32.tflite"))
                src = hits[0] if hits else None
            if src and os.path.exists(src):
                os.makedirs(ASSETS, exist_ok=True)
                dst = os.path.join(ASSETS, "unified_float32.tflite")
                shutil.copy(src, dst)
                print(f"OK -> {dst}")
            else:
                print("Export finished but .tflite not found; look in the *_saved_model folder.")
        except Exception as e:
            print(f"TFLite export failed: {e}")


if __name__ == "__main__":
    main()
