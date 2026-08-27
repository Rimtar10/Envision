#!/usr/bin/env python3
"""
build_unified_model.py
======================
ONE fast Envision detector: navigation-relevant COCO classes + Door/Stair/Window.

Pipeline: download a filtered COCO subset (FiftyOne, with auto-retry because
cocodataset.org resets connections) -> add your doors/stairs data -> merge into
one label space -> fine-tune YOLO11s -> export INT8 TFLite.

    pip install ultralytics roboflow fiftyone
    python build_unified_model.py
"""

import os, shutil, yaml, time

# Fix CUDA memory-fragmentation OOM on small GPUs (RTX 3050 4GB). Must be set
# BEFORE torch/ultralytics import anything CUDA.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# ── CONFIG ────────────────────────────────────────────────────────────────────
NAV_COCO = [
    "person", "bicycle", "car", "motorcycle", "bus", "truck",
    "traffic light", "stop sign", "bench", "chair", "couch",
    "dining table", "potted plant", "tv", "refrigerator", "dog", "cat", "backpack",
]
CUSTOM_KEEP = ["Door", "Stair", "Window"]

COCO_TRAIN_SAMPLES = 2500
COCO_VAL_SAMPLES   = 700
BASE_MODEL = "yolo11s.pt"
EPOCHS     = 60
IMGSZ      = 640
BATCH      = 2            # RTX 3050 4GB: 2 is safe. 4 OOMs at 640px.
# Key is read from metrics/roboflow_key.txt (gitignored) or the
# ROBOFLOW_API_KEY env var. It used to be hard-coded on this line.
import sys as _sys
_sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _secrets import roboflow_api_key
ROOT       = "unified_dataset"
EXPORT_INT8 = True


def get_coco(root):
    """Download COCO images with NAV_COCO classes (auto-retry, resumes from
    cache), keep only those labels, export into ROOT in YOLO format."""
    import fiftyone as fo
    import fiftyone.zoo as foz
    from fiftyone import ViewField as F

    NUM_DL_WORKERS = 3      # fewer parallel connections = fewer resets
    MAX_ATTEMPTS   = 60     # each attempt resumes from cache

    for split, n, out in [("train", COCO_TRAIN_SAMPLES, "train"),
                          ("validation", COCO_VAL_SAMPLES, "val")]:
        name = f"navcoco_{split}_{n}"
        ds = None
        for attempt in range(1, MAX_ATTEMPTS + 1):
            if fo.dataset_exists(name):
                fo.delete_dataset(name)
            try:
                print(f"[coco] {split}: attempt {attempt}/{MAX_ATTEMPTS} (resumes from cache) ...")
                ds = foz.load_zoo_dataset(
                    "coco-2017", split=split, label_types=["detections"],
                    classes=NAV_COCO, max_samples=n, only_matching=True,
                    num_workers=NUM_DL_WORKERS, dataset_name=name,
                )
                break
            except Exception as e:
                print(f"   interrupted ({type(e).__name__}); retrying in 3s ...")
                time.sleep(3)
        if ds is None:
            raise RuntimeError("COCO '" + split + "' download kept failing. Try a VPN / "
                               "more stable network, or lower COCO_*_SAMPLES.")
        view = ds.filter_labels("ground_truth", F("label").is_in(NAV_COCO))
        view.export(export_dir=root, dataset_type=fo.types.YOLOv5Dataset,
                    label_field="ground_truth", split=out, classes=NAV_COCO)
    print("[coco] subset ready (" + str(len(NAV_COCO)) + " classes)")


def add_custom(root):
    """Download + prep doors/stairs data and merge into ROOT with class indices
    shifted to sit AFTER the COCO subset."""
    from roboflow import Roboflow
    rf = Roboflow(api_key=roboflow_api_key())
    n_coco = len(NAV_COCO)

    print("[custom] downloading doors dataset ...")
    doors = rf.workspace("rims-workspace-spb1p") \
              .project("doors-dataset-updated-fqlay-q7qtb").version(1).download("yolov8")
    loc = doors.location
    with open(f"{loc}/data.yaml") as f:
        old = yaml.safe_load(f)["names"]
    keep = {old.index(c): i for i, c in enumerate(CUSTOM_KEEP) if c in old}
    for split in ["train", "valid", "test"]:
        lp = f"{loc}/{split}/labels"
        if not os.path.isdir(lp):
            continue
        for lf in os.listdir(lp):
            fp = os.path.join(lp, lf); out = []
            with open(fp) as f:
                for line in f:
                    p = line.split()
                    if p and int(p[0]) in keep:
                        p[0] = str(keep[int(p[0])]); out.append(" ".join(p))
            with open(fp, "w") as f:
                f.write("\n".join(out))

    print("[custom] downloading + merging stairs ...")
    stairs = rf.workspace("yolo-datasets-f9og9") \
               .project("stairs_detection-9av4i-lyswf-orcim-duw1m").version(3).download("yolov8")
    with open(f"{stairs.location}/data.yaml") as f:
        s_names = yaml.safe_load(f)["names"]
    src = next((i for i, n in enumerate(s_names) if "stair" in n.lower()), 0)
    stair_local = CUSTOM_KEEP.index("Stair")
    for split in ["train", "valid", "test"]:
        si, sl = f"{stairs.location}/{split}/images", f"{stairs.location}/{split}/labels"
        di, dl = f"{loc}/{split}/images", f"{loc}/{split}/labels"
        if not os.path.isdir(si):
            continue
        os.makedirs(di, exist_ok=True); os.makedirs(dl, exist_ok=True)
        for im in os.listdir(si):
            shutil.copy(os.path.join(si, im), os.path.join(di, f"stairs_{im}"))
        for lb in os.listdir(sl):
            out = []
            with open(os.path.join(sl, lb)) as f:
                for line in f:
                    p = line.split()
                    if p and int(p[0]) == src:
                        p[0] = str(stair_local); out.append(" ".join(p))
            with open(os.path.join(dl, f"stairs_{lb}"), "w") as f:
                f.write("\n".join(out))

    for csrc, cdst in [("train", "train"), ("valid", "val")]:
        si, sl = f"{loc}/{csrc}/images", f"{loc}/{csrc}/labels"
        di, dl = f"{root}/images/{cdst}", f"{root}/labels/{cdst}"
        os.makedirs(di, exist_ok=True); os.makedirs(dl, exist_ok=True)
        if not os.path.isdir(si):
            continue
        for im in os.listdir(si):
            shutil.copy(os.path.join(si, im), os.path.join(di, f"cust_{im}"))
        for lb in os.listdir(sl):
            out = []
            with open(os.path.join(sl, lb)) as f:
                for line in f:
                    p = line.split()
                    if p:
                        p[0] = str(int(p[0]) + n_coco); out.append(" ".join(p))
            with open(os.path.join(dl, f"cust_{lb}"), "w") as f:
                f.write("\n".join(out))


def write_yaml(root):
    names = list(NAV_COCO) + CUSTOM_KEEP
    path = os.path.join(root, "unified.yaml")
    with open(path, "w") as f:
        yaml.dump({"path": os.path.abspath(root),
                   "train": "images/train", "val": "images/val",
                   "nc": len(names), "names": names}, f, sort_keys=False)
    print("[merge] " + str(len(names)) + " classes -> " + path)
    return path


def main():
    from ultralytics import YOLO

    # If the dataset is already built (images + labels on disk), skip the
    # download/merge phase entirely and go straight to training. Set
    # FORCE_PREP=1 in the environment to rebuild from scratch.
    already_built = (
        os.path.isdir(os.path.join(ROOT, "images", "train"))
        and os.path.isdir(os.path.join(ROOT, "labels", "train"))
        and os.environ.get("FORCE_PREP") != "1"
    )
    if already_built:
        print("[prep] unified_dataset already present -> skipping download/merge.")
        data_yaml = write_yaml(ROOT)   # just (re)write the yaml, cheap
    else:
        get_coco(ROOT)
        add_custom(ROOT)
        data_yaml = write_yaml(ROOT)

    print("\n[train] YOLO11s on " + str(len(NAV_COCO) + len(CUSTOM_KEEP)) + " classes ...")
    model = YOLO(BASE_MODEL)
    model.train(data=data_yaml, epochs=EPOCHS, imgsz=IMGSZ, batch=BATCH,
                name="unified_envision", project="runs", patience=20)

    print("\n[val] validating ...")
    model.val(data=data_yaml, plots=True)

    if EXPORT_INT8:
        print("\n[export] INT8 + float32 TFLite ...")
        try:
            model.export(format="tflite", int8=True, imgsz=IMGSZ, data=data_yaml)
        except Exception as e:
            print("  INT8 export failed (" + str(e) + "); float32 only.")
        model.export(format="tflite", half=False, imgsz=IMGSZ)
    print("\nDone -> runs/detect/unified_envision/")


if __name__ == "__main__":
    main()
