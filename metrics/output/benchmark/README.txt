FIGURES — what each one shows, and on what data
================================================================
Generated 2026-08-26 22:41
Device for all latency figures: Galaxy S23 Ultra (SM-S918B, Snapdragon 8 Gen 2)
Build: Flutter profile build, TFLite CPU, 4 threads

Every accuracy number here was re-measured by this script. Nothing was read
from a previous run's CSV.

THE RULE: models are only compared against models validated on the SAME
dataset with the SAME protocol. Figures never mix datasets. If a model is
absent from a figure, its weights or its dataset could not be found -- the
script omits it rather than substituting an older number.

F1_per_class_accuracy.png
    Precision / recall / mAP for each class of the SHIPPING model at 320px.
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
