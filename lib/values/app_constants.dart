class AppConstants {
  const AppConstants._();

  static const instance = AppConstants._();

  // ── Input resolution ──────────────────────────────────────────────────────
  // Both models are exported at 320x320, down from 640x640.
  //
  // MEASURED at 640 (Galaxy S23 Ultra, profile build):
  //   total 2951 ms | preprocessing 61 | INFERENCE 2872 | postprocess 18
  //   -> 0.4 FPS, with BOTH models costing ~1.9 s per call despite a ~4x
  //      difference in parameter count.
  //
  // Work scales with inputSize^2, so 320 is a quarter of the values of 640 --
  // on the convolutions AND on the per-call tensor conversion that dominates
  // this pipeline. No other code changes: ModelGeometry reads input size and
  // anchor count from the model's own tensors at load time (320 -> 2100
  // anchors, 448 -> 4116, 640 -> 8400).
  //
  // The 448 pair is exported and sitting in assets/ if 320 costs too much
  // accuracy. Swap these two lines, rebuild, and compare -- but judge it on
  // STAIR RECALL, not mAP. mAP averages away the one failure that hurts.

  static const String cocoModelPath = 'assets/yolo11n_320_float32.tflite';
  static const String cocoLabelPath = 'assets/coco_labels.txt';

  // Accessibility model -- Door, Stair, Window. YOLO11s fine-tuned locally,
  // 120 epochs on 3,203 images. Measured on its own validation set:
  //   precision 0.723  recall 0.656  mAP50 0.710  mAP50-95 0.488
  //   per class (mAP50-95): Stair 0.624, Door 0.529, Window 0.312
  static const String accessibilityModelPath =
      'assets/accessibility_v2_320_float32.tflite';
  static const String accessibilityLabelPath =
      'assets/accessibility_v2_labels.txt';
}
