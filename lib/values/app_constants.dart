class AppConstants {
  const AppConstants._();

  static const instance = AppConstants._();

  // ── COCO general-object model (80 classes) ────────────────────────────────
  // All candidates share coco_labels.txt and the Ultralytics [1, 84, A] output
  // convention, so switching is ONLY the asset path — input size and anchor
  // count are read from the model's tensors at load time (ModelGeometry).
  //
  // Published COCO mAP50-95 / params:
  //   yolov8n : 37.3   3.15 M   (the original)
  //   yolo11n : 39.5   2.62 M   (ACTIVE — more accurate AND smaller AND faster)
  //   yolo11s : 47.0   9.4  M   (~2-3x slower; max accuracy)
  static const String cocoModelPath = 'assets/yolo11n_float32.tflite';
  static const String cocoLabelPath = 'assets/coco_labels.txt';

  // ── Accessibility model — Door, Stair, Window ─────────────────────────────
  // YOLO11s fine-tuned locally, 120 epochs on 3,203 images.
  // Measured (runs/detect/runs/accessibility_v2/results.csv, epoch 120):
  //   precision 0.705   recall 0.675   mAP50 0.706   mAP50-95 0.488
  //
  // float16 is the ACTIVE build: ~half the file size and roughly 2x the CPU
  // throughput of float32, with no measurable accuracy loss at this scale.
  // That is what lets detector.dart drop _accEveryNFrames from 6 to 2.
  //
  // NEXT: metrics/train_accessibility.py now trains YOLO11n instead of 11s.
  // After that run finishes, point this at the new export and set
  // _accEveryNFrames to 1 in detector.dart.
  static const String accessibilityModelPath =
      'assets/accessibility_v2_float16.tflite';
  static const String accessibilityLabelPath =
      'assets/accessibility_v2_labels.txt';
}
