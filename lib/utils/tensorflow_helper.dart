import 'dart:math';
import 'dart:typed_data';
import 'dart:ui';

import 'package:image/image.dart';
import 'package:tensorflow_demo/models/detected_object/detected_object_dm.dart';
import 'package:tensorflow_demo/values/typedefs.dart';
import 'package:tflite_flutter/tflite_flutter.dart';

/// Geometry of a loaded YOLO model, read from the interpreter's own tensors
/// instead of being hard-coded.
///
/// This used to be `640` and `8400` scattered through a dozen places, which
/// meant changing the export resolution required editing the Dart source in
/// several spots and getting the anchor count right by hand
/// (640 -> 80²+40²+20² = 8400; 448 -> 56²+28²+14² = 4116). Reading it from the
/// tensors makes a re-export at a different `imgsz` a drop-in change.
class ModelGeometry {
  const ModelGeometry({
    required this.inputSize,
    required this.numClasses,
    required this.numAnchors,
  });

  /// Square model input edge in pixels (e.g. 640, 448).
  final int inputSize;

  /// Number of object classes the head predicts.
  final int numClasses;

  /// Number of candidate boxes per frame (the `8400` in `[1, nc+4, 8400]`).
  final int numAnchors;

  /// Derive geometry from a loaded interpreter.
  ///
  /// Input  tensor: [1, S, S, 3]
  /// Output tensor: [1, nc + 4, A]
  factory ModelGeometry.fromInterpreter(Interpreter interpreter) {
    final inShape = interpreter.getInputTensors().first.shape;
    final outShape = interpreter.getOutputTensors().first.shape;

    // Input is [1, H, W, 3] for the Ultralytics TFLite export. Guard against a
    // channels-first export by picking the largest spatial dim.
    final inputSize = inShape.length >= 3
        ? [inShape[1], inShape[2]].reduce(max)
        : 640;

    // Output is [1, nc + 4, anchors]; anchors is always the larger of the two.
    final a = outShape.length >= 3 ? outShape[1] : 84;
    final b = outShape.length >= 3 ? outShape[2] : 8400;
    final channels = min(a, b);
    final anchors = max(a, b);

    return ModelGeometry(
      inputSize: inputSize,
      numClasses: channels - 4,
      numAnchors: anchors,
    );
  }

  @override
  String toString() =>
      'ModelGeometry(input: ${inputSize}x$inputSize, nc: $numClasses, '
      'anchors: $numAnchors)';
}

class TensorflowHelper {
  const TensorflowHelper._();

  // ── Fused preprocessing feature flag ───────────────────────────────────────
  //
  // The default pipeline does THREE full-image passes per frame:
  //   1. YUV/NV21 -> RGB   (ImageUtils, full sensor resolution)
  //   2. copyRotate        (full-resolution allocation + copy)
  //   3. letterbox sample  (into the model input buffer)
  //
  // `_fillInputBufferFused` collapses steps 2 and 3 into one, removing a
  // full-resolution image allocation and copy per frame. On a 720x480 stream
  // that is ~345k pixels of allocation + copy eliminated, every frame.
  //
  // It is OFF by default because the rotation index maths must match the
  // `image` package's `copyRotate` convention EXACTLY, and a silent mismatch
  // would rotate or mirror every bounding box in an app blind users rely on.
  //
  // TO ENABLE:
  //     flutter test test/letterbox_parity_test.dart
  // That test asserts the fused path is byte-identical to
  // copyRotate + letterbox for 0/90/180/270 across several image sizes.
  // If it passes, set this to true and you get the speedup for free.
  // If it fails, it prints the correct mapping for your `image` version.
  static const bool useFusedRotationLetterbox = false;

  // ── Reusable buffers, keyed by input size ─────────────────────────────────
  // Allocating these per frame created heavy GC pressure. Keyed by size so a
  // 640 model and a 448 model can coexist without clobbering each other.
  static final Map<int, Float32List> _inputBuffers = {};
  static final Map<int, Int32List> _xSrcMaps = {};

  static Float32List _inputBuffer(int size) =>
      _inputBuffers[size] ??= Float32List(1 * size * size * 3);

  static Int32List _xSrcMap(int size) => _xSrcMaps[size] ??= Int32List(size);

  /// Read-only view of the model input buffer, for tests only.
  /// See test/letterbox_parity_test.dart.
  static Float32List debugInputBuffer(int size) => _inputBuffer(size);

  // ── Class-specific confidence thresholds ─────────────────────────────────
  // Hazardous / mobility-relevant classes use a lower threshold so they are
  // rarely missed. Everything else uses the general confidenceThreshold (0.40).
  // The pre-filter pass uses _minEffectiveThreshold so no candidate is skipped
  // before we know its class label.
  static const double _minEffectiveThreshold = 0.28;

  static const Map<String, double> _hazardThresholds = {
    'person': 0.28,
    'car': 0.28,
    'truck': 0.28,
    'bus': 0.28,
    'motorcycle': 0.28,
    'bicycle': 0.30,
    'dog': 0.32,
    'fire hydrant': 0.34,
    'stop sign': 0.34,
    'traffic light': 0.34,
    'bench': 0.38,
    'chair': 0.38,
    'dining table': 0.38,
    // Custom accessibility classes, from the dedicated 3-class model.
    // Measured on runs/detect/runs/accessibility_v2 (yolo11s, 120 epochs):
    //   overall  precision 0.705  recall 0.675  mAP50 0.706  mAP50-95 0.488
    // Stair is kept lowest of the three: it is the weakest class to miss and
    // the most dangerous. Window is highest: it is the least accurate class
    // AND the least actionable for a blind pedestrian, so false positives
    // there cost more than false negatives.
    'stair': 0.45,
    'stairs': 0.45,
    'door': 0.55,
    'window': 0.60,
  };

  static void drawOnImage({
    required Image imageInput,
    required Rect rect,
    required num score,
    String? classification,
    Color? color,
  }) {
    final drawColor = color ?? ColorRgb8(255, 255, 255);

    final top = rect.top.toInt();
    final left = rect.left.toInt();

    drawRect(
      imageInput,
      x1: left,
      y1: top,
      x2: rect.right.toInt(),
      y2: rect.bottom.toInt(),
      color: drawColor,
      thickness: 3,
    );

    if (classification == null) return;
    drawString(
      imageInput,
      '$classification ${score.toStringAsFixed(2)}',
      font: arial14,
      x: left + 1,
      y: top + 1,
      color: drawColor,
    );
  }

  /// Letterbox resize: scale image to fit [size]x[size] while preserving aspect
  /// ratio, then center on a black canvas. Used only when drawing/returning an
  /// image (the photo path), never on the camera hot path.
  static Image _resizeWithLetterbox(Image inputImage, int size) {
    final scale = size / max(inputImage.width, inputImage.height);
    final newWidth = (inputImage.width * scale).round();
    final newHeight = (inputImage.height * scale).round();
    final resized = copyResize(inputImage, width: newWidth, height: newHeight);
    final canvas = Image(width: size, height: size, numChannels: 3);
    final xOffset = (size - newWidth) ~/ 2;
    final yOffset = (size - newHeight) ~/ 2;
    for (int y = 0; y < resized.height; y++) {
      for (int x = 0; x < resized.width; x++) {
        canvas.setPixel(x + xOffset, y + yOffset, resized.getPixel(x, y));
      }
    }
    return canvas;
  }

  /// Fills the model input buffer with a letterboxed, normalized view of
  /// [image] and returns (padX, padY) for bounding-box coordinate remapping.
  ///
  /// [image] must already be rotated to display orientation.
  static (double, double) fillInputBufferLetterbox(Image image, int modelSize) {
    final buffer = _inputBuffer(modelSize);
    final xSrcMap = _xSrcMap(modelSize);

    final scale = modelSize / max(image.width, image.height);
    final newWidth = (image.width * scale).round();
    final newHeight = (image.height * scale).round();
    final xOffset = (modelSize - newWidth) ~/ 2;
    final yOffset = (modelSize - newHeight) ~/ 2;
    final srcWidth = image.width;
    final srcHeight = image.height;

    // Pre-compute source-x for each model-x column (replaces per-pixel division)
    for (int x = 0; x < modelSize; x++) {
      xSrcMap[x] = (x < xOffset || x >= xOffset + newWidth)
          ? -1
          : ((x - xOffset) / scale).round().clamp(0, srcWidth - 1);
    }

    // Raw RGB bytes — direct memory access, no per-pixel Pixel object overhead
    final bytes = image.getBytes(order: ChannelOrder.rgb);

    int offset = 0;
    const inv255 = 1.0 / 255.0;

    for (int y = 0; y < modelSize; y++) {
      final srcY = (y < yOffset || y >= yOffset + newHeight)
          ? -1
          : ((y - yOffset) / scale).round().clamp(0, srcHeight - 1);
      final rowBase = srcY < 0 ? 0 : srcY * srcWidth;

      for (int x = 0; x < modelSize; x++) {
        final srcX = xSrcMap[x];
        if (srcY < 0 || srcX < 0) {
          buffer[offset] = 0.0;
          buffer[offset + 1] = 0.0;
          buffer[offset + 2] = 0.0;
        } else {
          final srcOff = (rowBase + srcX) * 3;
          buffer[offset] = bytes[srcOff] * inv255;
          buffer[offset + 1] = bytes[srcOff + 1] * inv255;
          buffer[offset + 2] = bytes[srcOff + 2] * inv255;
        }
        offset += 3;
      }
    }

    return (xOffset.toDouble(), yOffset.toDouble());
  }

  /// Fused rotate + letterbox: samples directly from the UNROTATED [image],
  /// applying a [rotationDegrees] rotation via index arithmetic instead of
  /// materialising a rotated copy.
  ///
  /// Saves one full-resolution allocation and copy per frame. Guarded by
  /// [useFusedRotationLetterbox]; see the comment on that flag and the
  /// parity test in test/letterbox_parity_test.dart.
  ///
  /// Rotation is clockwise, matching `copyRotate(image, angle: n)`.
  /// For a source of W x H, destination dimensions are:
  ///   0/180  -> W x H
  ///   90/270 -> H x W
  /// Inverse mapping (destination -> source):
  ///     0   : sx = dx,               sy = dy
  ///    90   : sx = dy,               sy = (H - 1) - dx
  ///   180   : sx = (W - 1) - dx,     sy = (H - 1) - dy
  ///   270   : sx = (W - 1) - dy,     sy = dx
  static (double, double) fillInputBufferFused(
    Image image,
    int modelSize,
    int rotationDegrees,
  ) {
    final buffer = _inputBuffer(modelSize);

    final rot = ((rotationDegrees % 360) + 360) % 360;
    final swap = rot == 90 || rot == 270;

    final srcW = image.width;
    final srcH = image.height;
    // Dimensions AFTER rotation — this is the geometry the letterbox sees.
    final rotW = swap ? srcH : srcW;
    final rotH = swap ? srcW : srcH;

    final scale = modelSize / max(rotW, rotH);
    final newWidth = (rotW * scale).round();
    final newHeight = (rotH * scale).round();
    final xOffset = (modelSize - newWidth) ~/ 2;
    final yOffset = (modelSize - newHeight) ~/ 2;

    // Map model-x -> rotated-image-x once per frame.
    final xSrcMap = _xSrcMap(modelSize);
    for (int x = 0; x < modelSize; x++) {
      xSrcMap[x] = (x < xOffset || x >= xOffset + newWidth)
          ? -1
          : ((x - xOffset) / scale).round().clamp(0, rotW - 1);
    }

    final bytes = image.getBytes(order: ChannelOrder.rgb);

    int offset = 0;
    const inv255 = 1.0 / 255.0;

    for (int y = 0; y < modelSize; y++) {
      final rotY = (y < yOffset || y >= yOffset + newHeight)
          ? -1
          : ((y - yOffset) / scale).round().clamp(0, rotH - 1);

      for (int x = 0; x < modelSize; x++) {
        final rotX = xSrcMap[x];
        if (rotY < 0 || rotX < 0) {
          buffer[offset] = 0.0;
          buffer[offset + 1] = 0.0;
          buffer[offset + 2] = 0.0;
          offset += 3;
          continue;
        }

        // Un-rotate: (rotX, rotY) in display space -> (sx, sy) in sensor space
        final int sx;
        final int sy;
        switch (rot) {
          case 90:
            sx = rotY;
            sy = (srcH - 1) - rotX;
          case 180:
            sx = (srcW - 1) - rotX;
            sy = (srcH - 1) - rotY;
          case 270:
            sx = (srcW - 1) - rotY;
            sy = rotX;
          default:
            sx = rotX;
            sy = rotY;
        }

        final srcOff = (sy * srcW + sx) * 3;
        buffer[offset] = bytes[srcOff] * inv255;
        buffer[offset + 1] = bytes[srcOff + 1] * inv255;
        buffer[offset + 2] = bytes[srcOff + 2] * inv255;
        offset += 3;
      }
    }

    return (xOffset.toDouble(), yOffset.toDouble());
  }

  /// Runs detection on [image].
  ///
  /// [geometry] must come from the same interpreter that will run inference.
  /// If [rotationDegrees] is non-zero AND [useFusedRotationLetterbox] is on,
  /// the rotation is applied during sampling and [image] should be the raw,
  /// unrotated frame. Otherwise [image] must already be rotated.
  static AnalyseImageCallback analyseImage(
    Image image, {
    required Interpreter interpreter,
    required List<String> label,
    required ModelGeometry geometry,
    int rotationDegrees = 0,
    bool returnDetectedImage = true,
    bool drawObjectOnImage = true,
    double confidenceThreshold = 0.40,
    double iouThreshold = 0.50,
  }) {
    final size = geometry.inputSize;

    final (padX, padY) =
        (useFusedRotationLetterbox && rotationDegrees % 360 != 0)
            ? fillInputBufferFused(image, size, rotationDegrees)
            : fillInputBufferLetterbox(image, size);

    final rawOutput = _runInferenceOnly(interpreter, geometry);

    final detectedObjectList = _parseYoloOutput(
      rawOutput,
      label,
      confidenceThreshold,
      iouThreshold,
      geometry: geometry,
      padX: padX,
      padY: padY,
    );

    // Camera path: skip all image work (most common case)
    if (!drawObjectOnImage && !returnDetectedImage) {
      return (imageBytes: null, detectedObjects: detectedObjectList);
    }

    // Photo path: create letterbox Image only when drawing/returning is needed
    final resizedImage = _resizeWithLetterbox(image, size);
    if (drawObjectOnImage) {
      for (final detection in detectedObjectList) {
        drawOnImage(
          classification: detection.label,
          imageInput: resizedImage,
          rect: detection.location,
          score: detection.score,
        );
      }
    }

    final imageOutput = returnDetectedImage
        ? encodeJpg(
            copyResize(
              resizedImage,
              height: image.height,
              width: image.width,
            ),
          )
        : null;

    return (imageBytes: imageOutput, detectedObjects: detectedObjectList);
  }

  /// Reusable output buffers keyed by class count. Allocating the
  /// [1, nc+4, anchors] tensor on every frame (~2.8 MB for the 80-class COCO
  /// model at 640) created heavy GC pressure; we allocate once per model shape
  /// and reuse it — the interpreter overwrites it on each run.
  static final Map<String, List> _outputCache = {};

  static List<List<List<double>>> _runInferenceOnly(
    Interpreter interpreter,
    ModelGeometry g,
  ) {
    final input =
        _inputBuffer(g.inputSize).reshape([1, g.inputSize, g.inputSize, 3]);

    final key = '${g.numClasses}x${g.numAnchors}';
    final output = _outputCache[key] ??=
        Float32List(1 * (g.numClasses + 4) * g.numAnchors)
            .reshape([1, g.numClasses + 4, g.numAnchors]);

    interpreter.run(input, output);

    final batch = output[0] as List;
    final rows = List<List<double>>.generate(
      batch.length,
      (i) => (batch[i] as List).cast<double>(),
      growable: false,
    );
    return [rows];
  }

  /// Sigmoid activation function
  static double _sigmoid(double x) => 1.0 / (1.0 + exp(-x));

  static List<DetectedObjectDm> _parseYoloOutput(
    List<List<List<double>>> output,
    List<String> labels,
    double confidenceThreshold,
    double iouThreshold, {
    required ModelGeometry geometry,
    double padX = 0,
    double padY = 0,
  }) {
    final data = output[0]; // Shape: [nc+4, anchors]
    final nc = geometry.numClasses;
    final anchors = geometry.numAnchors;
    final size = geometry.inputSize.toDouble();
    final numChannels = nc + 4;
    final detections = <Map<String, dynamic>>[];

    // Ultralytics TFLite exports already apply sigmoid; detect a raw-logit
    // export by looking for scores outside [0, 1].
    bool useSigmoid = false;
    for (int i = 0; i < min(100, anchors) && !useSigmoid; i++) {
      for (int c = 4; c < min(14, numChannels); c++) {
        final v = data[c][i];
        if (v < 0 || v > 1.0) {
          useSigmoid = true;
          break;
        }
      }
    }

    // Pre-filter using the lowest possible effective threshold so no hazard
    // class candidate is discarded before we read its label.
    final double rawPreFilter = useSigmoid
        ? log(_minEffectiveThreshold / (1.0 - _minEffectiveThreshold))
        : _minEffectiveThreshold;

    // Letterbox content extent, used to undo the padding.
    final contentW = size - 2 * padX;
    final contentH = size - 2 * padY;

    for (int i = 0; i < anchors; i++) {
      double maxRaw = -double.infinity;
      int bestClass = 0;

      for (int c = 4; c < numChannels; c++) {
        final rawScore = data[c][i];
        if (rawScore > maxRaw) {
          maxRaw = rawScore;
          bestClass = c - 4;
        }
      }

      if (maxRaw < rawPreFilter) continue;

      final double maxScore = useSigmoid ? _sigmoid(maxRaw) : maxRaw;

      // Apply class-specific threshold: hazard classes use a lower bar,
      // everything else uses the general confidenceThreshold.
      final labelName = bestClass < labels.length ? labels[bestClass] : '';
      final double effectiveThreshold =
          _hazardThresholds[labelName.toLowerCase()] ?? confidenceThreshold;
      if (maxScore < effectiveThreshold) continue;

      final cx = data[0][i] * size;
      final cy = data[1][i] * size;
      final w = data[2][i] * size;
      final h = data[3][i] * size;

      final remapCx = (cx - padX) / contentW * size;
      final remapCy = (cy - padY) / contentH * size;
      final remapW = w / contentW * size;
      final remapH = h / contentH * size;

      final x1 = (remapCx - remapW / 2).clamp(0.0, size);
      final y1 = (remapCy - remapH / 2).clamp(0.0, size);
      final x2 = (remapCx + remapW / 2).clamp(0.0, size);
      final y2 = (remapCy + remapH / 2).clamp(0.0, size);

      detections.add({
        'x1': x1,
        'y1': y1,
        'x2': x2,
        'y2': y2,
        'class': bestClass,
        'confidence': maxScore,
        'label': labelName.isEmpty ? '???' : labelName,
        // Vertical stretch introduced by undoing the letterbox. VoiceService
        // divides the box height by this before estimating distance, otherwise
        // objects read as closer than they are whenever padding lands on the
        // y axis (portrait sensors, front camera, tablets).
        'yStretch': size / contentH,
      });
    }

    final nmsDetections = _applyNMS(detections, iouThreshold);

    return nmsDetections.map((det) {
      return DetectedObjectDm(
        label: det['label'] as String,
        score: det['confidence'] as double,
        location: Rect.fromLTRB(
          det['x1'] as double,
          det['y1'] as double,
          det['x2'] as double,
          det['y2'] as double,
        ),
        yStretch: det['yStretch'] as double,
        modelSize: size,
      );
    }).toList();
  }

  /// Non-maximum suppression, applied PER CLASS.
  ///
  /// This used to be class-agnostic, which meant any two overlapping boxes
  /// suppressed each other regardless of what they were: a person standing in
  /// a doorway, a person on a bicycle, or a chair at a dining table would lose
  /// one of the two detections entirely. YOLO's reference NMS is per-class, and
  /// for this app the dropped box was often the one the user needed to hear.
  static List<Map<String, dynamic>> _applyNMS(
    List<Map<String, dynamic>> detections,
    double iouThreshold,
  ) {
    detections.sort((a, b) =>
        (b['confidence'] as double).compareTo(a['confidence'] as double));

    final keep = <Map<String, dynamic>>[];

    while (detections.isNotEmpty) {
      final best = detections.removeAt(0);
      keep.add(best);

      final bestClass = best['class'] as int;
      detections.removeWhere((det) =>
          (det['class'] as int) == bestClass &&
          _calculateIoU(best, det) > iouThreshold);
    }

    return keep;
  }

  static double _calculateIoU(
    Map<String, dynamic> box1,
    Map<String, dynamic> box2,
  ) {
    final x1 = max(box1['x1'] as double, box2['x1'] as double);
    final y1 = max(box1['y1'] as double, box2['y1'] as double);
    final x2 = min(box1['x2'] as double, box2['x2'] as double);
    final y2 = min(box1['y2'] as double, box2['y2'] as double);

    final intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1);

    final area1 = ((box1['x2'] as double) - (box1['x1'] as double)) *
        ((box1['y2'] as double) - (box1['y1'] as double));
    final area2 = ((box2['x2'] as double) - (box2['x1'] as double)) *
        ((box2['y2'] as double) - (box2['y1'] as double));

    final union = area1 + area2 - intersection;

    return union > 0 ? intersection / union : 0;
  }
}
