import 'dart:developer' as dev;
import 'dart:math';
import 'dart:typed_data';
import 'dart:ui';

import 'package:image/image.dart';
import 'package:tensorflow_demo/models/detected_object/detected_object_dm.dart';
import 'package:tensorflow_demo/values/app_constants.dart';
import 'package:tensorflow_demo/values/typedefs.dart';
import 'package:tflite_flutter/tflite_flutter.dart';

class TensorflowHelper {
  const TensorflowHelper._();

  /// Reusable buffers for performance (avoid creating new Float32List every frame)
  static final _inputBuffer = Float32List(1 * 640 * 640 * 3);
  static final _outputBuffer = Float32List(1 * 84 * 8400);

  /// Cached once on first inference — model never changes between frames.
  static bool? _cachedUseSigmoid;

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

    // Rectangle drawing
    drawRect(
      imageInput,
      x1: left,
      y1: top,
      x2: rect.right.toInt(),
      y2: rect.bottom.toInt(),
      color: drawColor,
      thickness: 3,
    );

    // Label drawing
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

  /// Letterbox resize: scale image to fit 640x640 while preserving aspect ratio,
  /// then center on a black canvas.
  static Image _resizeWithLetterbox(Image inputImage) {
    const modelInputSize = 640;
    double scale = modelInputSize / max(inputImage.width, inputImage.height);
    int newWidth = (inputImage.width * scale).round();
    int newHeight = (inputImage.height * scale).round();
    var resized = copyResize(inputImage, width: newWidth, height: newHeight);
    final canvas = Image(width: modelInputSize, height: modelInputSize, numChannels: 3);
    int xOffset = (modelInputSize - newWidth) ~/ 2;
    int yOffset = (modelInputSize - newHeight) ~/ 2;
    for (int y = 0; y < resized.height; y++) {
      for (int x = 0; x < resized.width; x++) {
        canvas.setPixel(x + xOffset, y + yOffset, resized.getPixel(x, y));
      }
    }
    return canvas;
  }

  static AnalyseImageCallback analyseImage(
    Image image, {
    required Interpreter interpreter,
    required List<String> label,
    bool returnDetectedImage = true,
    bool drawObjectOnImage = true,
    double confidenceThreshold = 0.5,  // Reduce false positives
    double iouThreshold = 0.45,
  }) {
    const modelSize = 640;

    // Always use letterbox (preserves aspect ratio → better accuracy)
    final resizedImage = _resizeWithLetterbox(image);

    // Run YOLOv8 inference
    final rawOutput = _runYoloInference(resizedImage, interpreter);

    // Compute inverse-transform to map 640x640 bbox back to original image
    final double scale = modelSize / max(image.width, image.height);
    final int newW = (image.width * scale).round();
    final int newH = (image.height * scale).round();
    final double padX = (modelSize - newW) / 2.0;
    final double padY = (modelSize - newH) / 2.0;

    // Parse YOLOv8 output and apply NMS
    final detectedObjectList = _parseYoloOutput(
      rawOutput,
      label,
      confidenceThreshold,
      iouThreshold,
      padX: padX,
      padY: padY,
    );

    // Draw bounding boxes if requested
    if (drawObjectOnImage) {
      for (var detection in detectedObjectList) {
        drawOnImage(
          classification: detection.label,
          imageInput: resizedImage,
          rect: detection.location,
          score: detection.score,
        );
      }
    }

    // Encode output image if requested
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

  static List<List<List<double>>> _runYoloInference(
    Image image,
    Interpreter interpreter,
  ) {
    // Create input tensor: [1, 640, 640, 640, 3] normalized to [0.0, 1.0]
    // Reuse static buffer for performance
    int offset = 0;
    for (int y = 0; y < image.height; y++) {
      for (int x = 0; x < image.width; x++) {
        final pixel = image.getPixel(x, y);
        _inputBuffer[offset++] = pixel.r.toDouble() / 255.0;
        _inputBuffer[offset++] = pixel.g.toDouble() / 255.0;
        _inputBuffer[offset++] = pixel.b.toDouble() / 255.0;
      }
    }
    final input = _inputBuffer.reshape([1, 640, 640, 3]);

    // YOLOv8 output: [1, 84, 8400]
    // 84 = 4 bbox coords + 80 class scores
    // 8400 = number of predictions
    // Reuse static buffer for performance
    final output = _outputBuffer.reshape([1, 84, 8400]);

    // Run inference
    interpreter.run(input, output);

    // Convert to List<List<List<double>>> for downstream use
    final result = <List<List<double>>>[];
    for (final batch in output) {
      final rows = <List<double>>[];
      for (final row in (batch as List)) {
        rows.add((row as List).cast<double>());
      }
      result.add(rows);
    }
    return [result[0]];
  }

  /// Sigmoid activation function for converting logits to probabilities
  static double _sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
  }

  static List<DetectedObjectDm> _parseYoloOutput(
    List<List<List<double>>> output,
    List<String> labels,
    double confidenceThreshold,
    double iouThreshold, {
    double padX = 0,
    double padY = 0,
  }) {
    final data = output[0]; // Shape: [84, 8400]
    final detections = <Map<String, dynamic>>[];

    // ── Determine sigmoid need once, cache for all subsequent frames ──
    if (_cachedUseSigmoid == null) {
      bool foundOutOfRange = false;
      for (int i = 0; i < min(100, 8400) && !foundOutOfRange; i++) {
        for (int c = 4; c < min(14, 84); c++) {
          final v = data[c][i];
          if (v < 0 || v > 1.0) {
            foundOutOfRange = true;
            break;
          }
        }
      }
      _cachedUseSigmoid = foundOutOfRange;
    }
    final bool useSigmoid = _cachedUseSigmoid!;

    // Pre-compute raw-score threshold for early exit.
    // Since sigmoid is monotonically increasing:
    //   sigmoid(x) >= threshold  ⟺  x >= ln(threshold / (1 - threshold))
    final double rawThreshold = useSigmoid
        ? log(confidenceThreshold / (1.0 - confidenceThreshold))
        : confidenceThreshold;

    // Parse all 8400 predictions
    for (int i = 0; i < 8400; i++) {
      // ── KEY OPTIMISATION ──
      // sigmoid is monotonically increasing, so argmax(raw) == argmax(sigmoid(raw)).
      // Find max RAW score first, then apply sigmoid ONLY to the winner.
      // Saves 79 sigmoid calls per prediction (80 → 1).
      double maxRaw = -double.infinity;
      int bestClass = 0;

      for (int c = 4; c < 84; c++) {
        final rawScore = data[c][i];
        if (rawScore > maxRaw) {
          maxRaw = rawScore;
          bestClass = c - 4;
        }
      }

      // Early exit on raw score — avoids sigmoid call for ~99% of predictions
      if (maxRaw < rawThreshold) continue;

      final double maxScore = useSigmoid ? _sigmoid(maxRaw) : maxRaw;
      if (maxScore < confidenceThreshold) continue;

      // Detection passes threshold
      {
        // YOLOv8 TFLite outputs bbox coords in NORMALIZED 0-1 range
        // Multiply by 640 to get letterbox-space pixels, then
        // remove letterbox padding and undo scale to get original-image coords.
        final cx = data[0][i] * 640;
        final cy = data[1][i] * 640;
        final w  = data[2][i] * 640;
        final h  = data[3][i] * 640;

        // Remove letterbox padding and remap to 0-640 range
        // so that renderLocation (which divides by 640) works correctly.
        // Content occupies [padX .. padX+newW] × [padY .. padY+newH] in
        // the 640×640 letterbox. We remap that region to [0..640].
        final contentW = 640.0 - 2 * padX; // == newW
        final contentH = 640.0 - 2 * padY; // == newH
        final remapCx = (cx - padX) / contentW * 640;
        final remapCy = (cy - padY) / contentH * 640;
        final remapW  = w / contentW * 640;
        final remapH  = h / contentH * 640;

        // Convert center coords to corner coords and clamp to 0-640
        final x1 = (remapCx - remapW / 2).clamp(0.0, 640.0);
        final y1 = (remapCy - remapH / 2).clamp(0.0, 640.0);
        final x2 = (remapCx + remapW / 2).clamp(0.0, 640.0);
        final y2 = (remapCy + remapH / 2).clamp(0.0, 640.0);

        detections.add({
          'x1': x1,
          'y1': y1,
          'x2': x2,
          'y2': y2,
          'class': bestClass,
          'confidence': maxScore,
          'label': bestClass < labels.length ? labels[bestClass] : '???',
        });
      }
    }

    // Apply Non-Maximum Suppression
    final nmsDetections = _applyNMS(detections, iouThreshold);

    // Convert to DetectedObjectDm list
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
      );
    }).toList();
  }

  static List<Map<String, dynamic>> _applyNMS(
    List<Map<String, dynamic>> detections,
    double iouThreshold,
  ) {
    // Sort by confidence (highest first)
    detections.sort((a, b) =>
        (b['confidence'] as double).compareTo(a['confidence'] as double));

    final keep = <Map<String, dynamic>>[];

    while (detections.isNotEmpty) {
      final best = detections.removeAt(0);
      keep.add(best);

      // Remove overlapping boxes
      detections.removeWhere((det) {
        final iou = _calculateIoU(best, det);
        return iou > iouThreshold;
      });
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