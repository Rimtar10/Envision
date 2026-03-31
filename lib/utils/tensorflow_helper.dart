import 'dart:math';
import 'dart:typed_data';
import 'dart:ui';

import 'package:image/image.dart';
import 'package:tensorflow_demo/models/detected_object/detected_object_dm.dart';
import 'package:tensorflow_demo/values/typedefs.dart';
import 'package:tflite_flutter/tflite_flutter.dart';

class TensorflowHelper {
  const TensorflowHelper._();

  /// Reusable buffers — avoid allocations on every frame
  static final _inputBuffer = Float32List(1 * 640 * 640 * 3);
  static final _outputBuffer = Float32List(1 * 84 * 8400);

  /// Pre-allocated x-coordinate map for letterbox sampling (640 entries)
  static final _xSrcMap = Int32List(640);

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

  /// Letterbox resize: scale image to fit 640x640 while preserving aspect ratio,
  /// then center on a black canvas. Used only when drawing/returning image.
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

  /// Single-pass: fills [_inputBuffer] with a letterboxed, normalized view of
  /// [image] and returns (padX, padY) for bounding-box coordinate remapping.
  ///
  /// Eliminates the intermediate 640×640 [Image] allocation used in the old
  /// pipeline for the camera path where no drawing is needed.
  static (double, double) _fillInputBufferLetterbox(Image image) {
    const modelSize = 640;
    final scale = modelSize / max(image.width, image.height);
    final newWidth = (image.width * scale).round();
    final newHeight = (image.height * scale).round();
    final xOffset = (modelSize - newWidth) ~/ 2;
    final yOffset = (modelSize - newHeight) ~/ 2;
    final srcWidth = image.width;
    final srcHeight = image.height;

    // Pre-compute source-x for each model-x column (replaces per-pixel division)
    for (int x = 0; x < modelSize; x++) {
      _xSrcMap[x] = (x < xOffset || x >= xOffset + newWidth)
          ? -1
          : ((x - xOffset) / scale).round().clamp(0, srcWidth - 1);
    }

    // Raw RGB bytes — direct memory access, no per-pixel overhead
    final bytes = image.getBytes(order: ChannelOrder.rgb);

    int offset = 0;
    const inv255 = 1.0 / 255.0;

    for (int y = 0; y < modelSize; y++) {
      final srcY = (y < yOffset || y >= yOffset + newHeight)
          ? -1
          : ((y - yOffset) / scale).round().clamp(0, srcHeight - 1);
      final rowBase = srcY < 0 ? 0 : srcY * srcWidth;

      for (int x = 0; x < modelSize; x++) {
        final srcX = _xSrcMap[x];
        if (srcY < 0 || srcX < 0) {
          _inputBuffer[offset] = 0.0;
          _inputBuffer[offset + 1] = 0.0;
          _inputBuffer[offset + 2] = 0.0;
        } else {
          final srcOff = (rowBase + srcX) * 3;
          _inputBuffer[offset] = bytes[srcOff] * inv255;
          _inputBuffer[offset + 1] = bytes[srcOff + 1] * inv255;
          _inputBuffer[offset + 2] = bytes[srcOff + 2] * inv255;
        }
        offset += 3;
      }
    }

    return (xOffset.toDouble(), yOffset.toDouble());
  }

  static AnalyseImageCallback analyseImage(
    Image image, {
    required Interpreter interpreter,
    required List<String> label,
    bool returnDetectedImage = true,
    bool drawObjectOnImage = true,
    double confidenceThreshold = 0.35,
    double iouThreshold = 0.45,
  }) {
    // Single-pass: fill model input buffer + get letterbox pad values
    final (padX, padY) = _fillInputBufferLetterbox(image);

    // Run YOLOv8 inference on pre-filled buffer
    final rawOutput = _runInferenceOnly(interpreter);

    // Parse YOLOv8 output and apply NMS
    final detectedObjectList = _parseYoloOutput(
      rawOutput,
      label,
      confidenceThreshold,
      iouThreshold,
      padX: padX,
      padY: padY,
    );

    // Camera path: skip all image work (most common case)
    if (!drawObjectOnImage && !returnDetectedImage) {
      return (imageBytes: null, detectedObjects: detectedObjectList);
    }

    // Photo path: create letterbox Image only when drawing/returning is needed
    final resizedImage = _resizeWithLetterbox(image);
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

  /// Runs inference on the pre-filled [_inputBuffer] and returns raw output.
  static List<List<List<double>>> _runInferenceOnly(Interpreter interpreter) {
    final input = _inputBuffer.reshape([1, 640, 640, 3]);
    final output = _outputBuffer.reshape([1, 84, 8400]);

    interpreter.run(input, output);

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

  /// Sigmoid activation function
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

    // Determine sigmoid need once, cache for all subsequent frames
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

    final double rawThreshold = useSigmoid
        ? log(confidenceThreshold / (1.0 - confidenceThreshold))
        : confidenceThreshold;

    for (int i = 0; i < 8400; i++) {
      double maxRaw = -double.infinity;
      int bestClass = 0;

      for (int c = 4; c < 84; c++) {
        final rawScore = data[c][i];
        if (rawScore > maxRaw) {
          maxRaw = rawScore;
          bestClass = c - 4;
        }
      }

      if (maxRaw < rawThreshold) continue;

      final double maxScore = useSigmoid ? _sigmoid(maxRaw) : maxRaw;
      if (maxScore < confidenceThreshold) continue;

      {
        final cx = data[0][i] * 640;
        final cy = data[1][i] * 640;
        final w  = data[2][i] * 640;
        final h  = data[3][i] * 640;

        final contentW = 640.0 - 2 * padX;
        final contentH = 640.0 - 2 * padY;
        final remapCx = (cx - padX) / contentW * 640;
        final remapCy = (cy - padY) / contentH * 640;
        final remapW  = w / contentW * 640;
        final remapH  = h / contentH * 640;

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
      );
    }).toList();
  }

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
