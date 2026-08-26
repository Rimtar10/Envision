import 'dart:math';
import 'dart:typed_data';

import 'package:image/image.dart';

/// Letterbox preprocessing: turns a camera frame into the normalized square
/// tensor a YOLO model expects.
///
/// This lives in its OWN file, deliberately importing nothing but `image`, so
/// it can be unit-tested with `flutter test` on a plain desktop VM. Its old
/// home (tensorflow_helper.dart) imports `tflite_flutter`, which tries to load
/// a native library at import time and makes the file untestable off-device.
/// The geometry maths here is exactly the part most worth testing.
class Letterbox {
  const Letterbox._();

  // ── Reusable buffers, keyed by input size ──────────────────────────────────
  // Allocating these per frame created heavy GC pressure. Keyed by size so a
  // 640 model and a 448 model can coexist without clobbering each other.
  static final Map<int, Float32List> _inputBuffers = {};
  static final Map<int, Int32List> _xSrcMaps = {};

  static Float32List buffer(int size) =>
      _inputBuffers[size] ??= Float32List(1 * size * size * 3);

  static Int32List _xSrcMap(int size) => _xSrcMaps[size] ??= Int32List(size);

  /// Fills the model input buffer with a letterboxed, normalized view of
  /// [image] and returns (padX, padY) for bounding-box coordinate remapping.
  ///
  /// [image] must ALREADY be rotated to display orientation.
  static (double, double) fill(Image image, int modelSize) {
    final buf = buffer(modelSize);
    final xSrcMap = _xSrcMap(modelSize);

    final scale = modelSize / max(image.width, image.height);
    final newWidth = (image.width * scale).round();
    final newHeight = (image.height * scale).round();
    final xOffset = (modelSize - newWidth) ~/ 2;
    final yOffset = (modelSize - newHeight) ~/ 2;
    final srcWidth = image.width;
    final srcHeight = image.height;

    // Pre-compute source-x per model-x column (replaces per-pixel division)
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
          buf[offset] = 0.0;
          buf[offset + 1] = 0.0;
          buf[offset + 2] = 0.0;
        } else {
          final srcOff = (rowBase + srcX) * 3;
          buf[offset] = bytes[srcOff] * inv255;
          buf[offset + 1] = bytes[srcOff + 1] * inv255;
          buf[offset + 2] = bytes[srcOff + 2] * inv255;
        }
        offset += 3;
      }
    }

    return (xOffset.toDouble(), yOffset.toDouble());
  }

  /// Fused rotate + letterbox: samples directly from the UNROTATED [image],
  /// applying [rotationDegrees] via index arithmetic instead of materialising a
  /// rotated copy. Saves one full-resolution allocation and copy per frame.
  ///
  /// Rotation is clockwise, matching `copyRotate(image, angle: n)`.
  /// For a source of W x H the rotated dimensions are:
  ///   0/180  -> W x H
  ///   90/270 -> H x W
  /// Inverse mapping (rotated -> source):
  ///     0  : sx = rx,             sy = ry
  ///    90  : sx = ry,             sy = (H - 1) - rx
  ///   180  : sx = (W - 1) - rx,   sy = (H - 1) - ry
  ///   270  : sx = (W - 1) - ry,   sy = rx
  static (double, double) fillFused(
    Image image,
    int modelSize,
    int rotationDegrees,
  ) {
    final buf = buffer(modelSize);

    final rot = ((rotationDegrees % 360) + 360) % 360;
    final swap = rot == 90 || rot == 270;

    final srcW = image.width;
    final srcH = image.height;
    // Dimensions AFTER rotation — the geometry the letterbox sees.
    final rotW = swap ? srcH : srcW;
    final rotH = swap ? srcW : srcH;

    final scale = modelSize / max(rotW, rotH);
    final newWidth = (rotW * scale).round();
    final newHeight = (rotH * scale).round();
    final xOffset = (modelSize - newWidth) ~/ 2;
    final yOffset = (modelSize - newHeight) ~/ 2;

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
          buf[offset] = 0.0;
          buf[offset + 1] = 0.0;
          buf[offset + 2] = 0.0;
          offset += 3;
          continue;
        }

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
        buf[offset] = bytes[srcOff] * inv255;
        buf[offset + 1] = bytes[srcOff + 1] * inv255;
        buf[offset + 2] = bytes[srcOff + 2] * inv255;
        offset += 3;
      }
    }

    return (xOffset.toDouble(), yOffset.toDouble());
  }
}
