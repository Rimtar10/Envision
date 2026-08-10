// Proves the fused rotate+letterbox path is byte-identical to the original
// copyRotate + letterbox path, for every rotation the camera can report.
//
// WHY THIS EXISTS
// ---------------
// TensorflowHelper.fillInputBufferFused removes a full-resolution image
// allocation and copy from every camera frame by folding the rotation into the
// sampling loop. That is a real speedup, but it depends on reproducing the
// `image` package's copyRotate convention EXACTLY. Get the direction wrong and
// every bounding box is silently rotated or mirrored — in an app blind people
// rely on to avoid stairs.
//
// So the fast path ships DISABLED. Run:
//     flutter test test/letterbox_parity_test.dart
// If it passes, set TensorflowHelper.useFusedRotationLetterbox = true.
// If it fails, the output tells you which mapping your `image` version wants.

import 'package:flutter_test/flutter_test.dart';
import 'package:image/image.dart';
import 'package:tensorflow_demo/utils/tensorflow_helper.dart';

/// Deterministic image where every pixel encodes its own coordinates, so any
/// transposition, mirroring or off-by-one shows up as a mismatch.
Image _syntheticImage(int width, int height) {
  final img = Image(width: width, height: height, numChannels: 3);
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      img.setPixelRgb(x, y, x % 256, y % 256, (x * 7 + y * 13) % 256);
    }
  }
  return img;
}

void main() {
  // Realistic camera geometries: 4:3 landscape (ResolutionPreset.medium),
  // 16:9, a square, and a portrait sensor.
  const sizes = <List<int>>[
    [720, 480],
    [640, 480],
    [1280, 720],
    [480, 480],
    [480, 720],
  ];
  const rotations = <int>[0, 90, 180, 270];
  const modelSizes = <int>[640, 448];

  for (final modelSize in modelSizes) {
    for (final size in sizes) {
      for (final rot in rotations) {
        test('fused == copyRotate+letterbox  '
            '${size[0]}x${size[1]} rot=$rot model=$modelSize', () {
          final src = _syntheticImage(size[0], size[1]);

          // Reference path: materialise the rotated image, then letterbox it.
          final rotated = rot == 0 ? src : copyRotate(src, angle: rot);
          final (refPadX, refPadY) =
              TensorflowHelper.fillInputBufferLetterbox(rotated, modelSize);
          final reference =
              List<double>.from(TensorflowHelper.debugInputBuffer(modelSize));

          // Fast path: sample straight from the unrotated image.
          final (fusePadX, fusePadY) =
              TensorflowHelper.fillInputBufferFused(src, modelSize, rot);
          final fused =
              List<double>.from(TensorflowHelper.debugInputBuffer(modelSize));

          expect(fusePadX, refPadX, reason: 'padX must match');
          expect(fusePadY, refPadY, reason: 'padY must match');
          expect(fused.length, reference.length);

          var mismatches = 0;
          var firstBad = -1;
          for (int i = 0; i < reference.length; i++) {
            if ((reference[i] - fused[i]).abs() > 1e-9) {
              if (firstBad < 0) firstBad = i;
              mismatches++;
            }
          }

          expect(
            mismatches,
            0,
            reason: 'ROTATION MAPPING MISMATCH at rot=$rot.\n'
                '$mismatches of ${reference.length} buffer entries differ '
                '(first at index $firstBad).\n'
                'The `image` package version in pubspec.lock rotates the other '
                'way, or uses a different origin, than the switch in '
                'TensorflowHelper.fillInputBufferFused assumes.\n'
                'Fix: swap the 90 and 270 cases in that switch and re-run. '
                'Leave useFusedRotationLetterbox = false until this passes.',
          );
        });
      }
    }
  }
}
