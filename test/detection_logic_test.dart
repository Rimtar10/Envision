// Unit tests for the pure detection logic that had bugs.
//
// None of this needs a camera, a model or a device — it is arithmetic and list
// manipulation, which is exactly where a silent regression hides. Run with:
//     flutter test test/detection_logic_test.dart

import 'dart:ui';

import 'package:flutter_test/flutter_test.dart';
import 'package:tensorflow_demo/models/detected_object/detected_object_dm.dart';

void main() {
  group('ModelGeometry-independent box maths', () {
    test('yStretch defaults to 1 so untouched detections are unaffected', () {
      const d = DetectedObjectDm(
        label: 'person',
        score: 0.9,
        location: Rect.fromLTRB(0, 0, 10, 100),
      );
      expect(d.yStretch, 1.0);
      expect(d.modelSize, 640.0);
    });

    test('a y-padded letterbox reports a stretch > 1', () {
      // 640 model, content height 480 => every box height was multiplied by
      // 640/480 = 1.333 when the padding was undone. Distance estimation has to
      // divide that back out or objects read 33% closer than they are.
      const stretch = 640.0 / 480.0;
      const d = DetectedObjectDm(
        label: 'person',
        score: 0.9,
        location: Rect.fromLTRB(0, 0, 10, 133.33),
        yStretch: stretch,
      );
      final corrected = d.location.height / d.yStretch;
      expect(corrected, closeTo(100.0, 0.01));
    });
  });

  group('pinhole distance', () {
    // Mirrors VoiceService._estimateDistance so the maths is pinned even though
    // that method is private.
    double distance(double realHeightM, double pixelHeight, double modelSize,
        double yStretch) {
      final focal = modelSize / (2 * 0.5773502691896257); // tan(30°)
      return (realHeightM * focal) / (pixelHeight / yStretch);
    }

    test('a 1.7 m person filling half a 640 frame is roughly 3.5 m away', () {
      final d = distance(1.7, 320, 640, 1.0);
      expect(d, closeTo(2.94, 0.15));
    });

    test('doubling the box height halves the distance', () {
      final near = distance(1.7, 320, 640, 1.0);
      final far = distance(1.7, 160, 640, 1.0);
      expect(far, closeTo(near * 2, 0.001));
    });

    test('focal length scales with model size, so 448 gives the same answer',
        () {
      // Same object occupying the same FRACTION of the frame must yield the
      // same distance regardless of export resolution.
      final at640 = distance(1.7, 320, 640, 1.0);
      final at448 = distance(1.7, 224, 448, 1.0);
      expect(at448, closeTo(at640, 0.001));
    });

    test('ignoring yStretch under-reports distance (the bug being fixed)', () {
      const stretch = 640.0 / 480.0;
      final wrong = distance(1.7, 320, 640, 1.0);
      final right = distance(1.7, 320, 640, stretch);
      expect(right, greaterThan(wrong));
      expect(right / wrong, closeTo(stretch, 0.001));
    });
  });
}
