import 'package:flutter/cupertino.dart';
import 'package:tensorflow_demo/models/screen_params.dart';

/// Represents the recognition output from the model.
class DetectedObjectDm {
  const DetectedObjectDm({
    required this.label,
    required this.score,
    required this.location,
    this.yStretch = 1.0,
    this.modelSize = 640.0,
  });

  /// Label of the result
  final String label;

  /// Confidence [0.0, 1.0]
  final num score;

  /// Bounding box in MODEL space: 0..[modelSize] on both axes, with the
  /// letterbox padding already removed (so the content fills the full square).
  final Rect location;

  /// Vertical stretch factor introduced by removing the letterbox padding.
  ///
  /// When padding lands on the y axis, undoing it multiplies every box height
  /// by `modelSize / contentHeight`. Distance estimation must divide it back
  /// out, otherwise objects are reported as closer than they really are — the
  /// error is silent, device-dependent, and always in the dangerous direction.
  final double yStretch;

  /// The square input edge of the model that produced this detection.
  /// Carried per-detection so a 640 model and a 448 model can coexist.
  final double modelSize;

  /// Bounding box rectangle in screen coordinates, for rendering.
  Rect get renderLocation {
    final previewSize = ScreenParams.screenPreviewSize;
    final double scaleX = previewSize.width / modelSize;
    final double scaleY = previewSize.height / modelSize;
    return Rect.fromLTWH(
      location.left * scaleX,
      location.top * scaleY,
      location.width * scaleX,
      location.height * scaleY,
    );
  }

  @override
  String toString() =>
      'DetectedObjectDm(label: $label, score: $score, location: $location)';
}
