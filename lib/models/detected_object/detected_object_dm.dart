import 'package:flutter/cupertino.dart';
import 'package:tensorflow_demo/models/screen_params.dart';
import 'package:tensorflow_demo/values/app_constants.dart';

/// Represents the recognition output from the model
class DetectedObjectDm {
  const DetectedObjectDm({
    required this.label,
    required this.score,
    required this.location,
  });

  /// Label of the result
  final String label;

  /// Confidence [0.0, 1.0]
  final num score;

  /// Location of bounding box rect
  ///
  /// The rectangle corresponds to the raw input image
  /// passed for inference
  final Rect location;

  /// Returns bounding box rectangle corresponding to the
  /// displayed image on screen
  ///
  /// This is the actual location where rectangle is rendered on
  /// the screen
  Rect get renderLocation {
    final previewSize = ScreenParams.screenPreviewSize;
    final double scaleX =
        previewSize.width / AppConstants.ssdCompatibleImageWidth;
    final double scaleY =
        previewSize.height / AppConstants.ssdCompatibleImageHeight;
    return Rect.fromLTWH(
      location.left * scaleX,
      location.top * scaleY,
      location.width * scaleX,
      location.height * scaleY,
    );
  }

  @override
  String toString() {
    return 'DetectedObjectDm(label: $label, score: $score, location: $location)';
  }
}
