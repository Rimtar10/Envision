import 'dart:math';
import 'dart:ui';

/// Singleton to record size related data
class ScreenParams {
  // by defaults sets to zero
  static Size screenSize = Size.zero;
  static Size? previewSize;

  static double get previewRatio {
    final size = previewSize;
    if (size == null) return 1;
    return max(size.height, size.width) / min(size.height, size.width);
  }

  static Size get screenPreviewSize =>
      Size(screenSize.width, screenSize.width * previewRatio);
}
