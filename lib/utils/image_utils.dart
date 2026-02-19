import 'dart:typed_data';

import 'package:camera/camera.dart';
import 'package:image/image.dart';

// ImageUtils
class ImageUtils {
  static Image? convertCameraImageToImage(CameraImage cameraImage) {
    return switch (cameraImage.format.group) {
      ImageFormatGroup.yuv420 => _convertYUV420ToRGBImage(cameraImage),
      ImageFormatGroup.bgra8888 => _convertBGRA8888ToRGBImage(cameraImage),
      ImageFormatGroup.jpeg => _convertJPEGToImage(cameraImage),
      ImageFormatGroup.nv21 => _convertNV21ToRGBImage(cameraImage),
      ImageFormatGroup.unknown => null,
    };
  }

  static Image? _convertJPEGToImage(CameraImage cameraImage) {
    // Extract the bytes from the CameraImage
    // first plane is responsible for holding all the image data
    final bytes = cameraImage.planes[0].bytes;

    // Create a new Image instance from the JPEG bytes
    final image = decodeImage(bytes);

    return image;
  }

  static Image _convertNV21ToRGBImage(CameraImage cameraImage) {
    // Extract the bytes from the CameraImage
    final yuvBytes = cameraImage.planes[0].bytes;
    final vuBytes = cameraImage.planes[1].bytes;

    // Create a new Image instance
    final image = Image(
      width: cameraImage.width,
      height: cameraImage.height,
    );

    // Convert NV21 to RGB
    _convertNV21ToRGB(
      yuvBytes,
      vuBytes,
      cameraImage.width,
      cameraImage.height,
      image,
    );

    return image;
  }

  static void _convertNV21ToRGB(
    Uint8List yuvBytes,
    Uint8List vuBytes,
    int width,
    int height,
    Image image,
  ) {
    // Conversion logic from NV21 to RGB
    // ...

    // Example conversion logic using the `imageLib` package
    // This is just a placeholder and may not be the most efficient method
    for (var y = 0; y < height; y++) {
      for (var x = 0; x < width; x++) {
        final yIndex = y * width + x;
        final uvIndex = (y ~/ 2) * (width ~/ 2) + (x ~/ 2);

        final yValue = yuvBytes[yIndex];
        final vValue = vuBytes[uvIndex * 2];     // V comes first in NV21
        final uValue = vuBytes[uvIndex * 2 + 1]; // U comes second

        // Convert YUV to RGB
        final r = yValue + 1.402 * (vValue - 128);
        final g =
            yValue - 0.344136 * (uValue - 128) - 0.714136 * (vValue - 128);
        final b = yValue + 1.772 * (uValue - 128);

        // Set the RGB pixel values in the Image instance
        image.setPixelRgba(x, y, r.toInt(), g.toInt(), b.toInt(), 255);
      }
    }
  }

  // Converts a [CameraImage] in BGRA888 format to [imageLib.Image] in RGB format
  static Image _convertBGRA8888ToRGBImage(CameraImage cameraImage) {
    final firstPlane = cameraImage.planes[0];
    return Image.fromBytes(
      width: firstPlane.width!,
      height: firstPlane.height!,
      bytes: firstPlane.bytes.buffer,
      order: ChannelOrder.bgra,
    );
  }

  static Image _convertYUV420ToRGBImage(CameraImage cameraImage) {
  final imageWidth = cameraImage.width;
  final imageHeight = cameraImage.height;

  final yPlane = cameraImage.planes[0];
  final uPlane = cameraImage.planes[1];
  final vPlane = cameraImage.planes[2];

  final yBuffer = yPlane.bytes;
  final uBuffer = uPlane.bytes;
  final vBuffer = vPlane.bytes;

  final int yRowStride = yPlane.bytesPerRow;
  final int yPixelStride = yPlane.bytesPerPixel!;

  final int uvRowStride = uPlane.bytesPerRow;
  final int uvPixelStride = uPlane.bytesPerPixel!;

  final image = Image(width: imageWidth, height: imageHeight);

  for (int h = 0; h < imageHeight; h++) {
    int uvh = (h / 2).floor();

    for (int w = 0; w < imageWidth; w++) {
      int uvw = (w / 2).floor();

      final yIndex = (h * yRowStride) + (w * yPixelStride);
      final int y = yBuffer[yIndex];

      final int uvIndex = (uvh * uvRowStride) + (uvw * uvPixelStride);
      
      // CRITICAL FIX: Subtract 128 to center U/V values around 0
      final int u = uBuffer[uvIndex] - 128;
      final int v = vBuffer[uvIndex] - 128;

      // Standard YUV to RGB conversion
      int r = (y + 1.402 * v).round();
      int g = (y - 0.344136 * u - 0.714136 * v).round();
      int b = (y + 1.772 * u).round();

      r = r.clamp(0, 255);
      g = g.clamp(0, 255);
      b = b.clamp(0, 255);

      image.setPixelRgb(w, h, r, g, b);
    }
  }
  return image;
}
}