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
    final bytes = cameraImage.planes[0].bytes;
    return decodeImage(bytes);
  }

  static Image _convertNV21ToRGBImage(CameraImage cameraImage) {
    final width = cameraImage.width;
    final height = cameraImage.height;
    final yuvBytes = cameraImage.planes[0].bytes;
    final vuBytes = cameraImage.planes[1].bytes;

    final rgbBytes = Uint8List(width * height * 3);
    int dst = 0;

    for (int y = 0; y < height; y++) {
      final uvRow = y >> 1;
      for (int x = 0; x < width; x++) {
        final yVal = yuvBytes[y * width + x];
        final uvIdx = (uvRow * (width ~/ 2) + (x >> 1)) * 2;
        final v = vuBytes[uvIdx] - 128;     // V first in NV21
        final u = vuBytes[uvIdx + 1] - 128; // U second

        rgbBytes[dst++] = (yVal + ((v * 1436) >> 10)).clamp(0, 255);
        rgbBytes[dst++] = (yVal - ((u * 352) >> 10) - ((v * 731) >> 10)).clamp(0, 255);
        rgbBytes[dst++] = (yVal + ((u * 1814) >> 10)).clamp(0, 255);
      }
    }

    return Image.fromBytes(
      width: width,
      height: height,
      bytes: rgbBytes.buffer,
      numChannels: 3,
      order: ChannelOrder.rgb,
    );
  }

  // Converts a [CameraImage] in BGRA888 format to [Image] in RGB format
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

    // Direct buffer fill with fixed-point integer arithmetic —
    // avoids setPixelRgb overhead and floating-point multiplications.
    final rgbBytes = Uint8List(imageWidth * imageHeight * 3);
    int dst = 0;

    for (int h = 0; h < imageHeight; h++) {
      final uvh = h >> 1;
      for (int w = 0; w < imageWidth; w++) {
        final uvw = w >> 1;
        final yVal = yBuffer[h * yRowStride + w * yPixelStride];
        final uvIndex = uvh * uvRowStride + uvw * uvPixelStride;
        final u = uBuffer[uvIndex] - 128;
        final v = vBuffer[uvIndex] - 128;

        // Fixed-point YUV→RGB (10-bit shift): replaces float multiply+round
        rgbBytes[dst++] = (yVal + ((v * 1436) >> 10)).clamp(0, 255);
        rgbBytes[dst++] = (yVal - ((u * 352) >> 10) - ((v * 731) >> 10)).clamp(0, 255);
        rgbBytes[dst++] = (yVal + ((u * 1814) >> 10)).clamp(0, 255);
      }
    }

    return Image.fromBytes(
      width: imageWidth,
      height: imageHeight,
      bytes: rgbBytes.buffer,
      numChannels: 3,
      order: ChannelOrder.rgb,
    );
  }
}
