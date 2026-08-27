import 'dart:math';
import 'dart:typed_data';
import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:google_mlkit_face_detection/google_mlkit_face_detection.dart';
import 'package:image/image.dart' as img;
import 'package:tensorflow_demo/models/screen_params.dart';
import 'package:tensorflow_demo/services/api_service.dart';
import 'package:tensorflow_demo/services/face_database_service.dart';
import 'package:tflite_flutter/tflite_flutter.dart';

/// Result of a single face recognition attempt.
class FaceResult {
  final Rect boundingBox;
  final String? name; // null = unknown
  final double confidence;
  final bool fromServer; // true = DeepFace, false = MobileFaceNet

  const FaceResult({
    required this.boundingBox,
    required this.name,
    required this.confidence,
    this.fromServer = false,
  });

  bool get isKnown => name != null && name != 'Unknown';
}

/// Handles real-time face detection (ML Kit) + recognition
/// Online: DeepFace server (ArcFace, high accuracy)
/// Offline: MobileFaceNet TFLite (fallback)
class FaceRecognitionService {
  static final FaceRecognitionService instance = FaceRecognitionService._();
  FaceRecognitionService._();

  // ML Kit face detector
  FaceDetector? _faceDetector;

  // MobileFaceNet TFLite interpreter (offline fallback)
  Interpreter? _interpreter;

  bool _isInitialized = false;
  bool _isProcessing = false;

  // Offline database
  List<RegisteredFace> _registeredFaces = [];

  // Threshold for offline matching
  static const double _threshold = 1.3;

  // Cooldown
  final Map<String, DateTime> _lastAnnounced = {};
  static const _announcementCooldown = Duration(seconds: 5);

  // Server check timer
  DateTime? _lastServerCheck;
  static const _serverCheckInterval = Duration(seconds: 10);

  bool get isInitialized => _isInitialized;

  // ── Initialization ────────────────────────────────────────────────────────

  Future<void> initialize() async {
    if (_isInitialized) return;

    // Initialize ML Kit face detector
    _faceDetector = FaceDetector(
      options: FaceDetectorOptions(
        performanceMode: FaceDetectorMode.fast,
        enableLandmarks: false,
        enableContours: false,
        enableTracking: false,
        enableClassification: false,
      ),
    );

    // Load MobileFaceNet TFLite (offline fallback)
    try {
      _interpreter = await Interpreter.fromAsset('assets/mobilefacenet.tflite');
      debugPrint('[FaceRecognition] MobileFaceNet loaded successfully');
    } catch (e) {
      debugPrint('[FaceRecognition] Failed to load MobileFaceNet: $e');
    }

    await refreshDatabase();
    _isInitialized = true;
    debugPrint('[FaceRecognition] Initialized successfully');

    // Check server status in background
    _checkServerInBackground();
  }

  void _checkServerInBackground() {
    Future.delayed(const Duration(seconds: 2), () async {
      final online = await ApiService.instance.checkStatus();
      debugPrint('[FaceRecognition] Server online: $online');
    });
  }

  Future<void> refreshDatabase() async {
    _registeredFaces = await FaceDatabaseService.instance.getAllFaces();
    debugPrint(
        '[FaceRecognition] Loaded ${_registeredFaces.length} registered faces');
  }

  // ── Server Status ─────────────────────────────────────────────────────────

  Future<bool> _isServerAvailable() async {
    final now = DateTime.now();
    if (_lastServerCheck != null &&
        now.difference(_lastServerCheck!) < _serverCheckInterval) {
      return ApiService.instance.isOnline;
    }
    _lastServerCheck = now;
    return await ApiService.instance.checkStatus();
  }

  bool get isServerOnline => ApiService.instance.isOnline;

  // ── Frame Processing ──────────────────────────────────────────────────────

  Future<List<FaceResult>> processFrame(CameraImage cameraImage) async {
    if (!_isInitialized || _isProcessing) return [];
    if (_faceDetector == null) return [];
    _isProcessing = true;

    try {
      // Convert camera image
      final image = _convertCameraImage(cameraImage);
      if (image == null) return [];

      // Build ML Kit input image
      final inputImage = _buildInputImage(cameraImage);
      if (inputImage == null) return [];

      // Detect faces
      final faces = await _faceDetector!.processImage(inputImage);
      if (faces.isEmpty) return [];

      debugPrint('[FaceRecognition] Detected ${faces.length} faces');

      // Check if server is available
      final serverAvailable = await _isServerAvailable();

      final results = <FaceResult>[];

      // Scale bounding boxes to screen
      final scaleX = ScreenParams.screenSize.width / cameraImage.width;
      final scaleY = ScreenParams.screenSize.height / cameraImage.height;

      for (final face in faces) {
        final boundingBox = face.boundingBox;

        final screenBox = Rect.fromLTRB(
          boundingBox.left * scaleX,
          boundingBox.top * scaleY,
          boundingBox.right * scaleX,
          boundingBox.bottom * scaleY,
        );

        String? matchedName;
        double matchConfidence = 0.0;
        bool fromServer = false;

        if (serverAvailable) {
          // ── Online: use DeepFace server ──────────────────────────────────
          try {
            final jpegBytes = Uint8List.fromList(img.encodeJpg(image));
            final result = await ApiService.instance.recognize(jpegBytes);
            if (result != null && result.isKnown) {
              matchedName = result.name;
              matchConfidence = result.confidence;
              fromServer = true;
              debugPrint(
                  '[FaceRecognition] Server: ${result.name} (${result.confidence})');
            }
          } catch (e) {
            debugPrint('[FaceRecognition] Server recognition error: $e');
          }
        } else {
          // ── Offline: use MobileFaceNet ───────────────────────────────────
          if (_interpreter != null) {
            try {
              final x = boundingBox.left.clamp(0, image.width - 1).toInt();
              final y = boundingBox.top.clamp(0, image.height - 1).toInt();
              final w = boundingBox.width
                  .clamp(1, image.width - x.toDouble())
                  .toInt();
              final h = boundingBox.height
                  .clamp(1, image.height - y.toDouble())
                  .toInt();

              final cropped =
                  img.copyCrop(image, x: x, y: y, width: w, height: h);
              final resized = img.copyResizeCropSquare(cropped, size: 112);
              final embedding = _getEmbedding(resized);
              final match = _findMatch(embedding);

              if (match != null) {
                matchedName = match.name;
                matchConfidence = match.distance;
                fromServer = false;
                debugPrint(
                    '[FaceRecognition] Offline: ${match.name} (${match.distance})');
              }
            } catch (e) {
              debugPrint('[FaceRecognition] Offline recognition error: $e');
            }
          }
        }

        results.add(FaceResult(
          boundingBox: screenBox,
          name: matchedName,
          confidence: matchConfidence,
          fromServer: fromServer,
        ));
      }

      return results;
    } catch (e) {
      debugPrint('[FaceRecognition] Error: $e');
      return [];
    } finally {
      _isProcessing = false;
    }
  }

  // ── Registration ──────────────────────────────────────────────────────────

  /// Register a face — tries server first, falls back to local
  Future<bool> registerFaceFromBytes(Uint8List imageBytes, String name) async {
    final serverAvailable = await ApiService.instance.forceCheckStatus();

    if (serverAvailable) {
      // Register on server
      debugPrint('[FaceRecognition] Registering on server: $name');
      final success = await ApiService.instance.register(name, [imageBytes]);
      if (success) {
        debugPrint('[FaceRecognition] Server registration successful: $name');
        return true;
      }
    }

    // Fall back to local MobileFaceNet registration
    debugPrint('[FaceRecognition] Registering locally: $name');
    return await _registerLocally(imageBytes, name);
  }

  Future<bool> _registerLocally(Uint8List imageBytes, String name) async {
    try {
      if (_interpreter == null) return false;

      final image = img.decodeImage(imageBytes);
      if (image == null) return false;

      final resized = img.copyResizeCropSquare(image, size: 112);
      final embedding = _getEmbedding(resized);

      await FaceDatabaseService.instance.registerFace(name, embedding);
      await refreshDatabase();
      debugPrint('[FaceRecognition] Registered locally: $name');
      return true;
    } catch (e) {
      debugPrint('[FaceRecognition] Local registration error: $e');
      return false;
    }
  }

  // ── MobileFaceNet (offline) ───────────────────────────────────────────────

  List<double> _getEmbedding(img.Image faceImage) {
    final input = _imageToFloat32(faceImage, 112, 128.0, 128.0);
    final inputReshaped = input.reshape([1, 112, 112, 3]);
    final output = List.generate(1, (_) => List.filled(192, 0.0));
    _interpreter!.run(inputReshaped, output);
    return List<double>.from(output[0]);
  }

  Float32List _imageToFloat32(
      img.Image image, int inputSize, double mean, double std) {
    final convertedBytes = Float32List(1 * inputSize * inputSize * 3);
    final buffer = Float32List.view(convertedBytes.buffer);
    int pixelIndex = 0;
    for (int i = 0; i < inputSize; i++) {
      for (int j = 0; j < inputSize; j++) {
        final pixel = image.getPixel(j, i);
        buffer[pixelIndex++] = (pixel.r - mean) / std;
        buffer[pixelIndex++] = (pixel.g - mean) / std;
        buffer[pixelIndex++] = (pixel.b - mean) / std;
      }
    }
    return convertedBytes.buffer.asFloat32List();
  }

  _MatchResult? _findMatch(List<double> embedding) {
    if (_registeredFaces.isEmpty) return null;
    double minDistance = 999.0;
    RegisteredFace? bestMatch;
    for (final registered in _registeredFaces) {
      final distance = _euclideanDistance(embedding, registered.embedding);
      if (distance < minDistance) {
        minDistance = distance;
        bestMatch = registered;
      }
    }
    if (bestMatch != null && minDistance <= _threshold) {
      return _MatchResult(name: bestMatch.name, distance: minDistance);
    }
    return null;
  }

  double _euclideanDistance(List<double> a, List<double> b) {
    double sum = 0.0;
    final len = min(a.length, b.length);
    for (int i = 0; i < len; i++) {
      sum += pow(a[i] - b[i], 2);
    }
    return sqrt(sum);
  }

  // ── Cooldown ──────────────────────────────────────────────────────────────

  bool shouldAnnounce(String name) {
    final last = _lastAnnounced[name];
    final now = DateTime.now();
    if (last == null || now.difference(last) > _announcementCooldown) {
      _lastAnnounced[name] = now;
      return true;
    }
    return false;
  }

  // ── Helpers ───────────────────────────────────────────────────────────────

  InputImage? _buildInputImage(CameraImage cameraImage) {
    try {
      final WriteBuffer allBytes = WriteBuffer();
      for (final plane in cameraImage.planes) {
        allBytes.putUint8List(plane.bytes);
      }
      final bytes = allBytes.done().buffer.asUint8List();
      final metadata = InputImageMetadata(
        size: Size(cameraImage.width.toDouble(), cameraImage.height.toDouble()),
        rotation: InputImageRotation.rotation90deg,
        format: InputImageFormat.nv21,
        bytesPerRow: cameraImage.planes[0].bytesPerRow,
      );
      return InputImage.fromBytes(bytes: bytes, metadata: metadata);
    } catch (e) {
      return null;
    }
  }

  img.Image? _convertCameraImage(CameraImage cameraImage) {
    try {
      img.Image? image;
      if (cameraImage.format.group == ImageFormatGroup.yuv420) {
        image = _convertYUV420(cameraImage);
      } else if (cameraImage.format.group == ImageFormatGroup.bgra8888) {
        image = img.Image.fromBytes(
          width: cameraImage.width,
          height: cameraImage.height,
          bytes: cameraImage.planes[0].bytes.buffer,
          format: img.Format.uint8,
          numChannels: 4,
        );
      }
      if (image != null) {
        image = img.copyRotate(image, angle: 90);
      }
      return image;
    } catch (e) {
      return null;
    }
  }

  img.Image _convertYUV420(CameraImage cameraImage) {
    final width = cameraImage.width;
    final height = cameraImage.height;
    final result = img.Image(width: width, height: height);
    final yPlane = cameraImage.planes[0].bytes;
    final uPlane = cameraImage.planes[1].bytes;
    final vPlane = cameraImage.planes[2].bytes;
    final uvRowStride = cameraImage.planes[1].bytesPerRow;
    final uvPixelStride = cameraImage.planes[1].bytesPerPixel ?? 1;
    for (int y = 0; y < height; y++) {
      for (int x = 0; x < width; x++) {
        final uvIndex = uvPixelStride * (x ~/ 2) + uvRowStride * (y ~/ 2);
        final yVal = yPlane[y * width + x];
        final uVal = uPlane[uvIndex];
        final vVal = vPlane[uvIndex];
        final r = (yVal + 1.370705 * (vVal - 128)).clamp(0, 255).toInt();
        final g = (yVal - 0.337633 * (uVal - 128) - 0.698001 * (vVal - 128))
            .clamp(0, 255)
            .toInt();
        final b = (yVal + 1.732446 * (uVal - 128)).clamp(0, 255).toInt();
        result.setPixelRgb(x, y, r, g, b);
      }
    }
    return result;
  }

  void dispose() {
    _faceDetector?.close();
    _interpreter?.close();
    _isInitialized = false;
  }
}

class _MatchResult {
  final String name;
  final double distance;
  const _MatchResult({required this.name, required this.distance});
}
