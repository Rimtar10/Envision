import 'dart:io';

import 'package:flutter/foundation.dart';
import 'package:google_mlkit_text_recognition/google_mlkit_text_recognition.dart';
import 'package:tensorflow_demo/services/api_service.dart';

/// Where a [TextReadingResult] came from.
enum TextReadingSource {
  /// PaddleOCR on the Flask server — supports Arabic + English.
  server,

  /// Google ML Kit on-device recognizer — English (Latin script) only.
  /// Used as a fallback when the server is unreachable.
  onDeviceFallback,
}

/// Normalized result of a text-reading attempt, regardless of which
/// engine produced it.
class TextReadingResult {
  final String text;
  final bool hasText;
  final TextReadingSource source;

  const TextReadingResult({
    required this.text,
    required this.hasText,
    required this.source,
  });

  factory TextReadingResult.empty(TextReadingSource source) =>
      TextReadingResult(text: '', hasText: false, source: source);
}

/// Handles the "read text in this photo" flow.
///
/// Online: sends the image to the PaddleOCR server (`/ocr`), which supports
///   both Arabic and English — matches the FYP proposal's OCR module spec.
/// Offline: falls back to Google ML Kit's on-device text recognizer. ML Kit's
///   on-device recognizer only supports Latin-based scripts, so this fallback
///   is English-only — Arabic text will not be read while offline.
class TextReadingService {
  static final TextReadingService instance = TextReadingService._();
  TextReadingService._();

  final TextRecognizer _onDeviceRecognizer = TextRecognizer(
    script: TextRecognitionScript.latin,
  );

  /// Read text from a captured photo.
  ///
  /// Tries the server first (best accuracy, bilingual). Falls back to the
  /// on-device recognizer if the server can't be reached, so the feature
  /// still works (English only) without a network/PC nearby.
  Future<TextReadingResult> readText(Uint8List imageBytes) async {
    final serverAvailable = await ApiService.instance.forceCheckStatus();

    if (serverAvailable) {
      try {
        final result = await ApiService.instance.recognizeText(imageBytes);
        if (result != null && result.hasText) {
          return TextReadingResult(
            text: result.text,
            hasText: true,
            source: TextReadingSource.server,
          );
        }
        if (result != null && result.status == 'no_text_detected') {
          // Server is online and confidently found nothing — trust it,
          // don't bother falling back to the (weaker) on-device pass.
          return TextReadingResult.empty(TextReadingSource.server);
        }
      } catch (e) {
        debugPrint('[TextReading] Server OCR error, falling back: $e');
      }
    }

    return _readTextOnDevice(imageBytes);
  }

  Future<TextReadingResult> _readTextOnDevice(Uint8List imageBytes) async {
    File? tempFile;
    try {
      tempFile = await _writeTempFile(imageBytes);
      final inputImage = InputImage.fromFilePath(tempFile.path);
      final recognized = await _onDeviceRecognizer.processImage(inputImage);
      final text = recognized.text.trim();
      return TextReadingResult(
        text: text,
        hasText: text.isNotEmpty,
        source: TextReadingSource.onDeviceFallback,
      );
    } catch (e) {
      debugPrint('[TextReading] On-device OCR error: $e');
      return TextReadingResult.empty(TextReadingSource.onDeviceFallback);
    } finally {
      if (tempFile != null) {
        try {
          await tempFile.delete();
        } catch (e) {
          // Was an empty catch. Swallowing an error here is how this app
          // goes quiet without anyone noticing -- and quiet reads as
          // "path is clear". Logged, not handled: behaviour is unchanged.
          debugPrint('[text reading] ignored: $e');
        }
      }
    }
  }

  /// Writes image bytes to a throwaway file in the system temp directory.
  /// ML Kit's static-image API (`InputImage.fromFilePath`) needs a file path
  /// rather than raw bytes for already-encoded (jpg/png) images.
  Future<File> _writeTempFile(Uint8List bytes) async {
    final file = File(
      '${Directory.systemTemp.path}/envision_ocr_${DateTime.now().millisecondsSinceEpoch}.jpg',
    );
    return file.writeAsBytes(bytes, flush: true);
  }

  void dispose() {
    _onDeviceRecognizer.close();
  }
}
