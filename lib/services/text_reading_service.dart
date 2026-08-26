import 'dart:io';
import 'dart:typed_data';

import 'package:flutter/foundation.dart';
import 'package:tesseract_ocr/tesseract_ocr.dart';
import 'package:tesseract_ocr/ocr_engine_config.dart';

/// Result of a text-reading attempt.
class TextReadingResult {
  final String text;
  final bool hasText;

  const TextReadingResult({required this.text, required this.hasText});

  factory TextReadingResult.empty() =>
      const TextReadingResult(text: '', hasText: false);
}

/// Handles the "read text in this photo" flow — fully on-device, no server.
///
/// Uses Tesseract OCR (via the `tesseract_ocr` plugin) with combined
/// Arabic + English trained data ('ara+eng'), matching the FYP proposal's
/// requirement to support both languages, without depending on a manually
/// started server (per supervisor's requirement — see [face_server] which is
/// still used for face recognition only, not for OCR).
///
/// Trade-off vs. the original PaddleOCR-on-server design: Tesseract is
/// generally less accurate than PaddleOCR, especially on stylized or
/// low-contrast text, and a bit slower per-frame. That's the accepted cost
/// of removing the server dependency — worth calling out explicitly if asked
/// why the proposal's PaddleOCR isn't the one actually running.
class TextReadingService {
  static final TextReadingService instance = TextReadingService._();
  TextReadingService._();

  static const String _language = 'ara+eng';

  /// Read text from a captured photo.
  Future<TextReadingResult> readText(Uint8List imageBytes) async {
    File? tempFile;
    try {
      tempFile = await _writeTempFile(imageBytes);

      final config = OCRConfig(
        language: _language,
        engine: OCREngine.tesseract,
      );

      final extracted = await TesseractOcr.extractText(
        tempFile.path,
        config: config,
      );

      final text = extracted.trim();
      return TextReadingResult(text: text, hasText: text.isNotEmpty);
    } catch (e) {
      debugPrint('[TextReading] OCR error: $e');
      return TextReadingResult.empty();
    } finally {
      if (tempFile != null) {
        try {
          await tempFile.delete();
        } catch (_) {}
      }
    }
  }

  /// Writes image bytes to a throwaway file in the system temp directory.
  /// Tesseract's plugin API takes a file path, not raw bytes.
  Future<File> _writeTempFile(Uint8List bytes) async {
    final file = File(
      '${Directory.systemTemp.path}/envision_ocr_${DateTime.now().millisecondsSinceEpoch}.jpg',
    );
    return file.writeAsBytes(bytes, flush: true);
  }
}
