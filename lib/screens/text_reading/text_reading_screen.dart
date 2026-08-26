import 'dart:typed_data';

import 'package:flutter/material.dart';
import 'package:tensorflow_demo/services/snackbar_service.dart';
import 'package:tensorflow_demo/services/text_reading_service.dart';
import 'package:tensorflow_demo/services/voice_service.dart';

/// Shows a captured photo alongside the text extracted from it, and speaks
/// the result aloud. Mirrors [PhotoAnalyzedScreen]'s capture-then-analyze
/// shape, but for reading text instead of detecting objects.
class TextReadingScreen extends StatefulWidget {
  const TextReadingScreen({required this.imageBytes, super.key});

  final Uint8List imageBytes;

  @override
  State<TextReadingScreen> createState() => _TextReadingScreenState();
}

class _TextReadingScreenState extends State<TextReadingScreen> {
  TextReadingResult? _result;
  bool _isLoading = true;

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addPostFrameCallback((_) {
      SnackBarService.show('Reading text...');
      _readText();
    });
  }

  Future<void> _readText() async {
    final result = await TextReadingService.instance.readText(
      widget.imageBytes,
    );
    SnackBarService.remove();
    if (!mounted) return;
    setState(() {
      _result = result;
      _isLoading = false;
    });
    _announceResult(result);
  }

  void _announceResult(TextReadingResult result) {
    if (!result.hasText) {
      VoiceService.instance.speak('No text found in the photo.');
      return;
    }
    VoiceService.instance.speak(result.text);
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Read Text'),
        actions: [
          if (_result?.hasText ?? false)
            IconButton(
              tooltip: 'Read again',
              icon: const Icon(Icons.volume_up),
              onPressed: () => _announceResult(_result!),
            ),
        ],
      ),
      body: ListView(
        padding: const EdgeInsets.all(16),
        children: [
          ClipRRect(
            borderRadius: BorderRadius.circular(12),
            child: Image.memory(widget.imageBytes),
          ),
          const SizedBox(height: 16),
          if (_isLoading)
            const Padding(
              padding: EdgeInsets.symmetric(vertical: 32),
              child: Center(child: CircularProgressIndicator()),
            )
          else if (_result == null || !_result!.hasText)
            const Padding(
              padding: EdgeInsets.symmetric(vertical: 16),
              child: Text(
                'No text detected in this photo.',
                style: TextStyle(fontSize: 16),
              ),
            )
          else ...[
            const Text(
              'Extracted Text',
              style: TextStyle(fontSize: 20, fontWeight: FontWeight.w600),
            ),
            const SizedBox(height: 12),
            Container(
              width: double.infinity,
              padding: const EdgeInsets.all(16),
              decoration: BoxDecoration(
                color: Colors.black12,
                borderRadius: BorderRadius.circular(12),
              ),
              child: SelectableText(
                _result!.text,
                style: const TextStyle(fontSize: 16, height: 1.4),
              ),
            ),
          ],
        ],
      ),
    );
  }
}
