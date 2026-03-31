import 'package:flutter_tts/flutter_tts.dart';
import 'package:speech_to_text/speech_to_text.dart';
import 'dart:async';
import 'dart:ui';
import 'package:tensorflow_demo/models/detected_object/detected_object_dm.dart';

class VoiceService {
  static final VoiceService instance = VoiceService._();
  VoiceService._();

  final FlutterTts _tts = FlutterTts();
  final SpeechToText _stt = SpeechToText();

  // Track what we've announced recently to avoid repetition
  final Map<String, DateTime> _lastAnnounced = {};

  // Cooldown period (don't announce same object within 3 seconds)
  static const _cooldownSeconds = 3;

  // Track if TTS is currently speaking
  bool _isSpeaking = false;

  // Single pending message — replaces old queue to prevent speech pile-up.
  // If TTS is busy, the latest message waits here; older ones are discarded.
  String? _pendingMessage;

  // Callback to notify UI when listening state changes
  void Function(bool isListening)? onListeningStateChanged;

  // Store current detections for "what's in front of me" queries
  List<DetectedObjectDm> _currentDetections = [];

  bool get isListening => _stt.isListening;

  Future<void> initialize() async {
    // Configure TTS
    await _tts.setLanguage("en-US");
    await _tts.setSpeechRate(0.5);
    await _tts.setVolume(1.0);
    await _tts.setPitch(1.0);

    // All three handlers reset _isSpeaking so the pending message can play
    _tts.setCompletionHandler(_onTtsDone);
    _tts.setCancelHandler(_onTtsDone);
    _tts.setErrorHandler((_) => _onTtsDone());

    // Initialize STT
    final sttAvailable = await _stt.initialize(
      onError: (error) {
        print('STT Error: $error');
        onListeningStateChanged?.call(false);
      },
      onStatus: (status) {
        print('STT Status: $status');
        // "notListening" / "done" both mean listening has stopped
        if (status == 'notListening' || status == 'done') {
          onListeningStateChanged?.call(false);
        }
      },
    );
    print('STT initialized: $sttAvailable');
  }

  void _onTtsDone() {
    _isSpeaking = false;
    _processPending();
  }

  /// Main method: Announce detections (called every frame)
  Future<void> announceDetections(List<DetectedObjectDm> detections) async {
    if (detections.isEmpty) return;

    // Don't queue TTS announcements while user is speaking
    if (_stt.isListening) return;

    _currentDetections = detections;

    final nearest = _findNearestObject(detections);
    if (nearest == null) return;

    final distance = _estimateDistance(nearest.label, nearest.location);
    final distanceCategory = _getDistanceCategory(distance);

    final key = '${nearest.label}_$distanceCategory';
    final lastTime = _lastAnnounced[key];
    final now = DateTime.now();

    if (lastTime != null &&
        now.difference(lastTime).inSeconds < _cooldownSeconds) {
      return;
    }

    _lastAnnounced[key] = now;

    final message = distance > 0
        ? '${nearest.label} $distanceCategory'
        : nearest.label;

    _speak(message);
  }

  /// Find nearest object based on bounding box area (larger = closer)
  DetectedObjectDm? _findNearestObject(List<DetectedObjectDm> detections) {
    if (detections.isEmpty) return null;
    detections.sort((a, b) {
      final areaA = a.location.width * a.location.height;
      final areaB = b.location.width * b.location.height;
      return areaB.compareTo(areaA);
    });
    return detections.first;
  }

  /// Estimate distance using pinhole camera model.
  /// bounding box coords are in model space (0–640).
  /// Focal length ~500px is more accurate for typical phone cameras at 640px.
  double _estimateDistance(String label, Rect boundingBox) {
    final Map<String, double> objectHeights = {
      'person': 1.7,
      'chair': 0.9,
      'car': 1.5,
      'bicycle': 1.1,
      'motorcycle': 1.2,
      'bus': 3.0,
      'truck': 2.5,
      'bottle': 0.25,
      'cup': 0.12,
      'cell phone': 0.15,
      'laptop': 0.35,
      'tv': 0.9,
      'couch': 0.8,
      'bed': 0.6,
      'dining table': 0.75,
      'toilet': 0.7,
      'door': 2.0,
      'refrigerator': 1.7,
      'book': 0.2,
      'clock': 0.3,
      'backpack': 0.5,
      'handbag': 0.35,
      'suitcase': 0.7,
      'umbrella': 1.0,
      'traffic light': 3.0,
      'stop sign': 0.9,
      'bench': 0.5,
      'potted plant': 0.4,
    };

    if (!objectHeights.containsKey(label)) return -1;

    // Typical phone camera focal length in 640px image space is ~500px
    // (derived from ~60° vertical FOV: f = 640 / (2 * tan(30°)) ≈ 554)
    const double focalLength = 500.0;

    final double pixelHeight = boundingBox.height;
    if (pixelHeight <= 0) return -1;

    return (objectHeights[label]! * focalLength) / pixelHeight;
  }

  /// Convert distance to human-friendly category
  String _getDistanceCategory(double meters) {
    if (meters < 0) return '';
    if (meters < 1.0) return 'very close';
    if (meters < 2.5) return 'close';
    if (meters < 5.0) return 'ahead';
    return 'far ahead';
  }

  /// Queue a message: only the latest pending message is kept.
  /// If nothing is speaking, speak immediately.
  void _speak(String message) {
    if (_isSpeaking) {
      // Replace any pending message with the newer one
      _pendingMessage = message;
    } else {
      _pendingMessage = message;
      _processPending();
    }
  }

  Future<void> _processPending() async {
    if (_isSpeaking || _pendingMessage == null) return;
    final message = _pendingMessage!;
    _pendingMessage = null;
    _isSpeaking = true;
    print('Speaking: $message');
    await _tts.speak(message);
  }

  /// Voice command: "What's in front of me?"
  Future<void> describeScene() async {
    if (_currentDetections.isEmpty) {
      _speak('Nothing detected');
      return;
    }

    final sorted = [..._currentDetections];
    sorted.sort((a, b) {
      final distA = _estimateDistance(a.label, a.location);
      final distB = _estimateDistance(b.label, b.location);
      if (distA < 0) return 1;
      if (distB < 0) return -1;
      return distA.compareTo(distB);
    });

    final items = sorted.take(5).map((det) {
      final dist = _estimateDistance(det.label, det.location);
      final category = _getDistanceCategory(dist);
      return category.isEmpty ? det.label : '${det.label} $category';
    }).join(', ');

    _speak('I see $items');
  }

  /// Voice command: Count specific objects
  Future<void> countObjects(String objectName) async {
    final count = _currentDetections
        .where((det) =>
            det.label.toLowerCase().contains(objectName.toLowerCase()))
        .length;

    if (count == 0) {
      _speak('No $objectName detected');
    } else if (count == 1) {
      _speak('One $objectName');
    } else {
      _speak('$count ${objectName}s');
    }
  }

  /// Start listening for voice commands.
  /// Stops TTS first so mic doesn't pick up the speaker.
  Future<void> startListening(Function(String) onCommand) async {
    if (!_stt.isAvailable) {
      print('Speech recognition not available');
      return;
    }

    // Stop any ongoing speech before listening
    await stopSpeaking();

    await _stt.listen(
      onResult: (result) {
        if (result.finalResult) {
          final command = result.recognizedWords.toLowerCase();
          print('Heard: $command');

          if (command.contains('what') &&
              (command.contains('front') || command.contains('see'))) {
            describeScene();
          } else if (command.contains('how many')) {
            final words = command.split(' ');
            final objectIndex = words.indexOf('many') + 1;
            if (objectIndex < words.length) {
              countObjects(words[objectIndex]);
            }
          } else {
            onCommand(command);
          }

          onListeningStateChanged?.call(false);
        }
      },
      listenFor: const Duration(seconds: 5),
      pauseFor: const Duration(seconds: 3),
    );

    onListeningStateChanged?.call(true);
  }

  /// Stop listening
  Future<void> stopListening() async {
    await _stt.stop();
    onListeningStateChanged?.call(false);
  }

  /// Stop speaking immediately and clear any pending message
  Future<void> stopSpeaking() async {
    _pendingMessage = null;
    _isSpeaking = false;
    await _tts.stop();
  }

  /// Clean up old cooldown entries (call periodically)
  void cleanupCooldowns() {
    final now = DateTime.now();
    _lastAnnounced.removeWhere(
        (key, time) => now.difference(time).inSeconds > _cooldownSeconds * 2);
  }
}
