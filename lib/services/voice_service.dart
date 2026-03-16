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

  // Track if TTS is currently speaking (don't interrupt)
  bool _isSpeaking = false;

  // Queue for pending announcements
  final List<String> _speechQueue = [];

  // Store current detections for "what's in front of me" queries
  List<DetectedObjectDm> _currentDetections = [];

  Future<void> initialize() async {
    // Configure TTS
    await _tts.setLanguage("en-US");
    await _tts.setSpeechRate(0.5); // Slightly slower for clarity
    await _tts.setVolume(1.0);
    await _tts.setPitch(1.0);

    // Listen for TTS completion
    _tts.setCompletionHandler(() {
      _isSpeaking = false;
      _processQueue(); // Speak next item in queue
    });

    // Initialize STT
    await _stt.initialize(
      onError: (error) => print('STT Error: $error'),
      onStatus: (status) => print('STT Status: $status'),
    );
  }

  /// Main method: Announce detections (called every frame)
  Future<void> announceDetections(List<DetectedObjectDm> detections) async {
    if (detections.isEmpty) return;

    // Store current detections for voice queries
    _currentDetections = detections;

    // Find NEAREST object (highest priority)
    final nearest = _findNearestObject(detections);
    if (nearest == null) return;

    // Calculate distance
    final distance = _estimateDistance(nearest.label, nearest.location);
    final distanceCategory = _getDistanceCategory(distance);

    // Check cooldown - don't announce same object repeatedly
    final key = '${nearest.label}_$distanceCategory';
    final lastTime = _lastAnnounced[key];
    final now = DateTime.now();

    if (lastTime != null &&
        now.difference(lastTime).inSeconds < _cooldownSeconds) {
      return; // Still in cooldown period
    }

    // Update last announced time
    _lastAnnounced[key] = now;

    // Create announcement
    String message;
    if (distance > 0) {
      message = '${nearest.label} $distanceCategory';
    } else {
      message = nearest.label;
    }

    // Add to queue and process
    _speak(message);
  }

  /// Find nearest object based on bounding box size
  DetectedObjectDm? _findNearestObject(List<DetectedObjectDm> detections) {
    if (detections.isEmpty) return null;

    // Larger bounding box = closer object
    detections.sort((a, b) {
      final areaA = a.location.width * a.location.height;
      final areaB = b.location.width * b.location.height;
      return areaB.compareTo(areaA); // Descending
    });

    return detections.first;
  }

  /// Estimate distance using object size heuristics
  double _estimateDistance(String label, Rect boundingBox) {
    // Known average heights in real world (meters)
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

    // Camera focal length (approximate - works for most phones)
    const double focalLength = 600; // pixels

    // Calculate distance: distance = (realHeight * focalLength) / pixelHeight
    double pixelHeight = boundingBox.height;
    double realHeight = objectHeights[label]!;

    double distance = (realHeight * focalLength) / pixelHeight;

    return distance; // meters
  }

  /// Convert distance to human-friendly category
  String _getDistanceCategory(double meters) {
    if (meters < 0) return ''; // Unknown distance
    if (meters < 0.8) return 'very close';
    if (meters < 2.0) return 'close';
    if (meters < 4.0) return 'ahead';
    return 'far ahead';
  }

  /// Speak text (queued to avoid interruption)
  void _speak(String message) {
    _speechQueue.add(message);
    _processQueue();
  }

  /// Process speech queue (don't interrupt current speech)
  Future<void> _processQueue() async {
    if (_isSpeaking || _speechQueue.isEmpty) return;

    _isSpeaking = true;
    final message = _speechQueue.removeAt(0);

    print('🔊 Speaking: $message');
    await _tts.speak(message);
  }

  /// Voice command: "What's in front of me?"
  Future<void> describeScene() async {
    if (_currentDetections.isEmpty) {
      _speak('Nothing detected');
      return;
    }

    // Sort by distance (nearest first)
    final sorted = [..._currentDetections];
    sorted.sort((a, b) {
      final distA = _estimateDistance(a.label, a.location);
      final distB = _estimateDistance(b.label, b.location);
      if (distA < 0) return 1; // Unknown distances go last
      if (distB < 0) return -1;
      return distA.compareTo(distB);
    });

    // Create description
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
        .where((det) => det.label.toLowerCase().contains(objectName.toLowerCase()))
        .length;

    if (count == 0) {
      _speak('No $objectName detected');
    } else if (count == 1) {
      _speak('One $objectName');
    } else {
      _speak('$count ${objectName}s');
    }
  }

  /// Start listening for voice commands
  Future<void> startListening(Function(String) onCommand) async {
    if (!_stt.isAvailable) {
      print('Speech recognition not available');
      return;
    }

    await _stt.listen(
      onResult: (result) {
        if (result.finalResult) {
          final command = result.recognizedWords.toLowerCase();
          print('🎤 Heard: $command');

          // Process commands
          if (command.contains('what') &&
              (command.contains('front') || command.contains('see'))) {
            describeScene();
          } else if (command.contains('how many')) {
            // Extract object name
            final words = command.split(' ');
            final objectIndex = words.indexOf('many') + 1;
            if (objectIndex < words.length) {
              final objectName = words[objectIndex];
              countObjects(objectName);
            }
          } else {
            onCommand(command); // Custom command handling
          }
        }
      },
      listenFor: Duration(seconds: 5),
      pauseFor: Duration(seconds: 3),
    );
  }

  /// Stop listening
  Future<void> stopListening() async {
    await _stt.stop();
  }

  /// Stop speaking immediately
  Future<void> stopSpeaking() async {
    await _tts.stop();
    _isSpeaking = false;
    _speechQueue.clear();
  }

  /// Clean up old cooldown entries (call periodically)
  void cleanupCooldowns() {
    final now = DateTime.now();
    _lastAnnounced.removeWhere((key, time) =>
      now.difference(time).inSeconds > _cooldownSeconds * 2
    );
  }
}