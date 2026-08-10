import 'package:flutter_tts/flutter_tts.dart';
import 'package:speech_to_text/speech_to_text.dart';
import 'dart:async';
import 'dart:math';
import 'dart:ui';
import 'package:flutter/services.dart';
import 'package:tensorflow_demo/models/detected_object/detected_object_dm.dart';
import 'package:battery_plus/battery_plus.dart';
import 'package:intl/intl.dart';

class VoiceService {
  static final VoiceService instance = VoiceService._();
  VoiceService._();

  static const MethodChannel _appLauncherChannel =
      MethodChannel('envision/app_launcher');

  final FlutterTts _tts = FlutterTts();
  final SpeechToText _stt = SpeechToText();

  // ── Cooldown tracking ────────────────────────────────────────────────────
  final Map<String, DateTime> _lastAnnounced = {};
  /// Global floor: no detection announcement fires sooner than this after the last one.
  DateTime? _lastAnnouncementTime;
  static const _minAnnouncementInterval = Duration(seconds: 3);

  // ── Hazard-interrupt tracking ────────────────────────────────────────────
  /// Distance (metres) of the most recently announced object.
  double? _lastAnnouncedDistance;
  /// Guards against back-to-back hazard overrides (max one every 2 s).
  DateTime? _lastHazardOverrideTime;

  // ── Path-clear tracking ───────────────────────────────────────────────────
  /// True once we've said "path seems clear" so we don't repeat it.
  bool _pathClearAnnounced = false;
  /// Last time any detection was seen (used for path-clear timing).
  DateTime? _lastDetectionTime;

  // ── TTS state ────────────────────────────────────────────────────────────
  bool _isSpeaking = false;
  String? _pendingMessage;

  // ── Wake-word detection ──────────────────────────────────────────────────
  bool _wakeWordEnabled = false;
  bool _inConversation = false;
  Timer? _wakeWordTimer;

  // ── Conversation context ─────────────────────────────────────────────────
  String? _lastResponse;

  // ── Current scene state ──────────────────────────────────────────────────
  List<DetectedObjectDm> _currentDetections = [];
  List<Map<String, dynamic>> _cachedApks = [];

  // ── Callbacks ────────────────────────────────────────────────────────────
  void Function(bool isListening)? onListeningStateChanged;
  void Function()? onWakeWordDetected;
  /// Called with a human-readable summary of the last heard command (for UI display)
  void Function(String text)? onCommandHeard;

  // ── Command vs wake-word listening ───────────────────────────────────────
  // The wake-word polling loop keeps _stt busy in the background.
  // _commandListening is ONLY true when the user explicitly triggered a
  // command session (tap or wake word detected). This is what the UI checks.
  bool _commandListening = false;
  bool _isInitialized = false;
  Future<void>? _initializationFuture;

  /// True only when a user-triggered command session is active.
  /// (Wake-word background polling does NOT set this flag.)
  bool get isCommandListening => _commandListening;

  /// Raw STT state — used internally; prefer [isCommandListening] for UI.
  bool get isListening => _stt.isListening;
  bool get wakeWordEnabled => _wakeWordEnabled;

  // ── Priority / hazard classification ─────────────────────────────────────
  /// Classes that are dangerous to a visually impaired pedestrian.
  /// These get lower confidence thresholds AND are always sorted to the front
  /// of the announcement queue.
  // Keys are lowercase; callers must lower-case the label before checking,
  // because the accessibility model emits capitalized labels (e.g. "Stair").
  // Stairs are included here: for a blind pedestrian an unexpected staircase —
  // especially a descending one — is one of the most dangerous obstacles.
  static const Set<String> _hazardClasses = {
    'person', 'car', 'truck', 'bus', 'motorcycle', 'bicycle',
    'dog', 'fire hydrant', 'stop sign', 'traffic light', 'stair',
  };

  // ── Object real-world heights (metres) ──────────────────────────────────
  static const Map<String, double> _objectHeights = {
    'person': 1.7,
    'bicycle': 1.1,
    'car': 1.5,
    'motorcycle': 1.2,
    'bus': 3.0,
    'truck': 2.5,
    'traffic light': 3.0,
    'fire hydrant': 0.6,
    'stop sign': 0.9,
    'bench': 0.5,
    'bird': 0.25,
    'cat': 0.3,
    'dog': 0.5,
    'horse': 1.6,
    'backpack': 0.5,
    'umbrella': 1.0,
    'handbag': 0.35,
    'suitcase': 0.7,
    'bottle': 0.25,
    'cup': 0.12,
    'bowl': 0.1,
    'chair': 0.9,
    'couch': 0.8,
    'potted plant': 0.4,
    'bed': 0.6,
    'dining table': 0.75,
    'toilet': 0.7,
    'tv': 0.9,
    'laptop': 0.35,
    'cell phone': 0.15,
    'book': 0.2,
    'clock': 0.3,
    'refrigerator': 1.7,
    // ── Accessibility model classes (keys MUST be lowercase) ──────────────
    // The accessibility model emits capitalized labels (Door, Stair, Window,
    // Gate); _estimateDistance lower-cases the label before lookup, so these
    // keys are lowercase. Heights are the approximate visible vertical extent
    // of the object as it typically appears in-frame (used by the pinhole
    // distance estimate), NOT a single step's riser height.
    'door': 2.0,
    'stair': 1.0,    // visible height of a flight as it fills the frame
    'stairs': 1.0,
    'window': 1.2,
    'gate': 1.8,
    'curb': 0.15,
    'keyboard': 0.05,
    'mouse': 0.04,
    'remote': 0.2,
    'vase': 0.3,
    'scissors': 0.2,
    'toothbrush': 0.19,
  };

  // ── Initialization ────────────────────────────────────────────────────────
  Future<void> initialize() async {
    if (_isInitialized) return;
    if (_initializationFuture != null) return _initializationFuture!;

    _initializationFuture = _initializeInternal().then((_) {
      _isInitialized = true;
    }).whenComplete(() {
      _initializationFuture = null;
    });

    return _initializationFuture!;
  }

  Future<void> ensureInitialized() async {
    await initialize();
  }

  Future<void> _initializeInternal() async {
    await _tts.setLanguage('en-US');
    await _tts.setSpeechRate(0.48);
    await _tts.setVolume(1.0);
    await _tts.setPitch(1.0);

    _tts.setCompletionHandler(_onTtsDone);
    _tts.setCancelHandler(_onTtsDone);
    _tts.setErrorHandler((_) => _onTtsDone());

    final sttAvailable = await _stt.initialize(
      onError: (error) {
        print('STT Error: $error');
        if (_wakeWordEnabled && !_inConversation) {
          // Reschedule wake-word loop after an STT error
          _scheduleWakeWordPoll();
        }
        onListeningStateChanged?.call(false);
      },
      onStatus: (status) {
        print('STT Status: $status');
        if (status == 'notListening' || status == 'done') {
          onListeningStateChanged?.call(false);
          if (_wakeWordEnabled && !_inConversation) {
            _scheduleWakeWordPoll();
          }
        }
      },
    );
    print('STT initialized: $sttAvailable');
  }

  void _onTtsDone() {
    _isSpeaking = false;
    _processPending();
  }

  // ─────────────────────────────────────────────────────────────────────────
  // DETECTION ANNOUNCEMENTS
  // ─────────────────────────────────────────────────────────────────────────

  /// Called every camera frame with current detections.
  Future<void> announceDetections(List<DetectedObjectDm> detections) async {
    // Keep the scene snapshot fresh ALWAYS — describeScene() and countObjects()
    // read this, and they used to see stale data because the old early-return
    // sat above this line and fired on most frames.
    _currentDetections = detections;

    // ── Microphone arbitration ────────────────────────────────────────────
    // The wake-word poller holds the mic ~83% of the time (4 s listen, 800 ms
    // gap). Gating EVERY announcement on `_stt.isListening` therefore silenced
    // the app's primary safety function on roughly 5 of every 6 frames.
    //
    // New rule:
    //   - a user-initiated command session always wins (they asked a question)
    //   - ordinary objects still yield to the mic
    //   - a HAZARD does not yield: we take the mic back and speak
    final hasHazard = detections
        .any((d) => _hazardClasses.contains(d.label.toLowerCase()));

    if (_inConversation || _commandListening) return;

    if (_stt.isListening) {
      if (!hasHazard) return;
      // Release the mic before speaking, otherwise the TTS output is fed
      // straight back into the recogniser. The STT onStatus handler
      // reschedules the wake-word poll once the mic is free.
      await _stt.stop();
    }

    // ── Path-clear logic ──────────────────────────────────────────────────
    if (detections.isEmpty) {
      final now = DateTime.now();
      if (!_pathClearAnnounced &&
          !_isSpeaking &&
          _lastDetectionTime != null &&
          now.difference(_lastDetectionTime!) > const Duration(seconds: 6)) {
        _pathClearAnnounced = true;
        _lastAnnouncementTime = now;
        _speak('Path seems clear.');
      }
      return;
    }

    _pathClearAnnounced = false;
    _lastDetectionTime = DateTime.now();

    // ── Score + deduplicate ───────────────────────────────────────────────
    // Returns unique-by-label list sorted by: hazard first, then proximity.
    final scored = _scoreDetections(detections);
    if (scored.isEmpty) return;

    final nearest = scored.first;
    final meters = _estimateDistance(nearest);
    final now = DateTime.now();

    // ── Hazard override ───────────────────────────────────────────────────
    // If TTS is mid-sentence but a significantly closer hazard just appeared
    // (35% nearer, under 2 m, max one override every 2 s) — cut the speech.
    final hazardOverride = _isSpeaking
        && meters > 0
        && meters < 2.0
        && _lastAnnouncedDistance != null
        && meters < _lastAnnouncedDistance! * 0.65
        && (_lastHazardOverrideTime == null ||
            now.difference(_lastHazardOverrideTime!) >
                const Duration(seconds: 2));

    if (_isSpeaking && !hazardOverride) return;

    if (hazardOverride) {
      await _tts.stop();
      _isSpeaking = false;
      _lastHazardOverrideTime = now;
    }

    // ── Global rate limiter ───────────────────────────────────────────────
    if (!hazardOverride &&
        _lastAnnouncementTime != null &&
        now.difference(_lastAnnouncementTime!) < _minAnnouncementInterval) {
      return;
    }

    // ── Per-object cooldown ───────────────────────────────────────────────
    final key = '${nearest.label}_${_distanceBucket(meters)}';
    final lastTime = _lastAnnounced[key];
    if (!hazardOverride &&
        lastTime != null &&
        now.difference(lastTime).inSeconds < _dynamicCooldown(meters)) {
      return;
    }

    _lastAnnounced[key] = now;
    _lastAnnouncementTime = now;
    _lastAnnouncedDistance = meters;

    // ── Build announcement ────────────────────────────────────────────────
    final position = _getPosition(nearest);
    final phrase = _buildDistancePhrase(nearest.label, meters, position);

    String announcement;
    if (meters >= 0 && meters < 0.8) {
      // Urgent — just primary, no secondary clutter
      announcement = 'Warning! $phrase.';
    } else {
      announcement = '$phrase.';

      // Append a second object when it's a different class and within 5 m.
      // Gives the user richer spatial context in one speech act.
      if (scored.length > 1) {
        final second = scored[1];
        if (second.label != nearest.label) {
          final secMeters = _estimateDistance(second);
          if (secMeters < 0 || secMeters < 5.0) {
            final secPos = _getPosition(second);
            announcement +=
                ' ${_buildDistancePhrase(second.label, secMeters, secPos)}.';
          }
        }
      }
    }

    _speak(announcement);
  }

  // ─────────────────────────────────────────────────────────────────────────
  // DETECTION PRIORITISATION
  // ─────────────────────────────────────────────────────────────────────────

  /// Deduplicate by label (keep nearest instance of each class) then sort:
  ///   1. Hazard classes always come first.
  ///   2. Within the same priority tier, closer (larger bbox) wins.
  List<DetectedObjectDm> _scoreDetections(List<DetectedObjectDm> detections) {
    // Keep the largest (nearest) bbox per label
    final byLabel = <String, DetectedObjectDm>{};
    for (final det in detections) {
      final existing = byLabel[det.label];
      if (existing == null) {
        byLabel[det.label] = det;
      } else {
        final area = det.location.width * det.location.height;
        final existingArea = existing.location.width * existing.location.height;
        if (area > existingArea) byLabel[det.label] = det;
      }
    }

    final unique = byLabel.values.toList();
    unique.sort((a, b) {
      final aH = _hazardClasses.contains(a.label.toLowerCase()) ? 1 : 0;
      final bH = _hazardClasses.contains(b.label.toLowerCase()) ? 1 : 0;
      if (aH != bH) return bH - aH; // hazard first
      // Same tier → closer (larger area) first
      final aArea = a.location.width * a.location.height;
      final bArea = b.location.width * b.location.height;
      return bArea.compareTo(aArea);
    });

    return unique;
  }

  /// Spoken welcome message — call once when the camera screen is ready.
  Future<void> speakWelcome() async {
    await Future.delayed(const Duration(milliseconds: 800));
    _speak(
      'Envision is ready. '
      'Objects around you will be announced automatically. '
      'Tap anywhere to give a voice command, '
      'or double tap to describe the scene.',
    );
  }

  // ─────────────────────────────────────────────────────────────────────────
  // DISTANCE & POSITION HELPERS
  // ─────────────────────────────────────────────────────────────────────────

  /// Returns the nearest object by largest bounding-box area (bigger = closer).
  DetectedObjectDm? _findNearestObject(List<DetectedObjectDm> detections) {
    if (detections.isEmpty) return null;
    return detections.reduce((a, b) {
      final areaA = a.location.width * a.location.height;
      final areaB = b.location.width * b.location.height;
      return areaA >= areaB ? a : b;
    });
  }

  /// Pinhole-camera distance estimate, in metres. Returns -1 when the class
  /// has no reference height.
  ///
  /// Focal length is derived from the model input edge for an assumed ~60°
  /// vertical field of view:  f = size / (2 * tan(30°))  (≈554 px at 640).
  /// It scales automatically if the model is re-exported at another imgsz.
  ///
  /// The box height is divided by [DetectedObjectDm.yStretch] first. Undoing
  /// the letterbox padding stretches boxes on whichever axis was padded; when
  /// that is the y axis, an unstretched height makes every object read as
  /// closer than it is. That error is silent and always in the dangerous
  /// direction, so it is corrected here rather than left to luck of geometry.
  ///
  /// ACCURACY NOTE: this is a heuristic, not a measurement. It assumes a fixed
  /// real-world height per class (see [_objectHeights]) and a nominal FOV, so
  /// expect roughly +/-30%. Spoken output is deliberately rounded to match.
  double _estimateDistance(DetectedObjectDm det) {
    final h = _objectHeights[det.label.toLowerCase()];
    if (h == null) return -1;
    final focalLength = det.modelSize / (2 * tan(pi / 6));
    final pixelHeight = det.location.height / det.yStretch;
    if (pixelHeight <= 0) return -1;
    return (h * focalLength) / pixelHeight;
  }

  /// Which horizontal third of the frame is the object centre in?
  String _getPosition(DetectedObjectDm det) {
    final cx = det.location.left + det.location.width / 2;
    if (cx < det.modelSize / 3.0) return 'to your left';
    if (cx > det.modelSize * 2.0 / 3.0) return 'to your right';
    return 'ahead';
  }

  /// Bucket distance for cooldown-key deduplication.
  String _distanceBucket(double m) {
    if (m < 0) return 'unknown';
    if (m < 0.8) return 'urgent';
    if (m < 2.0) return 'very_close';
    if (m < 4.0) return 'close';
    if (m < 7.0) return 'ahead';
    return 'far';
  }

  /// Per-object cooldown in seconds. Even urgent objects need a few seconds
  /// between repeats — the global _minAnnouncementInterval handles fast firing.
  int _dynamicCooldown(double meters) {
    if (meters < 0) return 6;
    if (meters < 0.8) return 3;   // very close  — re-announce every 3 s
    if (meters < 2.0) return 4;   // close        — every 4 s
    if (meters < 4.0) return 6;   // ahead        — every 6 s
    return 9;                      // far           — every 9 s
  }

  /// Compose the spoken phrase with capitalized label and natural word order.
  /// Examples: "Person, 1.5 meters, to your left"
  ///           "Warning! Chair right in front of you"
  ///           "Car far ahead"
  String _buildDistancePhrase(String label, double meters, String position) {
    final cap = label.isNotEmpty
        ? label[0].toUpperCase() + label.substring(1)
        : label;
    if (meters < 0) return '$cap, $position';
    if (meters < 0.8) return '$cap right in front of you';
    if (meters < 10.0) return '$cap, ${_spokenDistance(meters)}, $position';
    return '$cap far ahead';
  }

  /// Renders a distance the way a person would say it.
  ///
  /// The pinhole estimate is roughly +/-30% accurate, so "1.7 meters" claims a
  /// precision the app does not have and sounds robotic read aloud. Rounding to
  /// the half-metre below 3 m and the whole metre above is both more honest and
  /// faster to hear.
  String _spokenDistance(double meters) {
    if (meters < 3.0) {
      final half = (meters * 2).round() / 2.0;
      if (half <= 0.5) return 'half a meter';
      if (half == 1.0) return 'one meter';
      return half == half.roundToDouble()
          ? '${half.toInt()} meters'
          : '${half.toStringAsFixed(1)} meters';
    }
    return 'about ${meters.round()} meters';
  }

  // ─────────────────────────────────────────────────────────────────────────
  // TTS HELPERS
  // ─────────────────────────────────────────────────────────────────────────

  /// Public API for one-off spoken feedback (UI buttons, confirmations, etc.).
  void speak(String message) => _speak(message);

  void _speak(String message) {
    _pendingMessage = message;
    if (!_isSpeaking) _processPending();
  }

  Future<void> _processPending() async {
    if (_isSpeaking || _pendingMessage == null) return;
    final message = _pendingMessage!;
    _pendingMessage = null;
    _isSpeaking = true;
    print('Speaking: $message');
    await _tts.speak(message);
  }

  /// Stop all speech and clear pending queue.
  Future<void> stopSpeaking() async {
    _pendingMessage = null;
    _isSpeaking = false;
    await _tts.stop();
  }

  // ─────────────────────────────────────────────────────────────────────────
  // WAKE-WORD DETECTION
  // ─────────────────────────────────────────────────────────────────────────
  // Uses polling short-burst STT sessions to detect "envision" or
  // "hey envision" while the app is in foreground or background service.
  //
  // NOTE: Launching the app from a fully closed state requires OS-level
  // integration (Android VoiceInteractionService / Porcupine SDK).
  // That is documented in the report as a future-work item.

  Future<void> setWakeWordEnabled(bool enable) async {
    _wakeWordEnabled = enable;
    _wakeWordTimer?.cancel();
    _wakeWordTimer = null;
    if (enable) {
      print('[WakeWord] Wake-word detection enabled');
      _scheduleWakeWordPoll();
    } else {
      print('[WakeWord] Wake-word detection disabled');
      if (_stt.isListening) await _stt.stop();
    }
  }

  void _scheduleWakeWordPoll() {
    _wakeWordTimer?.cancel();
    // Small delay so any current STT session fully releases the mic
    _wakeWordTimer = Timer(const Duration(milliseconds: 800), _pollWakeWord);
  }

  Future<void> _pollWakeWord() async {
    if (!_wakeWordEnabled || _inConversation || _isSpeaking || _stt.isListening) {
      // Try again later
      _scheduleWakeWordPoll();
      return;
    }

    if (!_stt.isAvailable) return;

    bool detected = false;

    try {
      await _stt.listen(
        onResult: (result) {
          final words = result.recognizedWords.toLowerCase();
          if (!detected &&
              (words.contains('envision') || words.contains('vision'))) {
            detected = true;
            _stt.stop();
            _handleWakeWordDetected();
          }
        },
        listenFor: const Duration(seconds: 4),
        pauseFor: const Duration(seconds: 1),
        partialResults: true,
        cancelOnError: true,
      );
    } catch (e) {
      print('[WakeWord] STT error during poll: $e');
    }

    // Next poll is scheduled by the onStatus callback ('done'/'notListening')
    // to ensure the mic is free. The fallback timer handles edge cases.
    if (!detected) {
      _scheduleWakeWordPoll();
    }
  }

  void _handleWakeWordDetected() {
    print('[WakeWord] Detected!');
    onWakeWordDetected?.call();
    _inConversation = true;
    _speak('Yes, how can I help?');
    onCommandHeard?.call('Wake word detected');

    // Wait for TTS to finish, then start a full command session
    _waitForTtsThenListen();
  }

  void _waitForTtsThenListen() {
    // Poll until TTS finishes, then start command listening
    Timer.periodic(const Duration(milliseconds: 200), (timer) {
      if (!_isSpeaking) {
        timer.cancel();
        startListening((unrecognized) {
          _speak("Sorry, I didn't catch that. Try saying 'help' for a list of commands.");
          onCommandHeard?.call('Unrecognized: $unrecognized');
        });
      }
    });
  }

  // ─────────────────────────────────────────────────────────────────────────
  // VOICE COMMAND LISTENING
  // ─────────────────────────────────────────────────────────────────────────

  /// Start a full command-listening session (stops TTS + any wake-word poll).
  Future<void> startListening(Function(String) onUnrecognized) async {
    if (!_stt.isAvailable) {
      print('Speech recognition not available');
      return;
    }

    // Stop the wake-word poll so it doesn't compete for the mic
    _wakeWordTimer?.cancel();
    if (_stt.isListening) await _stt.stop();

    await stopSpeaking();

    _commandListening = true;
    onListeningStateChanged?.call(true);

    await _stt.listen(
      onResult: (result) {
        if (result.finalResult) {
          final command = result.recognizedWords.toLowerCase().trim();
          print('Heard: $command');
          onCommandHeard?.call('"$command"');
          _commandListening = false;
          _inConversation = false;
          _parseVoiceCommand(command, onUnrecognized);
          onListeningStateChanged?.call(false);

          // Resume wake-word polling after command is processed
          if (_wakeWordEnabled) _scheduleWakeWordPoll();
        }
      },
      listenFor: const Duration(seconds: 10),
      pauseFor: const Duration(seconds: 3),
    );
  }

  Future<void> stopListening() async {
    _commandListening = false;
    _inConversation = false;
    await _stt.stop();
    onListeningStateChanged?.call(false);
    // Resume wake-word polling if it was active
    if (_wakeWordEnabled) _scheduleWakeWordPoll();
  }

  // ─────────────────────────────────────────────────────────────────────────
  // COMMAND PARSING
  // ─────────────────────────────────────────────────────────────────────────

  Future<void> _parseVoiceCommand(
      String command, Function(String) onUnrecognized) async {
    // ── STOP / CANCEL ────────────────────────────────────────────────────
    if (_matchesAny(command, ['stop', 'cancel', 'quiet', 'silence', 'shut up'])) {
      await stopSpeaking();
      _speak('Stopped.');
      _lastResponse = 'Stopped';
    }
    // ── HELP ─────────────────────────────────────────────────────────────
    else if (_matchesAny(command, ['help', 'what can you do', 'commands'])) {
      await _tellHelp();
    }
    // ── REPEAT ───────────────────────────────────────────────────────────
    else if (_matchesAny(command, ['repeat', 'say again', 'again'])) {
      if (_lastResponse != null) {
        _speak(_lastResponse!);
      } else {
        _speak('Nothing to repeat.');
      }
    }
    // ── TIME ─────────────────────────────────────────────────────────────
    else if (_matchesAny(command, ['time', 'what time', 'what hour'])) {
      await tellTime();
    }
    // ── BATTERY ──────────────────────────────────────────────────────────
    else if (_matchesAny(command,
        ['battery', 'charge', 'battery level', 'how much battery'])) {
      await tellBattery();
    }
    // ── DATE ─────────────────────────────────────────────────────────────
    else if (_matchesAny(command, ['date', 'what day', "today's date"])) {
      await tellDate();
    }
    // ── DESCRIBE SCENE ───────────────────────────────────────────────────
    else if (_matchesAny(command, [
      "what's in front",
      'what do you see',
      'what is in front',
      'describe',
      "what's around",
      'what around',
      'surroundings',
    ])) {
      await describeScene();
    }
    // ── COUNT OBJECTS ────────────────────────────────────────────────────
    else if (command.contains('how many')) {
      final afterMany = _extractAfter(command, 'how many');
      if (afterMany != null) await countObjects(afterMany);
    }
    // ── OPEN / LAUNCH APP ────────────────────────────────────────────────
    else if (command.startsWith('open ') || command.startsWith('launch ')) {
      final appName = command.replaceFirst(RegExp(r'^(open|launch)\s+'), '');
      if (appName.isNotEmpty) await launchApp(appName);
    }
    // ── NAVIGATE / GO TO ─────────────────────────────────────────────────
    else if (_matchesAny(command, ['navigate to', 'directions to', 'take me to', 'go to'])) {
      final dest = command
          .replaceFirst(RegExp(r'^(navigate to|directions to|take me to|go to)\s*'), '')
          .trim();
      await navigateTo(dest.isNotEmpty ? dest : 'your destination');
    }
    // ── SCAN STORAGE ─────────────────────────────────────────────────────
    else if (_matchesAny(command, ['scan storage', 'find apk', 'search storage'])) {
      await scanStorageForApks();
    }
    // ── UNRECOGNIZED ─────────────────────────────────────────────────────
    else {
      onUnrecognized(command);
    }
  }

  /// Returns true if [text] contains any of the [phrases].
  bool _matchesAny(String text, List<String> phrases) {
    return phrases.any((p) => text.contains(p));
  }

  /// Extracts the substring after [keyword] in [text], or null if not found.
  String? _extractAfter(String text, String keyword) {
    final idx = text.indexOf(keyword);
    if (idx < 0) return null;
    final rest = text.substring(idx + keyword.length).trim();
    return rest.isEmpty ? null : rest;
  }

  // ─────────────────────────────────────────────────────────────────────────
  // INDIVIDUAL COMMANDS
  // ─────────────────────────────────────────────────────────────────────────

  Future<void> describeScene() async {
    if (_currentDetections.isEmpty) {
      const response = 'Nothing detected in front of you.';
      _speak(response);
      _lastResponse = response;
      return;
    }

    // Count occurrences of each label for natural grouping
    final labelCounts = <String, int>{};
    for (final det in _currentDetections) {
      labelCounts[det.label] = (labelCounts[det.label] ?? 0) + 1;
    }

    // Priority-sorted, one per label
    final scored = _scoreDetections(_currentDetections);

    final items = scored.take(5).map((det) {
      final dist = _estimateDistance(det);
      final pos = _getPosition(det);
      final count = labelCounts[det.label] ?? 1;

      // Group label — "3 people" vs "A person"
      final String labelStr;
      if (count > 1) {
        labelStr = '$count ${_pluralise(det.label)}';
      } else {
        labelStr = det.label[0].toUpperCase() + det.label.substring(1);
      }

      if (dist < 0) return '$labelStr, $pos';
      if (dist < 0.8) return '$labelStr right in front of you';
      if (dist < 10.0) return '$labelStr, ${_spokenDistance(dist)}, $pos';
      return '$labelStr, far ahead';
    }).join('; ');

    final response = 'I can see: $items.';
    _speak(response);
    _lastResponse = response;
  }

  /// English pluralisation for the handful of labels where naive +'s' is wrong.
  /// Without this the app says "3 persons", "3 bus" and "3 sheeps".
  static const Map<String, String> _irregularPlurals = {
    'person': 'people',
    'bus': 'buses',
    'sheep': 'sheep',
    'skis': 'skis',
    'scissors': 'scissors',
    'mouse': 'mice',
    'knife': 'knives',
    'sandwich': 'sandwiches',
    'couch': 'couches',
    'bench': 'benches',
    'stair': 'stairs',
    'glass': 'glasses',
  };

  static String _pluralise(String label) {
    final lower = label.toLowerCase();
    final irregular = _irregularPlurals[lower];
    if (irregular != null) return irregular;
    if (lower.endsWith('s') || lower.endsWith('x') || lower.endsWith('ch') ||
        lower.endsWith('sh')) {
      return '${label}es';
    }
    return '${label}s';
  }

  Future<void> countObjects(String objectName) async {
    final cleaned = objectName.replaceAll(RegExp(r'\s+(are|is|there|do you see).*'), '').trim();
    final count = _currentDetections
        .where((det) => det.label.toLowerCase().contains(cleaned.toLowerCase()))
        .length;

    String response;
    if (count == 0) {
      response = 'No $cleaned detected right now.';
    } else if (count == 1) {
      response = 'One $cleaned detected.';
    } else {
      response = '$count ${_pluralise(cleaned)} detected.';
    }
    _speak(response);
    _lastResponse = response;
  }

  Future<void> tellTime() async {
    try {
      final now = DateTime.now();
      final timeStr = DateFormat('h:mm a').format(now);
      final response = 'The time is $timeStr.';
      _speak(response);
      _lastResponse = response;
    } catch (e) {
      _speak('Unable to get the current time.');
      print('Error getting time: $e');
    }
  }

  Future<void> tellDate() async {
    try {
      final now = DateTime.now();
      final dateStr = DateFormat('EEEE, MMMM d, yyyy').format(now);
      final response = 'Today is $dateStr.';
      _speak(response);
      _lastResponse = response;
    } catch (e) {
      _speak('Unable to get the date.');
    }
  }

  Future<void> tellBattery() async {
    try {
      final battery = Battery();
      final level = await battery.batteryLevel;
      final state = await battery.batteryState;
      final charging = state == BatteryState.charging ? ', and it is charging' : '';
      final response = 'Battery is at $level percent$charging.';
      _speak(response);
      _lastResponse = response;
    } catch (e) {
      _speak('Unable to get battery information.');
      print('Error getting battery: $e');
    }
  }

  Future<void> _tellHelp() async {
    const response =
        'You can say: describe, open followed by an app name, '
        'what time is it, battery, today\'s date, '
        'how many people, navigate to a place, '
        'repeat, stop, or help.';
    _speak(response);
    _lastResponse = response;
  }

  /// Opens Google Maps or Waze with [destination] as the target.
  Future<void> navigateTo(String destination) async {
    // Try to open Maps first, then Waze as fallback.
    final opened = await _tryLaunchApp('maps') || await _tryLaunchApp('waze');
    if (opened) {
      _speak('Opening navigation to $destination.');
      _lastResponse = 'Opening navigation to $destination';
    } else {
      _speak('I could not find a navigation app on this device.');
    }
  }

  Future<bool> _tryLaunchApp(String name) async {
    try {
      final dynamic resp = await _appLauncherChannel.invokeMethod<dynamic>(
        'openAppByName',
        {'appName': name},
      );
      if (resp is Map) return resp['opened'] == true;
      if (resp is bool) return resp;
    } catch (_) {}
    return false;
  }

  Future<void> launchApp(String appName) async {
    try {
      final dynamic resp = await _appLauncherChannel.invokeMethod<dynamic>(
        'openAppByName',
        {'appName': appName},
      );

      String response;
      if (resp is Map) {
        final opened = resp['opened'] == true;
        if (opened) {
          final label = resp['label'] as String? ?? appName;
          response = 'Opening $label.';
        } else {
          final reason = resp['reason'] as String? ?? 'unknown';
          if (reason == 'no_match') {
            final candidates = resp['candidates'] as List<dynamic>?;
            if (candidates != null && candidates.isNotEmpty) {
              final top = candidates
                  .take(2)
                  .map((c) => c['label'] ?? c['package'])
                  .join(' or ');
              response = 'Could not find $appName. Did you mean $top?';
            } else {
              // Check cached APKs
              final target = appName.trim().toLowerCase();
              final localMatches = _cachedApks.where((a) {
                final lbl = (a['label'] ?? '').toString().toLowerCase();
                final pkg = (a['package'] ?? '').toString().toLowerCase();
                return lbl.contains(target) || pkg.contains(target);
              }).toList();
              if (localMatches.isNotEmpty) {
                final top = localMatches
                    .take(2)
                    .map((c) => c['label'] ?? c['package'])
                    .join(' or ');
                response =
                    'App not installed, but found APK on storage: $top. Say install followed by the app name to install.';
              } else {
                response = '$appName is not installed on this device.';
              }
            }
          } else if (reason == 'no_launch_intent') {
            response = 'Found $appName but could not open it.';
          } else {
            response = 'Could not open $appName.';
          }
        }
      } else if (resp == true) {
        response = 'Opening $appName.';
      } else {
        response = '$appName is not installed on this device.';
      }
      _speak(response);
      _lastResponse = response;
    } on MissingPluginException {
      _speak('App launching is only available on Android.');
    } catch (e) {
      _speak('There was an error opening the app.');
      print('Error launching app: $e');
    }
  }

  Future<void> scanStorageForApks() async {
    try {
      final granted = await _appLauncherChannel
              .invokeMethod<bool>('requestStoragePermission') ??
          false;
      if (!granted) {
        _speak('Please grant storage permission to search for apps.');
        return;
      }
      final resp =
          await _appLauncherChannel.invokeMethod<dynamic>('scanExternalApks');
      if (resp is Map && resp['granted'] == true) {
        final List<dynamic>? apks = resp['apks'] as List<dynamic>?;
        if (apks == null || apks.isEmpty) {
          _speak('No APK files found on storage.');
          _cachedApks = [];
          return;
        }
        _cachedApks = apks
            .map((e) => {
                  'label': e['label'] ?? '',
                  'package': e['package'] ?? '',
                  'path': e['path'] ?? ''
                })
            .toList();
        final top =
            _cachedApks.take(3).map((e) => e['label'] ?? e['package']).join(', ');
        _speak('Found APKs: $top.');
      } else {
        _speak('Storage scan not permitted.');
      }
    } on MissingPluginException {
      _speak('Storage scanning is only available on Android.');
    } catch (e) {
      _speak('Error scanning storage.');
      print('scanStorageForApks error: $e');
    }
  }

  // ─────────────────────────────────────────────────────────────────────────
  // CLEANUP
  // ─────────────────────────────────────────────────────────────────────────

  /// Remove stale cooldown entries. Called every 10 s from the screen.
  void cleanupCooldowns() {
    final now = DateTime.now();
    _lastAnnounced.removeWhere(
        (key, time) => now.difference(time).inSeconds > 12);
    // Reset global throttle if it's been idle long enough
    if (_lastAnnouncementTime != null &&
        now.difference(_lastAnnouncementTime!) > const Duration(seconds: 12)) {
      _lastAnnouncementTime = null;
    }
  }

  void dispose() {
    _wakeWordTimer?.cancel();
    _tts.stop();
  }
}
