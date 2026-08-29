import 'package:flutter_tts/flutter_tts.dart';
import 'package:speech_to_text/speech_to_text.dart';
import 'dart:async';
import 'dart:math';
import 'dart:ui';
import 'package:flutter/services.dart';
import 'package:tensorflow_demo/models/detected_object/detected_object_dm.dart';
import 'package:battery_plus/battery_plus.dart';
import 'package:intl/intl.dart';
import 'package:tensorflow_demo/services/mistral_service.dart';

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

  /// Consecutive polls that ended almost instantly (< 1.2s) with nothing
  /// heard — a sign something is wrong (usually missing mic permission)
  /// rather than just silence. Used to back off instead of hammering the
  /// mic (and its start/stop beep) every ~800ms forever.
  int _consecutiveFastFailures = 0;
  static const _maxFastFailuresBeforeBackoff = 3;

  // ── Conversation context ─────────────────────────────────────────────────
  String? _lastResponse;

  // ── Current scene state ──────────────────────────────────────────────────
  List<DetectedObjectDm> _currentDetections = [];
  List<Map<String, dynamic>> _cachedApks = [];

  /// Name of the currently-recognised face nearest the camera, if any. Set by
  /// the screen owning face recognition every time it processes a frame.
  /// When present, the object-detection announcement pipeline says this name
  /// instead of the generic "Person" for the person class — a recognized
  /// person should be greeted by name, not treated as an anonymous obstacle.
  String? _nearbyPersonName;

  /// Called by the camera screen with the name of the nearest recognised
  /// face, or null when no known face is currently in view.
  void setNearbyPersonName(String? name) {
    _nearbyPersonName = (name != null && name.isNotEmpty) ? name : null;
  }

  // ── Distance-estimation calibration ─────────────────────────────────────
  /// Vertical field of view used by the pinhole distance formula. Defaults to
  /// a generic 60° assumption; [calibrateCameraFov] overwrites this with the
  /// device's real value read from Camera2 characteristics, which is the
  /// single biggest source of error in the old fixed-FOV estimate (phones'
  /// main lenses commonly range ~55°-75° vertical FOV).
  double _verticalFovDegrees = 60.0;

  /// Recent per-label distance samples, used to smooth out frame-to-frame
  /// bounding-box jitter before it's spoken as a distance. Keyed by label
  /// rather than a tracked object id (the app has no object tracker), so a
  /// mid-swap between two same-class objects will blend briefly — an
  /// acceptable trade for calmer, less jittery output.
  final Map<String, List<double>> _recentDistances = {};
  static const int _distanceSmoothingWindow = 4;

  /// Reads the real camera vertical FOV via platform channel and applies it
  /// to future distance estimates. Safe to call once at camera startup; a
  /// no-op (keeps the 60° default) if the platform call fails or the device
  /// doesn't expose the characteristics.
  Future<void> calibrateCameraFov() async {
    try {
      final dynamic resp =
          await _appLauncherChannel.invokeMethod<dynamic>('getCameraFov');
      if (resp is Map && resp['available'] == true) {
        final fov = (resp['verticalFovDegrees'] as num?)?.toDouble();
        if (fov != null && fov > 10 && fov < 170) {
          _verticalFovDegrees = fov;
          print('[Calibration] Vertical FOV set to ${fov.toStringAsFixed(1)}°');
        }
      }
    } catch (e) {
      print('[Calibration] getCameraFov failed, using 60° default: $e');
    }
  }

  /// Smooths a raw per-frame distance estimate with recent samples of the
  /// same label. Unknown-distance (-1) values pass through untouched.
  double _smoothDistance(String label, double raw) {
    if (raw < 0) return raw;
    final samples = _recentDistances.putIfAbsent(label, () => []);
    samples.add(raw);
    if (samples.length > _distanceSmoothingWindow) samples.removeAt(0);
    return samples.reduce((a, b) => a + b) / samples.length;
  }

  // ── Callbacks ────────────────────────────────────────────────────────────
  void Function(bool isListening)? onListeningStateChanged;
  void Function()? onWakeWordDetected;

  /// Called with a human-readable summary of the last heard command (for UI display)
  void Function(String text)? onCommandHeard;

  /// Called when the user asks to read text ("read text", "what does this say", …).
  /// The screen owns the camera, so it's the one that actually captures the
  /// photo and navigates to the text-reading flow.
  void Function()? onReadTextRequested;

  // ── Screen-specific voice actions ────────────────────────────────────────
  // Each of these mirrors [onReadTextRequested]: the currently-mounted screen
  // sets the callback it can serve, and _parseVoiceCommand fires whichever
  // one is registered. Only one screen is visible at a time, so there's no
  // ambiguity about which handler should run.

  /// Camera screen: take a photo and analyse it for objects.
  void Function()? onTakePhotoRequested;

  /// Camera screen: toggle face recognition on/off.
  void Function()? onToggleFaceRecognitionRequested;

  /// Camera screen: navigate to the face-registration flow.
  void Function()? onOpenFaceRegistrationRequested;

  /// Home screen: navigate to the live camera screen.
  void Function()? onOpenCameraRequested;

  /// Face registration screen: begin the auto-capture sequence.
  void Function()? onStartCaptureRequested;

  /// Face registration screen: retake the captured photos.
  void Function()? onRetakeRequested;

  /// Face registration screen: confirm/submit the registration.
  void Function()? onConfirmRegistrationRequested;

  /// Face registration screen: delete a registered face by spoken name.
  void Function(String name)? onDeleteFaceRequested;

  /// Face registration screen: fill the name field from a spoken name.
  void Function(String name)? onNameProvided;

  /// Text reading / photo analysis screens: re-speak the last result.
  void Function()? onReadAgainRequested;

  /// Any screen: go back to the previous screen.
  void Function()? onGoBackRequested;

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
    final meters = _smoothDistance(nearest.label, _estimateDistance(nearest));
    final now = DateTime.now();

    // ── Hazard override ───────────────────────────────────────────────────
    // If TTS is mid-sentence but a significantly closer hazard just appeared
    // (35% nearer, under 2 m, max one override every 2 s) — cut the speech.
    final hazardOverride = _isSpeaking &&
        meters > 0 &&
        meters < 2.0 &&
        _lastAnnouncedDistance != null &&
        meters < _lastAnnouncedDistance! * 0.65 &&
        (_lastHazardOverrideTime == null ||
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
          final secMeters =
              _smoothDistance(second.label, _estimateDistance(second));
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
    final halfFovRadians = (_verticalFovDegrees * pi / 180) / 2;
    final focalLength = det.modelSize / (2 * tan(halfFovRadians));
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
    if (meters < 0.8) return 3; // very close  — re-announce every 3 s
    if (meters < 2.0) return 4; // close        — every 4 s
    if (meters < 4.0) return 6; // ahead        — every 6 s
    return 9; // far           — every 9 s
  }

  /// Compose the spoken phrase with capitalized label and natural word order.
  /// Examples: "Person, 1.5 meters, to your left"
  ///           "Warning! Chair right in front of you"
  ///           "Car far ahead"
  String _buildDistancePhrase(String label, double meters, String position) {
    // A recognised face should be greeted by name, not announced as a
    // generic "Person" — see [_nearbyPersonName].
    if (label.toLowerCase() == 'person' && _nearbyPersonName != null) {
      return _buildDistancePhraseForLabel(_nearbyPersonName!, meters, position,
          capitalize: false);
    }
    return _buildDistancePhraseForLabel(label, meters, position, capitalize: true);
  }

  String _buildDistancePhraseForLabel(
      String label, double meters, String position,
      {required bool capitalize}) {
    final cap = capitalize
        ? (label.isNotEmpty ? label[0].toUpperCase() + label.substring(1) : label)
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

  void _scheduleWakeWordPoll([Duration delay = const Duration(milliseconds: 800)]) {
    _wakeWordTimer?.cancel();
    // Small delay so any current STT session fully releases the mic
    _wakeWordTimer = Timer(delay, _pollWakeWord);
  }

  /// Delay used for the next poll after a fast-failure streak. Grows with
  /// each consecutive fast failure (capped) so a persistent problem (e.g.
  /// missing mic permission) doesn't spam the mic's start/stop beep.
  Duration _backoffDelay() {
    if (_consecutiveFastFailures < _maxFastFailuresBeforeBackoff) {
      return const Duration(milliseconds: 800);
    }
    final extra = _consecutiveFastFailures - _maxFastFailuresBeforeBackoff;
    final seconds = (2 << extra.clamp(0, 4)); // 2,4,8,16,32s
    return Duration(seconds: seconds.clamp(2, 30));
  }

  Future<void> _pollWakeWord() async {
    if (!_wakeWordEnabled ||
        _inConversation ||
        _isSpeaking ||
        _stt.isListening) {
      // Try again later
      _scheduleWakeWordPoll();
      return;
    }

    if (!_stt.isAvailable) {
      // Recognizer isn't ready (often means no mic permission). Back off
      // instead of retrying every 800ms.
      _consecutiveFastFailures++;
      _scheduleWakeWordPoll(_backoffDelay());
      return;
    }

    bool detected = false;
    final startedAt = DateTime.now();

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

    final elapsed = DateTime.now().difference(startedAt);
    if (!detected && elapsed < const Duration(milliseconds: 1200)) {
      // The session ended almost instantly — likely an error, not silence.
      _consecutiveFastFailures++;
    } else {
      _consecutiveFastFailures = 0;
    }

    // Next poll is scheduled by the onStatus callback ('done'/'notListening')
    // to ensure the mic is free. The fallback timer handles edge cases.
    if (!detected) {
      _scheduleWakeWordPoll(_backoffDelay());
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
        // The fallback spoken message is now handled centrally in
        // _parseVoiceCommand's UNRECOGNIZED branch, so this callback only
        // needs to update the on-screen log.
        startListening((unrecognized) {
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
      String command, Function(String) onUnrecognized,
      {bool isRetry = false}) async {
    // ── STOP / CANCEL ────────────────────────────────────────────────────
    if (_matchesAny(
        command, ['stop', 'cancel', 'quiet', 'silence', 'shut up'])) {
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
    else if (_matchesAny(
        command, ['battery', 'charge', 'battery level', 'how much battery'])) {
      await tellBattery();
    }
    // ── DATE ─────────────────────────────────────────────────────────────
    else if (_matchesAny(command, ['date', 'what day', "today's date"])) {
      await tellDate();
    }
    // ── READ TEXT (OCR) ──────────────────────────────────────────────────
    else if (_matchesAny(command, [
      'read text',
      'read this',
      'read that',
      'read it',
      'what does this say',
      'what does it say',
      'read the sign',
      'read the label',
    ])) {
      if (onReadTextRequested != null) {
        _speak('Reading text.');
        onReadTextRequested!();
      } else {
        _speak('Text reading is not available right now.');
      }
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
    // ── OPEN CAMERA (home screen) ────────────────────────────────────────
    else if (_matchesAny(
        command, ['open camera', 'open the camera', 'go to camera', 'launch camera'])) {
      if (onOpenCameraRequested != null) {
        onOpenCameraRequested!();
      } else {
        _speak('Camera is not available from this screen.');
      }
    }
    // ── TAKE / ANALYSE PHOTO (camera screen) ────────────────────────────
    // Broad match instead of an exhaustive phrase list: "take" + a
    // photo/picture word, or "analyse/analyze" + this/photo/picture.
    else if ((command.contains('take') &&
            (command.contains('photo') ||
                command.contains('picture') ||
                command.contains('pic'))) ||
        ((command.contains('analyse') || command.contains('analyze')) &&
            (command.contains('photo') ||
                command.contains('picture') ||
                command.contains('this')))) {
      if (onTakePhotoRequested != null) {
        onTakePhotoRequested!();
      } else {
        _speak('Taking a photo is not available from this screen.');
      }
    }
    // ── TOGGLE FACE RECOGNITION (camera screen) ─────────────────────────
    else if (command.contains('face recognition') &&
        _matchesAny(command, ['on', 'off', 'toggle', 'enable', 'disable'])) {
      if (onToggleFaceRecognitionRequested != null) {
        onToggleFaceRecognitionRequested!();
      } else {
        _speak('Face recognition is not available from this screen.');
      }
    }
    // ── REGISTER A FACE — open the registration screen ──────────────────
    else if (_matchesAny(command,
        ['register a face', 'register face', 'add a face', 'new face'])) {
      if (onOpenFaceRegistrationRequested != null) {
        onOpenFaceRegistrationRequested!();
      } else {
        _speak('Face registration is not available from this screen.');
      }
    }
    // ── START CAPTURE (face registration screen) ─────────────────────────
    else if (_matchesAny(command, [
      'start capture',
      'start capturing',
      'begin capture',
      'begin capturing',
      'start registration',
      'capture photos',
    ])) {
      if (onStartCaptureRequested != null) {
        onStartCaptureRequested!();
      } else {
        _speak('Nothing to capture right now.');
      }
    }
    // ── RETAKE (face registration screen) ────────────────────────────────
    else if (_matchesAny(command, ['retake', 'try again', 'redo photos'])) {
      if (onRetakeRequested != null) {
        onRetakeRequested!();
      } else {
        _speak('Nothing to retake right now.');
      }
    }
    // ── CONFIRM REGISTRATION (face registration screen) ──────────────────
    else if (_matchesAny(command,
        ['confirm registration', 'save face', 'confirm', 'submit'])) {
      if (onConfirmRegistrationRequested != null) {
        onConfirmRegistrationRequested!();
      } else {
        _speak('Nothing to confirm right now.');
      }
    }
    // ── DELETE A REGISTERED FACE (face registration screen) ──────────────
    else if (command.startsWith('delete ') || command.startsWith('remove ')) {
      final name = command
          .replaceFirst(RegExp(r'^(delete|remove)\s+'), '')
          .trim();
      if (name.isNotEmpty && onDeleteFaceRequested != null) {
        onDeleteFaceRequested!(name);
      } else {
        _speak('I could not tell who to delete.');
      }
    }
    // ── PROVIDE A NAME (face registration screen) ─────────────────────────
    else if (_matchesAny(command, ['my name is', 'name is', 'call them'])) {
      final name = command
          .replaceFirst(RegExp(r'^(my name is|name is|call them)\s+'), '')
          .trim();
      if (name.isNotEmpty && onNameProvided != null) {
        onNameProvided!(name);
      } else {
        _speak('I did not catch the name.');
      }
    }
    // ── READ AGAIN (text reading / photo analysis screens) ────────────────
    else if (_matchesAny(command, ['read again', 'say the text again'])) {
      if (onReadAgainRequested != null) {
        onReadAgainRequested!();
      } else if (_lastResponse != null) {
        _speak(_lastResponse!);
      } else {
        _speak('Nothing to read again.');
      }
    }
    // ── GO BACK ────────────────────────────────────────────────────────────
    else if (_matchesAny(command, ['go back', 'go back to camera', 'back to camera'])) {
      if (onGoBackRequested != null) {
        onGoBackRequested!();
      } else {
        _speak('Cannot go back from this screen.');
      }
    }
    // ── OPEN / LAUNCH APP ────────────────────────────────────────────────
    else if (command.startsWith('open ') || command.startsWith('launch ')) {
      final appName = command.replaceFirst(RegExp(r'^(open|launch)\s+'), '');
      if (appName.isNotEmpty) await launchApp(appName);
    }
    // ── NAVIGATE / GO TO ─────────────────────────────────────────────────
    else if (_matchesAny(
        command, ['navigate to', 'directions to', 'take me to', 'go to'])) {
      final dest = command
          .replaceFirst(
              RegExp(r'^(navigate to|directions to|take me to|go to)\s*'), '')
          .trim();
      await navigateTo(dest.isNotEmpty ? dest : 'your destination');
    }
    // ── SCAN STORAGE ─────────────────────────────────────────────────────
    else if (_matchesAny(
        command, ['scan storage', 'find apk', 'search storage'])) {
      await scanStorageForApks();
    }
    // ── UNRECOGNIZED — try Mistral before giving up ──────────────────────
    else {
      await _handleUnmatchedCommand(command, onUnrecognized, isRetry: isRetry);
    }
  }

  /// Called when the local keyword matching in [_parseVoiceCommand] found
  /// nothing. Tries Mistral (if configured) in two steps:
  ///   1. Ask it to map the transcript onto one of Envision's own known
  ///      commands (fixes misheard/unusual phrasings like "take a picture"
  ///      without hardcoding every variant).
  ///   2. If that also comes back empty, treat it as an open-ended
  ///      question and answer it conversationally — this doubles as the
  ///      "ask Envision anything" chatbot.
  /// Falls back to the plain "didn't understand" message if Mistral isn't
  /// configured, errors out, or [isRetry] is true (prevents infinite
  /// recursion if a classified command somehow doesn't match on replay).
  Future<void> _handleUnmatchedCommand(
    String command,
    Function(String) onUnrecognized, {
    required bool isRetry,
  }) async {
    if (command.isEmpty) {
      _speak(
          "Sorry, I didn't hear anything. Try saying 'help' for a list of commands.");
      onUnrecognized(command);
      return;
    }

    if (!isRetry && MistralService.instance.isConfigured) {
      final matched = await MistralService.instance.classifyCommand(command);
      if (matched != null) {
        await _parseVoiceCommand(matched, onUnrecognized, isRetry: true);
        return;
      }

      // No known command matched — treat it as an open-ended question.
      final reply = await MistralService.instance.chat(command);
      _speak(reply);
      _lastResponse = reply;
      onUnrecognized(command);
      return;
    }

    _speak(
        "Sorry, I didn't understand \"$command\". Try saying 'help' for a list of commands.");
    onUnrecognized(command);
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
      final dist = _smoothDistance(det.label, _estimateDistance(det));
      final pos = _getPosition(det);
      final count = labelCounts[det.label] ?? 1;

      // Group label — "3 people" vs "A person"
      final String labelStr;
      if (det.label.toLowerCase() == 'person' &&
          count == 1 &&
          _nearbyPersonName != null) {
        // Greet a single recognised face by name instead of "Person".
        labelStr = _nearbyPersonName!;
      } else if (count > 1) {
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
    final cleaned = objectName
        .replaceAll(RegExp(r'\s+(are|is|there|do you see).*'), '')
        .trim();
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
      final charging =
          state == BatteryState.charging ? ', and it is charging' : '';
      final response = 'Battery is at $level percent$charging.';
      _speak(response);
      _lastResponse = response;
    } catch (e) {
      _speak('Unable to get battery information.');
      print('Error getting battery: $e');
    }
  }

  Future<void> _tellHelp() async {
    const response = 'You can say: describe, read text, take a photo, '
        'turn face recognition on or off, register a face, open camera, '
        'start capture, retake, confirm, delete followed by a name, '
        'my name is followed by a name, read again, go back, '
        'open followed by an app name, '
        'what time is it, battery, today\'s date, '
        'how many people, navigate to a place, '
        'repeat, stop, or help. '
        'You can also just ask me a question if it is not one of these.';
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
        final top = _cachedApks
            .take(3)
            .map((e) => e['label'] ?? e['package'])
            .join(', ');
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
    _lastAnnounced
        .removeWhere((key, time) => now.difference(time).inSeconds > 12);
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
