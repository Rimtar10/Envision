import 'dart:async';
import 'dart:developer';

import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:flutter_svg/svg.dart';
import 'package:image_picker/image_picker.dart';
import 'package:permission_handler/permission_handler.dart';
import 'package:tensorflow_demo/models/detected_object/detected_object_dm.dart';
import 'package:tensorflow_demo/models/screen_params.dart';
import 'package:tensorflow_demo/services/detector.dart';
import 'package:tensorflow_demo/services/tensorflow_service.dart';
import 'package:tensorflow_demo/services/navigation_service.dart';
import 'package:tensorflow_demo/services/voice_service.dart';
import 'package:tensorflow_demo/values/app_routes.dart';
import 'package:tensorflow_demo/widgets/box_widget.dart';

class LiveObjectDetectionScreen extends StatefulWidget {
  const LiveObjectDetectionScreen({super.key});

  @override
  State<LiveObjectDetectionScreen> createState() =>
      _LiveObjectDetectionScreenState();
}

class _LiveObjectDetectionScreenState extends State<LiveObjectDetectionScreen> {
  final _imagePicker = ImagePicker();

  String? message;

  late final AppLifecycleListener _appLifecycleListener;

  late List<CameraDescription> cameras;
  int cameraIndex = 0;

  CameraController? _cameraController;
  Detector? _detector;
  StreamSubscription? _objectDetectorStream;

  List<DetectedObjectDm>? detectedObjectList;

  Timer? _cleanupTimer;
  bool _isInitializing = false;

  // ── Accessibility / voice state ─────────────────────────────────────────
  bool _isListening = false;
  bool _wakeWordEnabled = false;
  String _voiceStatus = '';

  @override
  void initState() {
    super.initState();

    _appLifecycleListener = AppLifecycleListener(
      onResume: _onResume,
      onInactive: _onInactive,
    );
    _init();

    // Periodic cleanup of stale TTS cooldowns
    _cleanupTimer = Timer.periodic(
      const Duration(seconds: 10),
      (_) => VoiceService.instance.cleanupCooldowns(),
    );

    // ── Wire callbacks ──────────────────────────────────────────────────
    VoiceService.instance.onListeningStateChanged = (listening) {
      if (mounted) setState(() => _isListening = listening);
    };

    VoiceService.instance.onWakeWordDetected = () {
      if (mounted) {
        HapticFeedback.mediumImpact();
        setState(() {
          _voiceStatus = "I'm listening…";
          _isListening = true;
        });
      }
    };

    VoiceService.instance.onCommandHeard = (text) {
      if (mounted) {
        setState(() => _voiceStatus = text);
        Future.delayed(const Duration(seconds: 4), () {
          if (mounted) setState(() => _voiceStatus = '');
        });
      }
    };
  }

  // ─────────────────────────────────────────────────────────────────────────
  // BUILD
  // ─────────────────────────────────────────────────────────────────────────

  @override
  Widget build(BuildContext context) {
    // CRITICAL: ScreenParams.screenSize drives renderLocation scaling for all
    // bounding boxes. HomeScreen used to set this; now camera screen does it.
    ScreenParams.screenSize = MediaQuery.sizeOf(context);

    final controller = _cameraController;

    if (controller == null || !controller.value.isInitialized) {
      return Scaffold(
        backgroundColor: Colors.black,
        body: Center(
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              const CircularProgressIndicator(color: Colors.white),
              const SizedBox(height: 20),
              Text(
                message ?? 'Starting Envision…',
                style: const TextStyle(color: Colors.white, fontSize: 16),
                textAlign: TextAlign.center,
              ),
            ],
          ),
        ),
      );
    }

    return Scaffold(
      backgroundColor: Colors.black,
      // No AppBar — visually impaired users should never accidentally trigger
      // a back navigation. The app is intentionally a single-screen experience.
      body: SafeArea(
        child: Column(
          children: [
            // ── Camera + overlays (full remaining height) ───────────────
            Expanded(
              child: GestureDetector(
                // Single tap anywhere on the camera → activate mic
                onTap: _onCameraAreaTap,
                // Double tap → immediately describe the full scene
                onDoubleTap: _onCameraAreaDoubleTap,
                child: Stack(
                  fit: StackFit.expand,
                  children: [
                    // Camera preview
                    AspectRatio(
                      aspectRatio: 1 / controller.value.aspectRatio,
                      child: Stack(
                        fit: StackFit.expand,
                        children: [
                          CameraPreview(controller),
                          // Bounding boxes
                          ...?detectedObjectList?.map(
                            (obj) => Positioned.fromRect(
                              rect: obj.renderLocation,
                              child: BoxWidget.fromDetectedObject(obj),
                            ),
                          ),
                        ],
                      ),
                    ),

                    // ── Wake-word indicator (top-left) ──────────────────
                    Positioned(
                      top: 10,
                      left: 12,
                      child: AnimatedOpacity(
                        opacity: _wakeWordEnabled ? 1.0 : 0.0,
                        duration: const Duration(milliseconds: 300),
                        child: _StatusPill(
                          icon: Icons.hearing,
                          label: "Say 'Envision'",
                          color: Colors.green.shade700,
                        ),
                      ),
                    ),

                    // ── Listening indicator (top-right) ─────────────────
                    if (_isListening)
                      Positioned(
                        top: 10,
                        right: 12,
                        child: _StatusPill(
                          icon: Icons.mic,
                          label: 'Listening…',
                          color: Colors.red.shade700,
                          pulse: true,
                        ),
                      ),

                    // ── Performance overlay (top-center) ────────────────
                    if (_detector != null)
                      Positioned(
                        top: 10,
                        left: 0,
                        right: 0,
                        child: Center(
                          child: ValueListenableBuilder<PerfStats>(
                            valueListenable: _detector!.perf,
                            builder: (context, p, _) {
                              return Container(
                                padding: const EdgeInsets.symmetric(
                                    horizontal: 10, vertical: 5),
                                decoration: BoxDecoration(
                                  color: Colors.black.withValues(alpha: 0.6),
                                  borderRadius: BorderRadius.circular(10),
                                ),
                                child: Text(
                                  '${p.fps.toStringAsFixed(1)} FPS  •  '
                                  '${p.lastInferenceMs.toStringAsFixed(0)} ms  •  '
                                  '${p.detections} obj',
                                  style: const TextStyle(
                                    color: Colors.white,
                                    fontSize: 12,
                                    fontWeight: FontWeight.w600,
                                  ),
                                ),
                              );
                            },
                          ),
                        ),
                      ),

                    // ── Voice status banner (bottom of camera area) ─────
                    if (_voiceStatus.isNotEmpty)
                      Positioned(
                        bottom: 10,
                        left: 12,
                        right: 12,
                        child: Container(
                          padding: const EdgeInsets.symmetric(
                              horizontal: 14, vertical: 10),
                          decoration: BoxDecoration(
                            color: Colors.black.withValues(alpha: 0.75),
                            borderRadius: BorderRadius.circular(12),
                          ),
                          child: Row(
                            children: [
                              const Icon(Icons.chat_bubble_outline,
                                  color: Colors.white70, size: 16),
                              const SizedBox(width: 8),
                              Expanded(
                                child: Text(
                                  _voiceStatus,
                                  style: const TextStyle(
                                      color: Colors.white,
                                      fontSize: 14,
                                      height: 1.3),
                                  maxLines: 3,
                                  overflow: TextOverflow.ellipsis,
                                ),
                              ),
                            ],
                          ),
                        ),
                      ),
                  ],
                ),
              ),
            ),

            // ── Accessible bottom bar ──────────────────────────────────
            _buildAccessibleBottomBar(),
          ],
        ),
      ),
    );
  }

  // ─────────────────────────────────────────────────────────────────────────
  // BOTTOM CONTROL BAR
  // Designed for visually impaired users:
  //  • Large tap targets (min 64 px)
  //  • Mic button dominates the centre
  //  • All buttons are Semantics-labelled for screen readers
  // ─────────────────────────────────────────────────────────────────────────

  Widget _buildAccessibleBottomBar() {
    return Container(
      color: Colors.black,
      padding: const EdgeInsets.symmetric(vertical: 14, horizontal: 16),
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          // ── Main mic row ──────────────────────────────────────────────
          Row(
            children: [
              // Gallery
              Semantics(
                label: 'Analyse a photo from gallery',
                button: true,
                child: _ControlButton(
                  icon: null,
                  svgAsset: 'assets/vectors/gallery.svg',
                  onTap: _pickImageFromGallery,
                  size: 56,
                ),
              ),

              const SizedBox(width: 12),

              // ── Primary mic button (largest, centre) ──────────────────
              Expanded(
                child: Semantics(
                  label: _isListening
                      ? 'Stop listening'
                      : 'Tap to speak a command',
                  button: true,
                  child: GestureDetector(
                    onTap: _toggleVoiceCommand,
                    child: AnimatedContainer(
                      duration: const Duration(milliseconds: 200),
                      height: 64,
                      decoration: BoxDecoration(
                        color: _isListening
                            ? Colors.red.shade700
                            : Colors.white.withValues(alpha: 0.15),
                        borderRadius: BorderRadius.circular(32),
                        border: Border.all(
                          color: _isListening
                              ? Colors.red.shade300
                              : Colors.white30,
                          width: 2,
                        ),
                      ),
                      child: Row(
                        mainAxisAlignment: MainAxisAlignment.center,
                        children: [
                          Icon(
                            _isListening ? Icons.mic : Icons.mic_none,
                            color: Colors.white,
                            size: 28,
                          ),
                          const SizedBox(width: 8),
                          Text(
                            _isListening ? 'Listening…' : 'Tap to Speak',
                            style: const TextStyle(
                              color: Colors.white,
                              fontSize: 16,
                              fontWeight: FontWeight.w600,
                            ),
                          ),
                        ],
                      ),
                    ),
                  ),
                ),
              ),

              const SizedBox(width: 12),

              // Flip camera
              Semantics(
                label: 'Flip camera',
                button: true,
                child: _ControlButton(
                  icon: null,
                  svgAsset: 'assets/vectors/repeate-music.svg',
                  onTap: _flipCamera,
                  size: 56,
                ),
              ),
            ],
          ),

          const SizedBox(height: 10),

          // ── Secondary row: take photo + wake word toggle ──────────────
          Row(
            children: [
              // Take photo for analysis
              Expanded(
                child: Semantics(
                  label: 'Take photo and analyse objects',
                  button: true,
                  child: _SecondaryButton(
                    icon: Icons.camera_alt_outlined,
                    label: 'Analyse Photo',
                    onTap: _takePicture,
                  ),
                ),
              ),

              const SizedBox(width: 12),

              // Wake-word toggle
              Expanded(
                child: Semantics(
                  label: _wakeWordEnabled
                      ? "Wake word active — say Envision"
                      : "Enable wake word — say Envision to activate",
                  button: true,
                  child: _SecondaryButton(
                    icon: _wakeWordEnabled
                        ? Icons.hearing
                        : Icons.hearing_disabled,
                    label: _wakeWordEnabled ? 'Wake: ON' : 'Wake: OFF',
                    color: _wakeWordEnabled
                        ? Colors.green.shade700
                        : Colors.white24,
                    onTap: _toggleWakeWord,
                  ),
                ),
              ),
            ],
          ),

          // ── Hint text ─────────────────────────────────────────────────
          const SizedBox(height: 8),
          Text(
            'Tap camera area once to speak  •  Double-tap to describe scene',
            style: TextStyle(
              color: Colors.white.withValues(alpha: 0.45),
              fontSize: 11,
            ),
            textAlign: TextAlign.center,
          ),
        ],
      ),
    );
  }

  // ─────────────────────────────────────────────────────────────────────────
  // GESTURE HANDLERS
  // ─────────────────────────────────────────────────────────────────────────

  Future<void> _onCameraAreaTap() async {
    HapticFeedback.lightImpact();
    final voice = VoiceService.instance;
    // Tapping the camera area ALWAYS starts a command session.
    // If already in a command session, stop it first then restart.
    if (voice.isCommandListening) await voice.stopListening();
    HapticFeedback.mediumImpact();
    await voice.startListening((unrecognized) {
      if (mounted) {
        setState(() => _voiceStatus = 'Not understood — say "help" for commands.');
        Future.delayed(const Duration(seconds: 4),
            () => mounted ? setState(() => _voiceStatus = '') : null);
      }
    });
  }

  Future<void> _onCameraAreaDoubleTap() async {
    HapticFeedback.mediumImpact();
    // Describe the scene immediately on double-tap
    await VoiceService.instance.stopListening();
    await VoiceService.instance.describeScene();
    if (mounted) {
      setState(() => _voiceStatus = 'Describing scene…');
      Future.delayed(const Duration(seconds: 4),
          () => mounted ? setState(() => _voiceStatus = '') : null);
    }
  }

  Future<void> _toggleVoiceCommand() async {
    final voice = VoiceService.instance;
    // Use isCommandListening — NOT isListening, which is true during
    // background wake-word polling and would break the toggle logic.
    if (voice.isCommandListening) {
      await voice.stopListening();
    } else {
      HapticFeedback.mediumImpact();
      await voice.startListening((unrecognized) {
        if (mounted) {
          setState(() =>
              _voiceStatus = 'Not understood — say "help" for commands.');
          Future.delayed(const Duration(seconds: 4),
              () => mounted ? setState(() => _voiceStatus = '') : null);
        }
      });
    }
  }

  Future<void> _toggleWakeWord() async {
    final next = !_wakeWordEnabled;
    setState(() => _wakeWordEnabled = next);
    await VoiceService.instance.setWakeWordEnabled(next);
    HapticFeedback.selectionClick();

    final msg = next
        ? "Wake word enabled. Say 'Envision' to give a command."
        : 'Wake word disabled.';
    VoiceService.instance.speak(msg);
    if (mounted) {
      setState(() => _voiceStatus = msg);
      Future.delayed(const Duration(seconds: 4),
          () => mounted ? setState(() => _voiceStatus = '') : null);
    }
  }

  // ─────────────────────────────────────────────────────────────────────────
  // LIFECYCLE
  // ─────────────────────────────────────────────────────────────────────────

  void _onResume() => _init();

  void _onInactive() {
    try {
      _cameraController?.stopImageStream();
    } catch (_) {}
    _objectDetectorStream?.cancel();
    _objectDetectorStream = null;
    _detector?.stop();
    _detector = null;
  }

  @override
  void dispose() {
    _appLifecycleListener.dispose();
    try {
      _cameraController?.stopImageStream();
    } catch (_) {}
    _cameraController?.dispose();
    _cameraController = null;
    _objectDetectorStream?.cancel();
    _objectDetectorStream = null;
    _detector?.stop();
    _cleanupTimer?.cancel();
    // Leave wake word state as-is — the screen is the whole app.
    super.dispose();
  }

  // ─────────────────────────────────────────────────────────────────────────
  // INITIALIZATION
  // ─────────────────────────────────────────────────────────────────────────

  Future<void> _init() async {
    if (_isInitializing) return;
    _isInitializing = true;
    try {
      var modelReady = false;
      var voiceReady = false;

      try {
        await TensorflowService.ssdMobileNet.ensureInitialized();
        await TensorflowService.accessibilityModel.ensureInitialized();
        modelReady = true;
      } catch (e) {
        message = 'Model failed to load: $e';
        if (mounted) setState(() {});
      }

      try {
        await VoiceService.instance.ensureInitialized();
        voiceReady = true;
      } catch (e) {
        message = 'Voice features unavailable: $e';
        if (mounted) setState(() {});
      }

      final hasPermission = await _requestPermissions();
      if (!hasPermission) return;

      if (_cameraController == null || !_cameraController!.value.isInitialized) {
        await _initializeCamera();
      }
      if (_cameraController == null || !_cameraController!.value.isInitialized) {
        return;
      }

      if (modelReady && _detector == null) {
        await _initializeDetector();
      }

      if (!_cameraController!.value.isStreamingImages) {
        await _cameraController!.startImageStream(onLatestImageAvailable);
      }

      final size = _cameraController?.value.previewSize;
      if (size != null) {
        ScreenParams.previewSize = size;
      }

      if (mounted) {
        setState(() {});
        // Speak welcome message once camera is fully ready
        VoiceService.instance.speakWelcome();
      }

      // Wake word is deliberately OFF by default.
      //
      // The poller keeps an Android SpeechRecognizer session open almost
      // continuously, which (a) competes with obstacle announcements for the
      // mic, (b) streams ambient audio to a cloud ASR, and (c) drains battery.
      // The user enables it from the 'Wake: OFF' button when they want it.
      // Replace the polling loop with an on-device wake-word engine
      // (Porcupine / openWakeWord) before considering enabling this by default.
      if (voiceReady) {
        if (mounted) setState(() => _wakeWordEnabled = false);
      }
    } finally {
      _isInitializing = false;
    }
  }

  Future<bool> _requestPermissions() async {
    final cameraStatus = await Permission.camera.status;
    final micStatus = await Permission.microphone.status;
    if (cameraStatus.isGranted && micStatus.isGranted) return true;

    final results =
        await [Permission.camera, Permission.microphone].request();
    final cameraOk = results[Permission.camera]?.isGranted ?? false;
    final micOk = results[Permission.microphone]?.isGranted ?? false;

    if (cameraOk && micOk) return true;

    if (!cameraOk) {
      message = (results[Permission.camera]?.isPermanentlyDenied ?? false)
          ? 'Camera permission permanently denied.\nPlease enable it in Settings.'
          : 'Camera permission denied.';
    } else {
      message = (results[Permission.microphone]?.isPermanentlyDenied ?? false)
          ? 'Microphone permission permanently denied.\nPlease enable it in Settings.'
          : 'Microphone permission denied.';
    }

    if (mounted) setState(() {});
    log('Permissions: cam=${results[Permission.camera]}, mic=${results[Permission.microphone]}');
    return false;
  }

  Future<void> _initializeCamera() async {
    if (_cameraController != null && _cameraController!.value.isInitialized) {
      return;
    }

    final existingController = _cameraController;
    if (existingController != null) {
      try {
        await existingController.dispose();
      } catch (_) {}
    }

    try {
      cameras = await availableCameras();
    } catch (e) {
      message = 'Failed to get cameras: $e';
      if (mounted) setState(() {});
      return;
    }
    if (cameras.isEmpty) {
      message = 'No camera available.';
      if (mounted) setState(() {});
      return;
    }

    cameraIndex = 0;
    _cameraController = CameraController(
      cameras[cameraIndex],
      ResolutionPreset.medium, // balance: sharper than low, far cheaper than high; the model still
      // letterboxes to 640 so detection is unchanged. Drop to .medium/.low if a
      // low-end device (e.g. Galaxy Tab A7) can't hold preview+stream at high.
      enableAudio: false,
    );
    try {
      await _cameraController?.initialize();
    } catch (e) {
      message = 'Failed to initialize camera: $e';
      _cameraController = null;
      if (mounted) setState(() {});
    }
  }

  Future<void> _initializeDetector() async {
    final sensorOrientation = cameras[cameraIndex].sensorOrientation;
    final detector = await Detector.start(sensorOrientation: sensorOrientation);
    setState(() {
      _detector = detector;
      _objectDetectorStream =
          detector.resultsStream.listen((detectedObjects) {
        if (mounted) setState(() => detectedObjectList = detectedObjects);
        VoiceService.instance.announceDetections(detectedObjects);
      });
    });
  }

  // ─────────────────────────────────────────────────────────────────────────
  // CAMERA ACTIONS
  // ─────────────────────────────────────────────────────────────────────────

  void _flipCamera() {
    if (cameras.length <= 1) {
      VoiceService.instance.speak('Only one camera available.');
      return;
    }
    cameraIndex = cameraIndex == 1 ? 0 : 1;
    try {
      _cameraController?.stopImageStream();
    } catch (_) {}
    _cameraController?.dispose();
    _cameraController = CameraController(
      cameras[cameraIndex],
      ResolutionPreset.medium, // balance: sharper than low, far cheaper than high; the model still
      // letterboxes to 640 so detection is unchanged. Drop to .medium/.low if a
      // low-end device (e.g. Galaxy Tab A7) can't hold preview+stream at high.
      enableAudio: false,
    )..initialize().then((_) async {
        await _cameraController?.startImageStream(onLatestImageAvailable);
        ScreenParams.previewSize =
            _cameraController?.value.previewSize ?? ScreenParams.previewSize;
        if (mounted) setState(() {});
      }).catchError((e) {
      log('Failed to flip camera: $e');
    });
  }

  Future<void> _takePicture() async {
    VoiceService.instance.speak('Taking photo.');
    try {
      await _cameraController?.stopImageStream();
    } catch (_) {}
    final captured = await _cameraController?.takePicture();
    final bytes = await captured?.readAsBytes();
    if (bytes != null && bytes.isNotEmpty) {
      NavigationService.instance
          .pushNamed(AppRoutes.photoAnalyzedScreen, arguments: bytes);
    }
  }

  Future<void> _pickImageFromGallery() async {
    final result = await _imagePicker.pickImage(source: ImageSource.gallery);
    final bytes = await result?.readAsBytes();
    if (bytes != null && bytes.isNotEmpty) {
      NavigationService.instance
          .pushNamed(AppRoutes.photoAnalyzedScreen, arguments: bytes);
    }
  }

  void onLatestImageAvailable(CameraImage cameraImage) {
    if (!mounted) return;
    final controller = _cameraController;
    if (controller == null || !controller.value.isInitialized) return;
    _detector?.processFrame(cameraImage);
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// HELPER WIDGETS
// ─────────────────────────────────────────────────────────────────────────────

/// Small rounded pill with icon + label (wake word / listening indicators).
class _StatusPill extends StatefulWidget {
  const _StatusPill({
    required this.icon,
    required this.label,
    required this.color,
    this.pulse = false,
  });

  final IconData icon;
  final String label;
  final Color color;
  final bool pulse;

  @override
  State<_StatusPill> createState() => _StatusPillState();
}

class _StatusPillState extends State<_StatusPill>
    with SingleTickerProviderStateMixin {
  late final AnimationController _ctrl;

  @override
  void initState() {
    super.initState();
    _ctrl = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 700),
    );
    if (widget.pulse) _ctrl.repeat(reverse: true);
  }

  @override
  void dispose() {
    _ctrl.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    Widget pill = Container(
      padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 5),
      decoration: BoxDecoration(
        color: widget.color.withValues(alpha: 0.85),
        borderRadius: BorderRadius.circular(20),
      ),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          Icon(widget.icon, color: Colors.white, size: 14),
          const SizedBox(width: 5),
          Text(
            widget.label,
            style: const TextStyle(
                color: Colors.white,
                fontSize: 12,
                fontWeight: FontWeight.w600),
          ),
        ],
      ),
    );

    if (!widget.pulse) return pill;
    return FadeTransition(opacity: _ctrl, child: pill);
  }
}

/// Square icon button for the bottom bar.
class _ControlButton extends StatelessWidget {
  const _ControlButton({
    required this.onTap,
    required this.size,
    this.icon,
    this.svgAsset,
  });

  final VoidCallback onTap;
  final double size;
  final IconData? icon;
  final String? svgAsset;

  @override
  Widget build(BuildContext context) {
    return GestureDetector(
      onTap: onTap,
      child: Container(
        width: size,
        height: size,
        decoration: BoxDecoration(
          color: Colors.white.withValues(alpha: 0.12),
          borderRadius: BorderRadius.circular(16),
        ),
        child: Center(
          child: svgAsset != null
              ? SvgPicture.asset(svgAsset!,
                  width: 26,
                  height: 26,
                  colorFilter: const ColorFilter.mode(
                      Colors.white, BlendMode.srcIn))
              : Icon(icon, color: Colors.white, size: 26),
        ),
      ),
    );
  }
}

/// Wide secondary button with icon + text label.
class _SecondaryButton extends StatelessWidget {
  const _SecondaryButton({
    required this.icon,
    required this.label,
    required this.onTap,
    this.color,
  });

  final IconData icon;
  final String label;
  final VoidCallback onTap;
  final Color? color;

  @override
  Widget build(BuildContext context) {
    return GestureDetector(
      onTap: onTap,
      child: Container(
        height: 48,
        decoration: BoxDecoration(
          color: color ?? Colors.white.withValues(alpha: 0.12),
          borderRadius: BorderRadius.circular(12),
        ),
        child: Row(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Icon(icon, color: Colors.white, size: 20),
            const SizedBox(width: 6),
            Text(
              label,
              style: const TextStyle(
                  color: Colors.white,
                  fontSize: 13,
                  fontWeight: FontWeight.w600),
            ),
          ],
        ),
      ),
    );
  }
}
