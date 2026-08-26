import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:tensorflow_demo/services/api_service.dart';
import 'package:tensorflow_demo/services/face_database_service.dart';
import 'package:tensorflow_demo/services/face_recognition_service.dart';
import 'package:tensorflow_demo/services/voice_service.dart';

class FaceRegistrationScreen extends StatefulWidget {
  const FaceRegistrationScreen({super.key});

  @override
  State<FaceRegistrationScreen> createState() => _FaceRegistrationScreenState();
}

class _FaceRegistrationScreenState extends State<FaceRegistrationScreen> {
  CameraController? _cameraController;
  List<CameraDescription> _cameras = [];
  bool _isInitialized = false;
  bool _isCapturing = false;
  bool _isRegistering = false;

  // Multiple captured images
  List<Uint8List> _capturedImages = [];
  static const int _totalPhotos = 7;
  int _captureCountdown = 0;

  final TextEditingController _nameController = TextEditingController();
  String _status = 'Point camera at face and tap Capture';
  List<RegisteredFace> _registeredFaces = [];

  @override
  void initState() {
    super.initState();
    _initCamera();
    _loadRegisteredFaces();

    VoiceService.instance.onStartCaptureRequested = _startAutoCapture;
    VoiceService.instance.onRetakeRequested = _retake;
    VoiceService.instance.onConfirmRegistrationRequested = _registerFace;
    VoiceService.instance.onDeleteFaceRequested = _deleteFaceByName;
    VoiceService.instance.onNameProvided = _setNameFromVoice;
    VoiceService.instance.onGoBackRequested = _goBack;
  }

  void _setNameFromVoice(String name) {
    final capitalized = name.isNotEmpty
        ? name[0].toUpperCase() + name.substring(1)
        : name;
    _nameController.text = capitalized;
    VoiceService.instance.speak('Name set to $capitalized.');
  }

  Future<void> _deleteFaceByName(String spokenName) async {
    final target = spokenName.trim().toLowerCase();
    final match = _registeredFaces
        .where((f) => f.name.toLowerCase().contains(target))
        .toList();
    if (match.isEmpty) {
      VoiceService.instance.speak('No registered face matches $spokenName.');
      return;
    }
    await _deleteFace(match.first);
  }

  Future<void> _goBack() async {
    try {
      if (_cameraController?.value.isStreamingImages == true) {
        await _cameraController?.stopImageStream();
      }
      await _cameraController?.dispose();
      _cameraController = null;
    } catch (_) {}
    if (mounted) Navigator.pop(context);
  }

  Future<void> _initCamera() async {
    _cameras = await availableCameras();
    if (_cameras.isEmpty) return;

    final camera = _cameras.firstWhere(
      (c) => c.lensDirection == CameraLensDirection.back,
      orElse: () => _cameras.first,
    );

    _cameraController = CameraController(
      camera,
      ResolutionPreset.medium,
      enableAudio: false,
    );

    await _cameraController!.initialize();
    if (mounted) setState(() => _isInitialized = true);
  }

  Future<void> _loadRegisteredFaces() async {
    final faces = await FaceDatabaseService.instance.getAllFaces();
    if (mounted) setState(() => _registeredFaces = faces);
  }

  // ── Auto-capture multiple photos ─────────────────────────────────────────

  Future<void> _startAutoCapture() async {
    if (_cameraController == null || !_cameraController!.value.isInitialized) {
      return;
    }
    if (_isCapturing) return;

    setState(() {
      _isCapturing = true;
      _capturedImages = [];
      _status = 'Get ready…';
    });

    HapticFeedback.mediumImpact();

    // Countdown before starting
    for (int i = 3; i > 0; i--) {
      if (!mounted) return;
      setState(() => _status = 'Starting in $i…');
      await Future.delayed(const Duration(seconds: 1));
    }

    // Capture _totalPhotos photos with 1 second interval
    for (int i = 1; i <= _totalPhotos; i++) {
      if (!mounted) return;

      setState(() {
        _captureCountdown = i;
        _status =
            'Capturing photo $i of $_totalPhotos…\nSlightly change angle each time';
      });

      try {
        final photo = await _cameraController!.takePicture();
        final bytes = await photo.readAsBytes();
        _capturedImages.add(bytes);
        HapticFeedback.lightImpact();
        if (mounted) setState(() {});
      } catch (e) {
        debugPrint('Capture error: $e');
      }

      if (i < _totalPhotos) {
        await Future.delayed(const Duration(milliseconds: 1200));
      }
    }

    if (mounted) {
      setState(() {
        _isCapturing = false;
        _captureCountdown = 0;
        _status =
            '${_capturedImages.length} photos captured!\nEnter name and tap Register';
      });
    }
  }

  // ── Registration ──────────────────────────────────────────────────────────

  Future<void> _registerFace() async {
    final name = _nameController.text.trim();
    if (name.isEmpty) {
      setState(() => _status = 'Please enter a name');
      return;
    }
    if (_capturedImages.isEmpty) {
      setState(() => _status = 'Please capture photos first');
      return;
    }

    setState(() {
      _isRegistering = true;
      _status = 'Registering $name…\nThis may take a moment';
    });

    // Try server registration with all photos
    bool success = false;
    final serverOnline = await _checkServerOnline();

    if (serverOnline) {
      // Send all photos to server
      try {
        success = await ApiService.instance.register(name, _capturedImages);
        if (success) {
          await FaceRecognitionService.instance.refreshDatabase();
        }
      } catch (e) {
        debugPrint('Server registration error: $e');
      }
    }

    // Fall back to local registration with first photo
    if (!success) {
      success = await FaceRecognitionService.instance
          .registerFaceFromBytes(_capturedImages.first, name);
    }

    if (success) {
      VoiceService.instance.speak('$name has been registered.');
      setState(() {
        _status =
            '$name registered successfully with ${_capturedImages.length} photos!';
        _capturedImages = [];
        _nameController.clear();
      });
      await _loadRegisteredFaces();
    } else {
      setState(() => _status = 'Registration failed. Please try again.');
    }

    setState(() => _isRegistering = false);
  }

  Future<bool> _checkServerOnline() async {
    return FaceRecognitionService.instance.isServerOnline;
  }

  Future<void> _deleteFace(RegisteredFace face) async {
    await FaceDatabaseService.instance.deleteFace(face.id);
    await FaceRecognitionService.instance.refreshDatabase();
    await _loadRegisteredFaces();
    VoiceService.instance.speak('${face.name} removed.');
  }

  void _retake() {
    setState(() {
      _capturedImages = [];
      _status = 'Point camera at face and tap Capture';
    });
  }

  @override
  void dispose() {
    try {
      if (_cameraController?.value.isStreamingImages == true) {
        _cameraController?.stopImageStream();
      }
    } catch (_) {}
    _cameraController?.dispose();
    _cameraController = null;
    _nameController.dispose();
    VoiceService.instance.onStartCaptureRequested = null;
    VoiceService.instance.onRetakeRequested = null;
    VoiceService.instance.onConfirmRegistrationRequested = null;
    VoiceService.instance.onDeleteFaceRequested = null;
    VoiceService.instance.onNameProvided = null;
    VoiceService.instance.onGoBackRequested = null;
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.black,
      appBar: AppBar(
        backgroundColor: Colors.black,
        foregroundColor: Colors.white,
        title: const Text('Register Face',
            style: TextStyle(color: Colors.white, fontWeight: FontWeight.bold)),
        leading: IconButton(
          icon: const Icon(Icons.arrow_back, color: Colors.white),
          onPressed: _goBack,
        ),
      ),
      body: Column(
        children: [
          // ── Camera preview ────────────────────────────────────────────
          Expanded(
            flex: 3,
            child: _capturedImages.isNotEmpty && !_isCapturing
                ? Stack(
                    fit: StackFit.expand,
                    children: [
                      // Show grid of captured photos
                      GridView.count(
                        crossAxisCount: 3,
                        children: _capturedImages.map((bytes) {
                          return Padding(
                            padding: const EdgeInsets.all(2),
                            child: Image.memory(bytes, fit: BoxFit.cover),
                          );
                        }).toList(),
                      ),
                      Positioned(
                        top: 12,
                        right: 12,
                        child: GestureDetector(
                          onTap: _retake,
                          child: Container(
                            padding: const EdgeInsets.symmetric(
                                horizontal: 12, vertical: 6),
                            decoration: BoxDecoration(
                              color: Colors.black54,
                              borderRadius: BorderRadius.circular(20),
                            ),
                            child: const Row(
                              children: [
                                Icon(Icons.refresh,
                                    color: Colors.white, size: 16),
                                SizedBox(width: 4),
                                Text('Retake',
                                    style: TextStyle(color: Colors.white)),
                              ],
                            ),
                          ),
                        ),
                      ),
                    ],
                  )
                : _isInitialized && _cameraController != null
                    ? Stack(
                        fit: StackFit.expand,
                        children: [
                          CameraPreview(_cameraController!),
                          // Face guide oval
                          Center(
                            child: Container(
                              width: 200,
                              height: 260,
                              decoration: BoxDecoration(
                                border: Border.all(
                                    color: _isCapturing
                                        ? Colors.green
                                        : Colors.white60,
                                    width: 2),
                                borderRadius: BorderRadius.circular(120),
                              ),
                            ),
                          ),
                          // Countdown overlay
                          if (_isCapturing && _captureCountdown > 0)
                            Center(
                              child: Container(
                                width: 80,
                                height: 80,
                                decoration: BoxDecoration(
                                  color: Colors.green.withValues(alpha: 0.8),
                                  shape: BoxShape.circle,
                                ),
                                child: Center(
                                  child: Text(
                                    '$_captureCountdown',
                                    style: const TextStyle(
                                      color: Colors.white,
                                      fontSize: 36,
                                      fontWeight: FontWeight.bold,
                                    ),
                                  ),
                                ),
                              ),
                            ),
                          // Photo progress indicator
                          if (_isCapturing)
                            Positioned(
                              bottom: 16,
                              left: 0,
                              right: 0,
                              child: Row(
                                mainAxisAlignment: MainAxisAlignment.center,
                                children: List.generate(_totalPhotos, (i) {
                                  return Container(
                                    width: 12,
                                    height: 12,
                                    margin: const EdgeInsets.symmetric(
                                        horizontal: 4),
                                    decoration: BoxDecoration(
                                      shape: BoxShape.circle,
                                      color: i < _capturedImages.length
                                          ? Colors.green
                                          : Colors.white30,
                                    ),
                                  );
                                }),
                              ),
                            ),
                        ],
                      )
                    : const Center(
                        child: CircularProgressIndicator(color: Colors.white)),
          ),

          // ── Status ────────────────────────────────────────────────────
          Container(
            width: double.infinity,
            color: Colors.black,
            padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
            child: Text(
              _status,
              style: TextStyle(
                color: _status.contains('successfully')
                    ? Colors.green
                    : _status.contains('failed')
                        ? Colors.red
                        : Colors.white70,
                fontSize: 13,
              ),
              textAlign: TextAlign.center,
            ),
          ),

          // ── Buttons ───────────────────────────────────────────────────
          Container(
            color: Colors.black,
            padding: const EdgeInsets.all(16),
            child: Column(
              children: [
                if (_capturedImages.isNotEmpty && !_isCapturing) ...[
                  TextField(
                    controller: _nameController,
                    style: const TextStyle(color: Colors.white),
                    decoration: InputDecoration(
                      hintText: "Enter person's name",
                      hintStyle: const TextStyle(color: Colors.white38),
                      prefixIcon:
                          const Icon(Icons.person, color: Colors.white38),
                      enabledBorder: OutlineInputBorder(
                        borderRadius: BorderRadius.circular(12),
                        borderSide: const BorderSide(color: Colors.white24),
                      ),
                      focusedBorder: OutlineInputBorder(
                        borderRadius: BorderRadius.circular(12),
                        borderSide: const BorderSide(color: Colors.blue),
                      ),
                    ),
                    textCapitalization: TextCapitalization.words,
                    onSubmitted: (_) => _registerFace(),
                  ),
                  const SizedBox(height: 12),
                  SizedBox(
                    width: double.infinity,
                    height: 52,
                    child: ElevatedButton.icon(
                      onPressed: _isRegistering ? null : _registerFace,
                      icon: _isRegistering
                          ? const SizedBox(
                              width: 20,
                              height: 20,
                              child: CircularProgressIndicator(
                                  strokeWidth: 2, color: Colors.white))
                          : const Icon(Icons.save),
                      label: Text(
                          _isRegistering ? 'Registering…' : 'Register Face'),
                      style: ElevatedButton.styleFrom(
                        backgroundColor: Colors.blue,
                        foregroundColor: Colors.white,
                        shape: RoundedRectangleBorder(
                            borderRadius: BorderRadius.circular(12)),
                      ),
                    ),
                  ),
                ] else ...[
                  SizedBox(
                    width: double.infinity,
                    height: 52,
                    child: ElevatedButton.icon(
                      onPressed: _isCapturing ? null : _startAutoCapture,
                      icon: _isCapturing
                          ? const SizedBox(
                              width: 20,
                              height: 20,
                              child: CircularProgressIndicator(
                                  strokeWidth: 2, color: Colors.black))
                          : const Icon(Icons.camera_alt),
                      label: Text(_isCapturing
                          ? 'Capturing $_captureCountdown/$_totalPhotos…'
                          : 'Capture $_totalPhotos Photos'),
                      style: ElevatedButton.styleFrom(
                        backgroundColor: Colors.white,
                        foregroundColor: Colors.black,
                        shape: RoundedRectangleBorder(
                            borderRadius: BorderRadius.circular(12)),
                      ),
                    ),
                  ),
                  const SizedBox(height: 8),
                  Text(
                    'App will auto-capture $_totalPhotos photos\nSlightly move the camera between shots',
                    style: const TextStyle(color: Colors.white38, fontSize: 12),
                    textAlign: TextAlign.center,
                  ),
                ],
              ],
            ),
          ),

          // ── Registered faces list ─────────────────────────────────────
          if (_registeredFaces.isNotEmpty)
            Container(
              color: Colors.grey.shade900,
              constraints: const BoxConstraints(maxHeight: 140),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Padding(
                    padding: const EdgeInsets.fromLTRB(16, 10, 16, 4),
                    child: Text(
                      'Registered People (${_registeredFaces.length})',
                      style: const TextStyle(
                          color: Colors.white60,
                          fontSize: 12,
                          fontWeight: FontWeight.w600),
                    ),
                  ),
                  Expanded(
                    child: ListView.builder(
                      padding: const EdgeInsets.symmetric(horizontal: 8),
                      itemCount: _registeredFaces.length,
                      itemBuilder: (context, index) {
                        final face = _registeredFaces[index];
                        return ListTile(
                          dense: true,
                          leading: CircleAvatar(
                            backgroundColor: Colors.blue.shade800,
                            radius: 18,
                            child: Text(
                              face.name[0].toUpperCase(),
                              style: const TextStyle(
                                  color: Colors.white,
                                  fontWeight: FontWeight.bold),
                            ),
                          ),
                          title: Text(face.name,
                              style: const TextStyle(
                                  color: Colors.white, fontSize: 14)),
                          trailing: IconButton(
                            icon: const Icon(Icons.delete_outline,
                                color: Colors.red, size: 20),
                            onPressed: () => _deleteFace(face),
                          ),
                        );
                      },
                    ),
                  ),
                ],
              ),
            ),
        ],
      ),
    );
  }
}
