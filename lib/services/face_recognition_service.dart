import 'dart:async';
import 'dart:io';
import 'dart:math';
import 'dart:typed_data';

import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
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
  final bool fromServer; // true = DeepFace server, false = on-device

  const FaceResult({
    required this.boundingBox,
    required this.name,
    required this.confidence,
    this.fromServer = false,
  });

  bool get isKnown => name != null && name != 'Unknown';
}

/// Real-time face detection (ML Kit) + recognition (MobileFaceNet on-device).
///
/// THE APP RECOGNISES PEOPLE WITHOUT THE SERVER.
/// ---------------------------------------------
/// Registration always writes to the phone's own database, and matching always
/// runs on-device first. The DeepFace server is a bonus: when it happens to be
/// reachable it gets a second opinion on faces the phone could not place. Turn
/// the server off, walk out of Wi-Fi range, reinstall the server — none of it
/// stops recognition working.
///
/// THE BUG THIS REPLACES
/// ---------------------
/// Registration used to embed `copyResizeCropSquare(wholePhoto, 112)` — the
/// centre square of the entire camera frame, chest and background included —
/// while recognition embedded a tight crop of the detected face. Those two
/// images produce embeddings in effectively unrelated regions of the space, so
/// the distance between them was always far past any sane threshold and nobody
/// could ever be recognised. Both paths now run the SAME
/// detect → square-crop-with-margin → 112×112 → embed → L2-normalise pipeline,
/// which is the whole reason it works now.
class FaceRecognitionService {
  static final FaceRecognitionService instance = FaceRecognitionService._();
  FaceRecognitionService._();

  FaceDetector? _faceDetector; // fast mode, for live frames
  Interpreter? _interpreter; // MobileFaceNet

  bool _isInitialized = false;
  bool _isProcessing = false;

  List<RegisteredFace> _registeredFaces = [];

  // ── Tunables ──────────────────────────────────────────────────────────────

  /// MobileFaceNet's input resolution.
  static const int _faceSize = 112;

  /// Pixel normalisation. Must be identical for enrolment and recognition.
  static const double _pixelMean = 127.5;
  static const double _pixelStd = 128.0;

  /// Padding added around ML Kit's box before cropping. ML Kit boxes are tight
  /// to the face; MobileFaceNet was trained on crops with a little forehead and
  /// chin, so cropping tight measurably hurts.
  static const double _cropMargin = 0.25;

  /// Match threshold, as Euclidean distance between L2-normalised embeddings.
  /// That distance runs 0 (identical) to 2 (opposite). Same person typically
  /// lands under 0.9, different people above 1.2.
  ///
  /// Raise it if known people are being called "Unknown"; LOWER it if the app
  /// confidently says the wrong name — for an app that announces names out
  /// loud, a confident mistake is worse than an admission of ignorance. Every
  /// comparison is logged with its distance so you can tune this from real
  /// numbers instead of guessing.
  static const double _matchThreshold = 1.05;

  /// The runner-up must be at least this much further away than the winner.
  /// Without it, two people who look alike take turns being announced.
  static const double _minMargin = 0.05;

  /// Embedding width, read from the model at load time rather than hardcoded.
  int _embeddingDim = 192;

  /// Camera sensor orientation in degrees, set by the camera screen. ML Kit
  /// needs to be told how the frame is rotated or it looks for upright faces
  /// in a sideways image and finds none.
  int _sensorOrientation = 90;
  set sensorOrientation(int degrees) {
    _sensorOrientation = degrees;
    _lockedRotation = null;
    _rotationScores.clear();
    _probeFrames = 0;
  }

  /// Which way to rotate the RGB frame before cropping — DECIDED BY EXPERIMENT,
  /// not by assumption.
  ///
  /// ML Kit reports boxes in the upright frame. To cut that same region out of
  /// our own copy we have to rotate it the same way — and "rotate 90°" is
  /// ambiguous: the `image` package and ML Kit could disagree on direction. Get
  /// it wrong and the crop comes from the diagonally opposite corner of the
  /// frame every time: a wall, a shoulder, the ceiling. Detection still works,
  /// a box still appears, and every distance sits stubbornly around 1.2–1.3 —
  /// exactly the symptom of a face that is never recognised.
  ///
  /// So rather than reason about it, the first few frames with a face present
  /// are embedded BOTH ways and scored against the enrolled people. Whichever
  /// rotation produces the closer match wins and is latched for the session.
  int? _lockedRotation;
  final Map<int, double> _rotationScores = {};
  int _probeFrames = 0;
  static const int _probeFramesNeeded = 4;
  List<int> get _rotationCandidates =>
      [_sensorOrientation, (_sensorOrientation + 180) % 360];

  // ── Announcement cooldown ─────────────────────────────────────────────────
  final Map<String, DateTime> _lastAnnounced = {};
  static const _announcementCooldown = Duration(seconds: 5);

  // ── Server (optional second opinion) ──────────────────────────────────────
  DateTime? _lastServerCheck;
  static const _serverCheckInterval = Duration(seconds: 10);

  /// The server is asked about an unrecognised face at most this often. It used
  /// to be queried once per face per frame, which meant an HTTP round trip
  /// inside the camera loop.
  DateTime? _lastServerQuery;
  static const _serverQueryInterval = Duration(seconds: 2);

  bool get isInitialized => _isInitialized;
  bool get isServerOnline => ApiService.instance.isOnline;

  /// How many people are enrolled on this phone. Recognition is possible
  /// offline whenever this is greater than zero.
  int get registeredCount => _registeredFaces.length;

  /// The most recent 112x112 crop that was actually fed to MobileFaceNet.
  ///
  /// Shown as a thumbnail on the camera screen. A recognition pipeline can look
  /// perfectly healthy in the logs while feeding the model a picture of a wall,
  /// and no amount of threshold tuning fixes that. Being able to SEE the exact
  /// pixels being embedded turns an argument into an observation.
  Uint8List? _lastCropJpeg;
  Uint8List? get debugLastCrop => _lastCropJpeg;

  // ── Initialization ────────────────────────────────────────────────────────

  Future<void> initialize() async {
    if (_isInitialized) return;

    _faceDetector = FaceDetector(
      options: FaceDetectorOptions(
        performanceMode: FaceDetectorMode.fast,
        enableLandmarks: false,
        enableContours: false,
        enableTracking: false,
        enableClassification: false,
      ),
    );

    try {
      _interpreter = await Interpreter.fromAsset('assets/mobilefacenet.tflite');
      final outShape = _interpreter!.getOutputTensor(0).shape;
      if (outShape.isNotEmpty && outShape.last > 0) {
        _embeddingDim = outShape.last;
      }
      debugPrint('[FaceRecognition] MobileFaceNet loaded — '
          'embedding dim $_embeddingDim');
    } catch (e) {
      debugPrint('[FaceRecognition] Failed to load MobileFaceNet: $e');
    }

    await refreshDatabase();
    _isInitialized = true;
    debugPrint('[FaceRecognition] Ready — offline recognition '
        '${_interpreter != null ? "available" : "UNAVAILABLE (model failed to load)"}');

    _checkServerInBackground();
  }

  void _checkServerInBackground() {
    Future.delayed(const Duration(seconds: 2), () async {
      final online = await ApiService.instance.checkStatus();
      debugPrint('[FaceRecognition] Server online: $online '
          '(recognition does not depend on it)');
    });
  }

  Future<void> refreshDatabase() async {
    _registeredFaces = await FaceDatabaseService.instance.getAllFaces();
    debugPrint('[FaceRecognition] ${_registeredFaces.length} people enrolled');

    // Audit what is actually stored, every launch. Without this, a database
    // written by an older broken build looks identical to a good one, and you
    // end up tuning a threshold against embeddings that were never valid.
    for (final person in _registeredFaces) {
      final dim = person.embedding.length;
      final dimNote = dim == _embeddingDim ? '' : '  !! DIMENSION MISMATCH '
          '(stored $dim, model expects $_embeddingDim) -- re-register';
      if (person.samples.length < 2) {
        debugPrint('[FaceRecognition]   ${person.name}: ${person.shots} shot(s), '
            'no per-shot data (old entry -- re-register)$dimNote');
        continue;
      }
      final spread = _spread(person.samples);
      final verdict = spread <= 0.60
          ? 'GOOD'
          : spread <= 0.80
              ? 'weak'
              : 'BAD -- crops are not on the face, re-register';
      debugPrint('[FaceRecognition]   ${person.name}: ${person.samples.length} '
          'shots, self-spread ${spread.toStringAsFixed(3)} ($verdict)$dimNote');
    }
  }

  /// Mean pairwise distance within one person's own enrolment shots.
  double _spread(List<List<double>> samples) {
    var total = 0.0;
    var pairs = 0;
    for (var i = 0; i < samples.length; i++) {
      for (var j = i + 1; j < samples.length; j++) {
        total += _euclidean(samples[i], samples[j]);
        pairs++;
      }
    }
    return pairs == 0 ? 0.0 : total / pairs;
  }

  /// Refresh the cached online flag in the background, at most every 10 s.
  /// Never awaited from the frame path.
  Future<void> _refreshServerFlag() async {
    final now = DateTime.now();
    if (_lastServerCheck != null &&
        now.difference(_lastServerCheck!) < _serverCheckInterval) {
      return;
    }
    _lastServerCheck = now;
    try {
      await ApiService.instance.checkStatus();
    } catch (_) {}
  }

  // ── Frame processing ──────────────────────────────────────────────────────

  Future<List<FaceResult>> processFrame(CameraImage cameraImage) async {
    if (!_isInitialized || _isProcessing) return [];
    if (_faceDetector == null) return [];
    _isProcessing = true;

    try {
      final nv21 = _toNv21(cameraImage);
      final inputImage = _buildInputImage(cameraImage, nv21);
      if (inputImage == null) return [];

      final faces = await _faceDetector!.processImage(inputImage);
      _logDetection(faces.length);
      if (faces.isEmpty) return [];

      // The full-frame YUV→RGB conversion is pure Dart and costs real
      // milliseconds, so it is deferred until a face is actually present and
      // then done once for all faces in the frame.
      img.Image? base;
      if (_interpreter != null || isServerOnline) {
        base = _convertCameraImage(cameraImage, nv21); // not yet rotated
      }
      unawaited(_refreshServerFlag());

      if (base != null && _interpreter != null && _registeredFaces.isNotEmpty) {
        _calibrateRotation(base, faces.first.boundingBox);
      }
      final rotation = _lockedRotation ?? _sensorOrientation;
      final rgb = base == null ? null : img.copyRotate(base, angle: rotation);

      final scaleX = ScreenParams.screenSize.width / cameraImage.width;
      final scaleY = ScreenParams.screenSize.height / cameraImage.height;

      final results = <FaceResult>[];

      for (final face in faces) {
        final box = face.boundingBox;
        final screenBox = Rect.fromLTRB(
          box.left * scaleX,
          box.top * scaleY,
          box.right * scaleX,
          box.bottom * scaleY,
        );

        String? matchedName;
        double matchScore = 0.0;
        bool fromServer = false;

        // ── 1. On-device first. This is the path that works with no network.
        if (rgb != null && _interpreter != null) {
          try {
            final crop = _cropFace(rgb, box);
            if (crop != null) {
              _lastCropJpeg = Uint8List.fromList(img.encodeJpg(crop, quality: 80));
              final match = _findMatch(_embed(crop));
              if (match != null) {
                matchedName = match.name;
                matchScore = match.distance;
              }
            }
          } catch (e) {
            debugPrint('[FaceRecognition] On-device match failed: $e');
          }
        }

        // ── 2. Only if the phone could not place this face, and only every
        //       few seconds, ask the server for a second opinion.
        // Deliberately reads the CACHED flag rather than awaiting a fresh
        // check: with the server switched off, ApiService takes its full 3 s
        // connection timeout to fail, and awaiting that inside the frame loop
        // stalls face processing for three seconds at a time.
        if (matchedName == null && rgb != null && _shouldQueryServer()) {
          if (isServerOnline) {
            try {
              _lastServerQuery = DateTime.now();
              final jpeg = Uint8List.fromList(img.encodeJpg(rgb, quality: 80));
              final result = await ApiService.instance.recognize(jpeg);
              if (result != null && result.isKnown) {
                matchedName = result.name;
                matchScore = result.confidence;
                fromServer = true;
                debugPrint('[FaceRecognition] Server matched ${result.name}');
              }
            } catch (e) {
              debugPrint('[FaceRecognition] Server lookup failed: $e');
            }
          }
        }

        results.add(FaceResult(
          boundingBox: screenBox,
          name: matchedName,
          confidence: matchScore,
          fromServer: fromServer,
        ));
      }

      return results;
    } catch (e) {
      debugPrint('[FaceRecognition] Frame error: $e');
      return [];
    } finally {
      _isProcessing = false;
    }
  }

  /// Prints how many faces ML Kit is seeing, at most once a second, so "no box
  /// appears" can be told apart from "a box appears but says Unknown".
  DateTime? _lastDetectionLog;
  int _lastFaceCount = -1;
  void _logDetection(int count) {
    final now = DateTime.now();
    final changed = count != _lastFaceCount;
    final due = _lastDetectionLog == null ||
        now.difference(_lastDetectionLog!) > const Duration(seconds: 1);
    if (!changed && !due) return;
    _lastDetectionLog = now;
    _lastFaceCount = count;
    debugPrint('[FaceRecognition] ML Kit sees $count face(s); '
        '$registeredCount enrolled');
  }

  /// Try both plausible rotations, keep the one that actually matches.
  void _calibrateRotation(img.Image base, Rect box) {
    if (_lockedRotation != null) return;

    for (final angle in _rotationCandidates) {
      try {
        final crop = _cropFace(img.copyRotate(base, angle: angle), box);
        if (crop == null) continue;
        final match = _nearestDistance(_embed(crop));
        _rotationScores[angle] = (_rotationScores[angle] ?? 0) + match;
      } catch (_) {
        // A candidate that cannot even be cropped simply scores nothing.
      }
    }

    _probeFrames++;
    if (_probeFrames < _probeFramesNeeded || _rotationScores.isEmpty) return;

    final ranked = _rotationScores.entries.toList()
      ..sort((a, b) => a.value.compareTo(b.value));
    _lockedRotation = ranked.first.key;

    final summary = ranked
        .map((e) => '${e.key}deg avg '
            '${(e.value / _probeFrames).toStringAsFixed(3)}')
        .join(' | ');
    debugPrint('[FaceRecognition] rotation calibrated -> ${_lockedRotation}deg '
        '($summary)');

    final bestAvg = ranked.first.value / _probeFrames;
    if (bestAvg > 1.15) {
      debugPrint('[FaceRecognition] NOTE: even the better rotation averages '
          '${bestAvg.toStringAsFixed(3)}. If the enrolment self-spread above '
          'was GOOD, this person genuinely is not the enrolled one; if it was '
          'BAD or missing, re-register.');
    }
  }

  /// Distance to the closest enrolled sample, ignoring the threshold.
  double _nearestDistance(List<double> probe) {
    var best = double.infinity;
    for (final person in _registeredFaces) {
      final candidates =
          person.samples.isNotEmpty ? person.samples : [person.embedding];
      for (final c in candidates) {
        if (c.isEmpty) continue;
        final d = _euclidean(probe, c);
        if (d < best) best = d;
      }
    }
    return best.isFinite ? best : 2.0;
  }

  bool _shouldQueryServer() {
    final last = _lastServerQuery;
    return last == null ||
        DateTime.now().difference(last) > _serverQueryInterval;
  }

  // ── Registration ──────────────────────────────────────────────────────────

  /// Enrol a person from every photo captured during registration.
  ///
  /// Each photo is independently face-detected and embedded; photos with no
  /// detectable face are skipped rather than enrolled as garbage. Returns true
  /// if at least one usable shot was stored.
  ///
  /// More shots is better, and costs nothing at match time — the extra
  /// embeddings are a few hundred bytes each and matching takes the closest.
  /// One shot works; seven of the same person at slightly different angles is
  /// noticeably more reliable, especially in changing light. If registration
  /// feels slow, reduce the count in the registration screen — but understand
  /// that you are trading recognition reliability for enrolment speed.
  Future<bool> registerFaceFromPhotos(
    List<Uint8List> photos,
    String name, {
    bool alsoRegisterOnServer = true,
  }) async {
    if (photos.isEmpty) return false;

    if (_interpreter == null) {
      debugPrint('[FaceRecognition] Cannot enrol — MobileFaceNet not loaded');
      return false;
    }

    // A more careful detector than the live one. Enrolment happens once and is
    // not on the frame budget, so accuracy is free here.
    final enrolDetector = FaceDetector(
      options: FaceDetectorOptions(
        performanceMode: FaceDetectorMode.accurate,
        enableLandmarks: false,
        enableContours: false,
        enableTracking: false,
      ),
    );

    final samples = <List<double>>[];
    int skipped = 0;

    try {
      for (var i = 0; i < photos.length; i++) {
        try {
          // bakeOrientation is not optional here. takePicture() writes a JPEG
          // whose EXIF says "rotate me"; ML Kit's fromFilePath APPLIES that tag
          // before reporting box coordinates, but img.decodeImage does NOT.
          // Without this line the box describes an upright face while the image
          // is still sideways, so the crop lands on an ear, a shoulder or the
          // wall -- and the resulting embedding is of that, not of the person.
          final decoded = img.decodeImage(photos[i]);
          if (decoded == null) {
            skipped++;
            continue;
          }
          final full = img.bakeOrientation(decoded);

          final box = await _detectLargestFace(enrolDetector, photos[i]);
          if (box == null) {
            skipped++;
            debugPrint('[FaceRecognition] Shot ${i + 1}: no face found, skipped');
            continue;
          }

          final crop = _cropFace(full, box);
          if (crop == null) {
            skipped++;
            continue;
          }

          samples.add(_embed(crop));

          // Saved for inspection, not for matching.
          await FaceDatabaseService.instance
              .saveFaceCrop(name, i, img.encodeJpg(crop, quality: 92));
        } catch (e) {
          skipped++;
          debugPrint('[FaceRecognition] Shot ${i + 1} failed: $e');
        }
      }
    } finally {
      await enrolDetector.close();
    }

    if (samples.isEmpty) {
      debugPrint('[FaceRecognition] Enrolment failed for $name — '
          'no face detected in any of the ${photos.length} photos');
      return false;
    }

    _reportEnrolmentQuality(name, samples);

    await FaceDatabaseService.instance.registerFace(
      name,
      _meanTemplate(samples),
      samples: samples,
    );
    await refreshDatabase();

    debugPrint('[FaceRecognition] Enrolled $name from ${samples.length}'
        '/${photos.length} photos${skipped > 0 ? " ($skipped skipped)" : ""} — '
        'works offline');

    // Best-effort extra: keep the server's own database in step when it is up.
    // Its success or failure does not affect the result.
    if (alsoRegisterOnServer) {
      try {
        if (await ApiService.instance.forceCheckStatus()) {
          await ApiService.instance.register(name, photos);
        }
      } catch (e) {
        debugPrint('[FaceRecognition] Server enrolment skipped: $e');
      }
    }

    return true;
  }

  /// How far apart are this person's own enrolment shots?
  ///
  /// This is the measurement that says whether a recognition failure is an
  /// enrolment problem or a threshold problem. Seven photos of one face, taken
  /// seconds apart, should cluster tightly -- a mean well under 0.6. If they
  /// are spread as far apart as two different people would be, the crops are
  /// wrong and no threshold will save it.
  void _reportEnrolmentQuality(String name, List<List<double>> samples) {
    if (samples.length < 2) return;
    var total = 0.0, worst = 0.0;
    var pairs = 0;
    for (var i = 0; i < samples.length; i++) {
      for (var j = i + 1; j < samples.length; j++) {
        final d = _euclidean(samples[i], samples[j]);
        total += d;
        if (d > worst) worst = d;
        pairs++;
      }
    }
    final mean = total / pairs;
    debugPrint('[FaceRecognition] enrolment spread for $name: '
        'mean=${mean.toStringAsFixed(3)} worst=${worst.toStringAsFixed(3)} '
        '(want mean < 0.60)');
    if (mean > 0.8) {
      debugPrint('[FaceRecognition] WARNING: $name\'s own photos are as far '
          'apart as different people would be. The crops are probably not '
          'centred on the face -- recognition will not work reliably.');
    }
  }

  /// Single-photo convenience wrapper.
  Future<bool> registerFaceFromBytes(Uint8List imageBytes, String name) =>
      registerFaceFromPhotos([imageBytes], name);

  /// ML Kit on a still JPEG. Returns the largest face's box, or null.
  Future<Rect?> _detectLargestFace(FaceDetector detector, Uint8List jpeg) async {
    final file = File('${Directory.systemTemp.path}/'
        'enrol_${DateTime.now().microsecondsSinceEpoch}.jpg');
    try {
      await file.writeAsBytes(jpeg, flush: true);
      final faces =
          await detector.processImage(InputImage.fromFilePath(file.path));
      if (faces.isEmpty) return null;
      // Largest = closest to the camera = the person being registered, not
      // somebody in the background.
      return faces
          .map((f) => f.boundingBox)
          .reduce((a, b) => a.width * a.height >= b.width * b.height ? a : b);
    } catch (e) {
      debugPrint('[FaceRecognition] Still detection failed: $e');
      return null;
    } finally {
      try {
        if (await file.exists()) await file.delete();
      } catch (_) {}
    }
  }

  // ── Embedding ─────────────────────────────────────────────────────────────

  /// Square crop around [box] with margin, resized to the model's input.
  /// Square matters: a stretched face embeds differently from a round one.
  img.Image? _cropFace(img.Image full, Rect box) {
    final side = max(box.width, box.height) * (1 + _cropMargin);
    if (side < 16) return null;

    final half = side / 2;
    var left = (box.center.dx - half).round();
    var top = (box.center.dy - half).round();
    var size = side.round();

    if (left < 0) left = 0;
    if (top < 0) top = 0;
    if (left + size > full.width) size = full.width - left;
    if (top + size > full.height) size = full.height - top;
    if (size < 16) return null;

    final crop = img.copyCrop(full, x: left, y: top, width: size, height: size);
    return img.copyResize(
      crop,
      width: _faceSize,
      height: _faceSize,
      interpolation: img.Interpolation.linear,
    );
  }

  /// Run MobileFaceNet and L2-normalise the result.
  ///
  /// Normalising is what makes one fixed threshold meaningful: without it the
  /// distance between two embeddings depends on how brightly lit the face was,
  /// and no single cutoff can work for both a sunny street and a dim hallway.
  List<double> _embed(img.Image face) {
    final input = _imageToFloat32(face).reshape([1, _faceSize, _faceSize, 3]);
    final output = List.generate(1, (_) => List.filled(_embeddingDim, 0.0));
    _interpreter!.run(input, output);
    return _l2Normalise(List<double>.from(output[0]));
  }

  Float32List _imageToFloat32(img.Image image) {
    final buffer = Float32List(_faceSize * _faceSize * 3);
    var i = 0;
    for (var y = 0; y < _faceSize; y++) {
      for (var x = 0; x < _faceSize; x++) {
        final p = image.getPixel(x, y);
        buffer[i++] = (p.r - _pixelMean) / _pixelStd;
        buffer[i++] = (p.g - _pixelMean) / _pixelStd;
        buffer[i++] = (p.b - _pixelMean) / _pixelStd;
      }
    }
    return buffer;
  }

  List<double> _l2Normalise(List<double> v) {
    var sum = 0.0;
    for (final x in v) {
      sum += x * x;
    }
    final norm = sqrt(sum);
    if (norm == 0 || norm.isNaN) return v;
    return [for (final x in v) x / norm];
  }

  /// Mean of the sample embeddings, re-normalised.
  List<double> _meanTemplate(List<List<double>> samples) {
    if (samples.length == 1) return samples.first;
    final dim = samples.first.length;
    final mean = List<double>.filled(dim, 0.0);
    for (final s in samples) {
      for (var i = 0; i < dim && i < s.length; i++) {
        mean[i] += s[i];
      }
    }
    for (var i = 0; i < dim; i++) {
      mean[i] /= samples.length;
    }
    return _l2Normalise(mean);
  }

  // ── Matching ──────────────────────────────────────────────────────────────

  _MatchResult? _findMatch(List<double> probe) {
    if (_registeredFaces.isEmpty) return null;

    var bestDistance = double.infinity;
    var runnerUp = double.infinity;
    RegisteredFace? best;

    for (final person in _registeredFaces) {
      // Best-of-N over the individual shots, falling back to the mean template
      // for rows that predate per-sample storage.
      final candidates = person.samples.isNotEmpty
          ? person.samples
          : [person.embedding];

      var personBest = double.infinity;
      for (final candidate in candidates) {
        if (candidate.isEmpty) continue;
        final d = _euclidean(probe, candidate);
        if (d < personBest) personBest = d;
      }

      if (personBest < bestDistance) {
        runnerUp = bestDistance;
        bestDistance = personBest;
        best = person;
      } else if (personBest < runnerUp) {
        runnerUp = personBest;
      }
    }

    if (best == null) return null;

    final margin = runnerUp.isFinite ? runnerUp - bestDistance : double.infinity;

    debugPrint('[FaceRecognition] closest=${best.name} '
        'd=${bestDistance.toStringAsFixed(3)} '
        'margin=${margin.isFinite ? margin.toStringAsFixed(3) : "n/a"} '
        'threshold=$_matchThreshold');

    if (bestDistance > _matchThreshold) return null;
    if (margin < _minMargin) {
      debugPrint('[FaceRecognition] two people too close to call — '
          'staying silent');
      return null;
    }

    return _MatchResult(name: best.name, distance: bestDistance);
  }

  double _euclidean(List<double> a, List<double> b) {
    var sum = 0.0;
    final len = min(a.length, b.length);
    for (var i = 0; i < len; i++) {
      final d = a[i] - b[i];
      sum += d * d;
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

  // ── Camera helpers ────────────────────────────────────────────────────────

  InputImageRotation get _rotation {
    switch (_sensorOrientation) {
      case 0:
        return InputImageRotation.rotation0deg;
      case 180:
        return InputImageRotation.rotation180deg;
      case 270:
        return InputImageRotation.rotation270deg;
      default:
        return InputImageRotation.rotation90deg;
    }
  }

  /// Build ML Kit's input frame.
  ///
  /// THIS IS WHY NO FACE WAS EVER DETECTED IN THE LIVE CAMERA.
  /// The previous version concatenated the camera's three YUV_420_888 planes
  /// (Y, then all of U, then all of V) and declared the result NV21. NV21 is a
  /// completely different layout: the Y plane followed by V and U *interleaved*
  /// two bytes at a time. ML Kit therefore read the chroma as noise and, worse,
  /// the buffer never matched the stride it was told to expect -- so the live
  /// detector found nothing, every frame, while registration worked fine
  /// because that path hands ML Kit a real JPEG file instead.
  InputImage? _buildInputImage(CameraImage cameraImage, Uint8List? nv21) {
    try {
      if (nv21 == null) return null;

      return InputImage.fromBytes(
        bytes: nv21,
        metadata: InputImageMetadata(
          size: Size(
            cameraImage.width.toDouble(),
            cameraImage.height.toDouble(),
          ),
          rotation: _rotation,
          format: InputImageFormat.nv21,
          // We repacked Y at exactly `width` bytes per row, so the stride is
          // width -- not the camera's original padded bytesPerRow.
          bytesPerRow: cameraImage.width,
        ),
      );
    } catch (e) {
      debugPrint('[FaceRecognition] Could not build ML Kit frame: $e');
      return null;
    }
  }

  /// Convert a camera frame to true NV21: full Y plane, then V,U interleaved.
  /// Row and pixel strides are honoured -- Android pads rows on many devices,
  /// and ignoring that shears the image diagonally.
  Uint8List? _toNv21(CameraImage image) {
    // Some devices/configs already hand back NV21 in a single plane.
    if (image.planes.length == 1) return image.planes[0].bytes;
    if (image.planes.length < 3) return null;

    final width = image.width;
    final height = image.height;
    final yPlane = image.planes[0];
    final uPlane = image.planes[1];
    final vPlane = image.planes[2];

    final chromaWidth = (width + 1) ~/ 2;
    final chromaHeight = (height + 1) ~/ 2;
    final out = Uint8List(width * height + 2 * chromaWidth * chromaHeight);

    var offset = 0;
    for (var row = 0; row < height; row++) {
      final start = row * yPlane.bytesPerRow;
      out.setRange(offset, offset + width, yPlane.bytes, start);
      offset += width;
    }

    final uvRowStride = uPlane.bytesPerRow;
    final uvPixelStride = uPlane.bytesPerPixel ?? 1;
    for (var row = 0; row < chromaHeight; row++) {
      for (var col = 0; col < chromaWidth; col++) {
        final i = row * uvRowStride + col * uvPixelStride;
        if (i >= vPlane.bytes.length || i >= uPlane.bytes.length) break;
        out[offset++] = vPlane.bytes[i]; // V first -- that is what the 21 means
        out[offset++] = uPlane.bytes[i];
      }
    }

    return out;
  }

  img.Image? _convertCameraImage(CameraImage cameraImage, Uint8List? nv21) {
    try {
      img.Image? image;
      if (cameraImage.format.group == ImageFormatGroup.yuv420) {
        // Built from the SAME NV21 buffer that was handed to ML Kit, so the
        // pixels being cropped are provably the pixels ML Kit measured.
        image = nv21 == null
            ? null
            : _nv21ToImage(nv21, cameraImage.width, cameraImage.height);
      } else if (cameraImage.format.group == ImageFormatGroup.bgra8888) {
        image = img.Image.fromBytes(
          width: cameraImage.width,
          height: cameraImage.height,
          bytes: cameraImage.planes[0].bytes.buffer,
          format: img.Format.uint8,
          numChannels: 4,
        );
      }
      // Deliberately NOT rotated here -- the caller applies the rotation that
      // calibration selected.
      return image;
    } catch (e) {
      return null;
    }
  }

  /// NV21 → RGB.
  ///
  /// THIS REPLACES THE CONVERTER THAT WAS CORRUPTING EVERY FACE CROP.
  /// The old `_convertYUV420` read luma as `yPlane[y * width + x]`, ignoring
  /// the plane's `bytesPerRow`. Android pads that row stride on most devices,
  /// and when stride != width every row is read a few bytes late — the image
  /// shears diagonally into smeared nonsense. ML Kit never saw that, because it
  /// was given the properly-strided NV21 buffer, so it kept detecting faces
  /// perfectly while the crop being embedded was garbage. That is exactly why
  /// the distance sat at ~1.21 no matter which rotation was used: rotating
  /// nonsense gives you rotated nonsense.
  ///
  /// Deriving the RGB frame from the same NV21 bytes makes the two paths agree
  /// by construction.
  img.Image _nv21ToImage(Uint8List nv21, int width, int height) {
    final out = img.Image(width: width, height: height);
    final frameSize = width * height;

    for (var j = 0, yp = 0; j < height; j++) {
      var uvp = frameSize + (j >> 1) * width;
      var u = 0, v = 0;
      for (var i = 0; i < width; i++, yp++) {
        var y = (nv21[yp] & 0xff) - 16;
        if (y < 0) y = 0;
        if ((i & 1) == 0 && uvp + 1 < nv21.length) {
          v = (nv21[uvp++] & 0xff) - 128;
          u = (nv21[uvp++] & 0xff) - 128;
        }
        final y1192 = 1192 * y;
        final r = (y1192 + 1634 * v).clamp(0, 262143) >> 10;
        final g = (y1192 - 833 * v - 400 * u).clamp(0, 262143) >> 10;
        final b = (y1192 + 2066 * u).clamp(0, 262143) >> 10;
        out.setPixelRgb(i, j, r, g, b);
      }
    }
    return out;
  }

  void dispose() {
    _faceDetector?.close();
    _faceDetector = null;
    _interpreter?.close();
    _interpreter = null;
    _isInitialized = false;
  }
}

class _MatchResult {
  final String name;
  final double distance;
  const _MatchResult({required this.name, required this.distance});
}
