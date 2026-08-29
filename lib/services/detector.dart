import 'dart:async';
import 'dart:isolate';

import 'package:camera/camera.dart';
import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart' hide Image;
import 'package:flutter/services.dart';
import 'package:image/image.dart';
import 'package:tensorflow_demo/models/detected_object/detected_object_dm.dart';
import 'package:tensorflow_demo/services/tensorflow_service.dart';
import 'package:tensorflow_demo/utils/image_utils.dart';
import 'package:tensorflow_demo/utils/tensorflow_helper.dart';
import 'package:tensorflow_demo/values/enumerations.dart';
import 'package:tflite_flutter/tflite_flutter.dart';

/// Debug logging flag - set to false in production to avoid console spam
const bool _enableDebugLogs = false;

/// A command sent between [Detector] and [_DetectorServer].
class _Command {
  const _Command(this.processType, {this.args});

  final TensorflowProcessType processType;
  final List<Object>? args;
}

/// Snapshot of live detection performance, published on [Detector.perf].
class PerfStats {
  const PerfStats({
    required this.fps,
    required this.lastInferenceMs,
    required this.avgInferenceMs,
    required this.detections,
    this.stageMs = const <double>[0, 0, 0, 0, 0, 0],
  });

  /// Frames processed per second (rolling average over recent frames).
  final double fps;

  /// End-to-end processing time of the most recent frame, in milliseconds
  /// (camera-image conversion + rotation + inference + post-processing).
  final double lastInferenceMs;

  /// Rolling-average processing time over recent frames, in milliseconds.
  final double avgInferenceMs;

  /// Number of objects detected in the most recent frame.
  final int detections;

  /// Per-stage milliseconds for the most recent frame:
  /// [total, convert, rotate, letterbox, inference, postprocess].
  final List<double> stageMs;

  /// Milliseconds spent turning a camera frame into a model input tensor --
  /// conversion + rotation + letterboxing. If this rivals or exceeds
  /// [modelMs], the bottleneck is preprocessing, not the model.
  double get preprocessMs =>
      stageMs.length > 3 ? stageMs[1] + stageMs[2] + stageMs[3] : 0;

  /// Milliseconds actually spent inside the interpreters.
  double get modelMs => stageMs.length > 4 ? stageMs[4] : 0;

  static const zero = PerfStats(
      fps: 0,
      lastInferenceMs: 0,
      avgInferenceMs: 0,
      detections: 0,
      stageMs: <double>[0, 0, 0, 0, 0, 0]);
}

/// Runs object detection using two YOLOv8 models in a background isolate:
///   1. COCO model     — 80 everyday classes (person, car, chair, etc.)
///   2. Accessibility  — 3 classes (Door, Stair, Window)
/// Results from both models are merged before being emitted on [resultsStream].
class Detector {
  Detector._(
    this._isolate,
    this._cocoInterpreter,
    this._cocoLabels,
    this._accInterpreter,
    this._accLabels,
    this._sensorOrientation,
  );

  final Isolate _isolate;
  late final Interpreter _cocoInterpreter;
  late final List<String> _cocoLabels;
  late final Interpreter _accInterpreter;
  late final List<String> _accLabels;
  final int _sensorOrientation;

  late final SendPort _sendPort;
  bool _isReady = false;

  // ── Performance telemetry ─────────────────────────────────────────────────
  /// Live FPS / latency, updated on every result. Bind a ValueListenableBuilder
  /// to this to show an on-screen overlay.
  final ValueNotifier<PerfStats> perf = ValueNotifier<PerfStats>(PerfStats.zero);
  final List<double> _recentInferenceMs = [];
  final List<double> _recentIntervalsMs = [];
  DateTime? _lastResultTime;
  bool _perfCsvHeaderPrinted = false;

  // ── Frame throttle ────────────────────────────────────────────────────────
  DateTime _lastFrameTime = DateTime.fromMillisecondsSinceEpoch(0);
  // 80 ms hard-capped throughput at 12.5 FPS no matter how fast the models
  // got. 50 ms allows up to 20 FPS while still dropping frames the isolate
  // cannot keep up with (the busy/ready handshake does the real backpressure).
  static const Duration _minFrameInterval = Duration(milliseconds: 50);

  // ── Stability filter ─────────────────────────────────────────────────────
  List<DetectedObjectDm> _previousResults = [];

  final StreamController<List<DetectedObjectDm>> _resultsStreamController =
      StreamController<List<DetectedObjectDm>>();

  Stream<List<DetectedObjectDm>> get resultsStream =>
      _resultsStreamController.stream;

  /// Launch the server on a background isolate.
  static Future<Detector> start({int sensorOrientation = 90}) async {
    final ReceivePort receivePort = ReceivePort();
    final Isolate isolate =
        await Isolate.spawn(_DetectorServer._run, receivePort.sendPort);

    final Detector result = Detector._(
      isolate,
      TensorflowService.ssdMobileNet.interpreter,
      TensorflowService.ssdMobileNet.labels,
      TensorflowService.accessibilityModel.interpreter,
      TensorflowService.accessibilityModel.labels,
      sensorOrientation,
    );
    receivePort.listen((message) {
      result._handleCommand(message as _Command);
    });
    if (_enableDebugLogs) debugPrint('[Detector] start() complete, waiting for handshake...');
    return result;
  }

  /// Starts CameraImage processing
  void processFrame(CameraImage cameraImage) {
    if (_isReady) {
      final now = DateTime.now();
      if (now.difference(_lastFrameTime) < _minFrameInterval) return;
      _lastFrameTime = now;
      _sendPort.send(
        _Command(TensorflowProcessType.detect, args: [cameraImage]),
      );
    }
  }

  void _handleCommand(_Command command) {
    switch (command.processType) {
      case TensorflowProcessType.init:
        _sendPort = command.args?[0] as SendPort;
        RootIsolateToken rootIsolateToken = RootIsolateToken.instance!;
        _sendPort.send(_Command(TensorflowProcessType.init, args: [
          rootIsolateToken,
          _cocoInterpreter.address,
          _cocoLabels,
          _accInterpreter.address,
          _accLabels,
          _sensorOrientation,
        ]));
      case TensorflowProcessType.ready:
        _isReady = true;
        if (_enableDebugLogs) debugPrint('[Detector] Handshake complete, _isReady = true');
      case TensorflowProcessType.busy:
        _isReady = false;
      case TensorflowProcessType.result:
        _isReady = true;
        if (!_resultsStreamController.isClosed) {
          final rawResults = command.args?[0] as List<DetectedObjectDm>;
          final stabilized = _stabilize(rawResults);
          _previousResults = stabilized;
          if (_enableDebugLogs) debugPrint('[Detector] Got ${stabilized.length} detections');
          _resultsStreamController.add(stabilized);
          final inferenceMs = (command.args != null && command.args!.length > 1)
              ? (command.args![1] as num).toDouble()
              : 0.0;
          final stages = (command.args != null && command.args!.length > 2)
              ? (command.args![2] as List).cast<double>()
              : const <double>[0, 0, 0, 0, 0, 0];
          _updatePerf(inferenceMs, stabilized.length, stages);
        }
      default:
        if (_enableDebugLogs) debugPrint('Detector unrecognized command: ${command.processType}');
    }
  }

  /// Updates the rolling FPS / latency stats and emits a CSV telemetry line.
  ///
  /// To capture a session for offline graphing, run:
  ///   flutter run | tee perf_log.txt
  /// then feed perf_log.txt to metrics/runtime_report.py.
  void _updatePerf(double inferenceMs, int detections,
      [List<double> stageUs = const <double>[0, 0, 0, 0, 0, 0]]) {
    final now = DateTime.now();

    if (_lastResultTime != null) {
      final dtMs = now.difference(_lastResultTime!).inMicroseconds / 1000.0;
      if (dtMs > 0) {
        _recentIntervalsMs.add(dtMs);
        if (_recentIntervalsMs.length > 30) _recentIntervalsMs.removeAt(0);
      }
    }
    _lastResultTime = now;

    _recentInferenceMs.add(inferenceMs);
    if (_recentInferenceMs.length > 30) _recentInferenceMs.removeAt(0);

    final avgMs = _recentInferenceMs.isEmpty
        ? 0.0
        : _recentInferenceMs.reduce((a, b) => a + b) / _recentInferenceMs.length;
    final avgInterval = _recentIntervalsMs.isEmpty
        ? 0.0
        : _recentIntervalsMs.reduce((a, b) => a + b) / _recentIntervalsMs.length;
    final fps = avgInterval > 0 ? 1000.0 / avgInterval : 0.0;

    perf.value = PerfStats(
      fps: fps,
      lastInferenceMs: inferenceMs,
      avgInferenceMs: avgMs,
      detections: detections,
      stageMs: List<double>.unmodifiable(
          stageUs.map((us) => us / 1000.0).toList(growable: false)),
    );

    // Use print (not debugPrint) for telemetry: debugPrint rate-limits output
    // to ~1 KB/s and would DROP rows once the frame rate climbs in a
    // release/profile build, corrupting the runtime graphs.
    if (!_perfCsvHeaderPrinted) {
      // ignore: avoid_print
      _perfLog('PERF_CSV,timestamp_ms,inference_ms,fps,detections,'
          'convert_ms,rotate_ms,letterbox_ms,infer_ms,parse_ms');
      _perfCsvHeaderPrinted = true;
    }
    String ms(int i) =>
        (i < stageUs.length ? stageUs[i] / 1000.0 : 0.0).toStringAsFixed(2);
    // ignore: avoid_print
    _perfLog('PERF_CSV,${now.millisecondsSinceEpoch},'
        '${inferenceMs.toStringAsFixed(1)},${fps.toStringAsFixed(2)},$detections,'
        '${ms(1)},${ms(2)},${ms(3)},${ms(4)},${ms(5)}');
  }

  /// Classes where a one-frame delay is not acceptable. Mirrors
  /// VoiceService._hazardClasses; keep the two in sync.
  static const Set<String> _hazardClasses = {
    'person', 'car', 'truck', 'bus', 'motorcycle', 'bicycle',
    'dog', 'fire hydrant', 'stop sign', 'traffic light', 'stair',
  };

  /// Suppresses one-frame flicker by requiring a label to persist across two
  /// consecutive frames — EXCEPT for hazard classes, which are always passed
  /// through immediately.
  ///
  /// The old version filtered every class, so a car that appeared suddenly was
  /// discarded on its first frame: ~100 ms of extra latency on exactly the
  /// events that matter most.
  List<DetectedObjectDm> _stabilize(List<DetectedObjectDm> current) {
    if (_previousResults.isEmpty) return current;
    final prevLabels = _previousResults.map((d) => d.label).toSet();
    final stable = current
        .where((d) =>
            _hazardClasses.contains(d.label.toLowerCase()) ||
            prevLabels.contains(d.label))
        .toList();
    return stable.isEmpty ? current : stable;
  }

  /// Kills the background isolate and its detector server.
  ///
  /// `perf` is deliberately NOT disposed here: the screen nulls its `_detector`
  /// reference without an immediate setState, so a ValueListenableBuilder can
  /// still be mounted against this notifier for one more frame. Disposing it
  /// here risked a "used after dispose" crash on app resume. The notifier is a
  /// few bytes and dies with the Detector instance.
  void stop() {
    if (!_resultsStreamController.isClosed) _resultsStreamController.close();
    _isolate.kill(priority: Isolate.immediate);
  }
}

/// The portion of the [Detector] that runs on the background isolate.
class _DetectorServer {
  _DetectorServer(this._sendPort);

  Interpreter? _cocoInterpreter;
  List<String>? _cocoLabels;
  ModelGeometry? _cocoGeometry;
  Interpreter? _accInterpreter;
  List<String>? _accLabels;
  ModelGeometry? _accGeometry;
  int _sensorOrientation = 90;
  final SendPort _sendPort;

  // ── Model scheduling ──────────────────────────────────────────────────────
  // Running BOTH models on every frame meant two full 640×640 inferences per
  // frame — the original lag. Instead we run the COCO model EVERY frame (it
  // drives the responsive everyday classes: person, chair, car) and the lighter
  // accessibility model only every Nth frame (doors/stairs are static enough to
  // tolerate a lower refresh). Each emit merges the freshest result of each.
  //
  // UPDATE: with the accessibility model at float16 this drops from 6 to 2, and
  // once it is retrained as YOLO11n (metrics/train_accessibility.py) set it to
  // 1 so stairs — the most safety-critical class — refresh every frame.
  // For a blind pedestrian, refresh rate on Stair is worth more than the 2-4
  // mAP points the smaller backbone costs.
  static const int _accEveryNFrames = 2;
  int _frameCount = 0;
  List<DetectedObjectDm> _lastCoco = const [];
  List<DetectedObjectDm> _lastAcc = const [];

  // v2 accessibility model has exactly 3 trained classes (Door, Stair, Window),
  // so there are no untrained phantom channels left to filter. Kept as an empty
  // set so the merge code below stays unchanged.
  static const Set<String> _untrainedAccClasses = {};

  static void _run(SendPort sendPort) {
    ReceivePort receivePort = ReceivePort();
    final _DetectorServer server = _DetectorServer(sendPort);
    receivePort.listen((message) async {
      final _Command command = message as _Command;
      await server._handleCommand(command);
    });
    sendPort.send(
        _Command(TensorflowProcessType.init, args: [receivePort.sendPort]));
  }

  Future<void> _handleCommand(_Command command) async {
    switch (command.processType) {
      case TensorflowProcessType.init:
        RootIsolateToken rootIsolateToken =
            command.args?[0] as RootIsolateToken;
        BackgroundIsolateBinaryMessenger.ensureInitialized(rootIsolateToken);
        _cocoInterpreter = Interpreter.fromAddress(command.args?[1] as int);
        _cocoLabels     = command.args?[2] as List<String>;
        _accInterpreter = Interpreter.fromAddress(command.args?[3] as int);
        _accLabels      = command.args?[4] as List<String>;
        // Read input size / class count / anchor count from the models
        // themselves so a re-export at another imgsz needs no code change.
        _cocoGeometry = ModelGeometry.fromInterpreter(_cocoInterpreter!);
        _accGeometry  = ModelGeometry.fromInterpreter(_accInterpreter!);
        _sensorOrientation = command.args?[5] as int? ?? 90;
        _sendPort.send(const _Command(TensorflowProcessType.ready));
      case TensorflowProcessType.detect:
        _sendPort.send(const _Command(TensorflowProcessType.busy));
        _convertCameraImage(command.args?[0] as CameraImage);
      default:
        debugPrint('_DetectorService unrecognized command ${command.processType}');
    }
  }

  /// Per-stage timings for the frame currently being processed, in
  /// microseconds. Sent to the UI isolate as a plain List<double> (always
  /// transferable) rather than a custom class.
  final List<double> _stageUs = List<double>.filled(6, 0);

  void _convertCameraImage(CameraImage cameraImage) {
    final swTotal = Stopwatch()..start();
    try {
      // ── Stage 1: YUV/NV21/BGRA -> RGB ─────────────────────────────────
      // Pure Dart, one pass over every sensor pixel. Prime suspect for the
      // frame budget: a published benchmark puts the pure-Dart `image`
      // package at 82.5 ms for convert+rotate+resize where native OpenCV
      // does the same work in 5.4 ms.
      final swConvert = Stopwatch()..start();
      var image = ImageUtils.convertCameraImageToImage(cameraImage);
      swConvert.stop();

      if (image == null) {
        _sendPort.send(_Command(TensorflowProcessType.result,
            args: [<DetectedObjectDm>[], 0.0, _stageUs]));
        return;
      }

      // ── Stage 2: rotation to display orientation ──────────────────────
      // Allocates and copies a whole second full-resolution image. Becomes
      // free once TensorflowHelper.useFusedRotationLetterbox is enabled,
      // which folds this into the letterbox sampling loop.
      final swRotate = Stopwatch()..start();
      if (_sensorOrientation != 0 &&
          !TensorflowHelper.useFusedRotationLetterbox) {
        image = copyRotate(image, angle: _sensorOrientation);
      }
      swRotate.stop();

      // ── Stages 3-5: letterbox, inference, post-process (per model) ────
      _letterboxUs = 0;
      _inferUs = 0;
      _parseUs = 0;
      final results = _analyseImageCamera(image);

      swTotal.stop();

      _stageUs[0] = swTotal.elapsedMicroseconds.toDouble();
      _stageUs[1] = swConvert.elapsedMicroseconds.toDouble();
      _stageUs[2] = swRotate.elapsedMicroseconds.toDouble();
      _stageUs[3] = _letterboxUs.toDouble();
      _stageUs[4] = _inferUs.toDouble();
      _stageUs[5] = _parseUs.toDouble();

      _sendPort.send(_Command(TensorflowProcessType.result, args: [
        results,
        swTotal.elapsedMicroseconds / 1000.0,
        List<double>.from(_stageUs),
      ]));
    } catch (e, s) {
      if (_enableDebugLogs) debugPrint('[DetectorServer] ERROR: $e\n$s');
      _sendPort.send(_Command(TensorflowProcessType.result,
          args: [<DetectedObjectDm>[], 0.0, _stageUs]));
    }
  }

  // Accumulators across the (up to two) models run on one frame.
  int _letterboxUs = 0;
  int _inferUs = 0;
  int _parseUs = 0;

  void _collectStageTimings() {
    _letterboxUs += TensorflowHelper.lastLetterboxUs;
    _inferUs += TensorflowHelper.lastInferUs;
    _parseUs += TensorflowHelper.lastParseUs;
  }

  /// Runs the COCO model every frame and the accessibility model every Nth
  /// frame, merging the freshest result of each. COCO at full rate keeps people
  /// / chairs / cars responsive; doors and stairs refresh a little slower.
  List<DetectedObjectDm> _analyseImageCamera(Image image) {
    // ── COCO model (nc=80) — every frame ──────────────────────────────────
    if (_cocoInterpreter != null) {
      _lastCoco = TensorflowHelper.analyseImage(
        image,
        interpreter: _cocoInterpreter!,
        label: _cocoLabels ?? [],
        geometry: _cocoGeometry!,
        confidenceThreshold: 0.40,
        iouThreshold: 0.50,
        drawObjectOnImage: false,
        returnDetectedImage: false,
      ).detectedObjects;
      _collectStageTimings();
    }

    // ── Accessibility model (nc=3: Door, Stair, Window) — every Nth frame ──
    if (_accInterpreter != null && (_frameCount++ % _accEveryNFrames) == 0) {
      final acc = TensorflowHelper.analyseImage(
        image,
        interpreter: _accInterpreter!,
        label: _accLabels ?? [],
        geometry: _accGeometry!,
        confidenceThreshold: 0.45,
        iouThreshold: 0.50,
        drawObjectOnImage: false,
        returnDetectedImage: false,
      ).detectedObjects;
      _collectStageTimings();
      // v2 has no untrained channels, so this filter is a pass-through now.
      _lastAcc = acc
          .where((d) => !_untrainedAccClasses.contains(d.label))
          .toList(growable: false);
    }

    return [..._lastCoco, ..._lastAcc];
  }
}

/// Frame telemetry, emitted in debug and profile builds but never in release.
///
/// `kReleaseMode` rather than `kDebugMode` on purpose: the performance captures
/// this project relies on are taken with `flutter run --profile`, where
/// kDebugMode is false. Gating on kDebugMode would have silently broken the
/// measurement workflow.
void _perfLog(String line) {
  if (!kReleaseMode) print(line);
}
