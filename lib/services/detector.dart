import 'dart:async';
import 'dart:isolate';

import 'package:camera/camera.dart';
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

/// A Simple Detector that handles object detection via Service
///
/// All the heavy operations like pre-processing, detection, ets,
/// are executed in a background isolate.
/// This class just sends and receives messages to the isolate.
class Detector {
  Detector._(this._isolate, this._interpreter, this._labels, this._sensorOrientation);

  final Isolate _isolate;
  late final Interpreter _interpreter;
  late final List<String> _labels;
  final int _sensorOrientation;

  // To be used by detector (from UI) to send message to our Service ReceivePort
  late final SendPort _sendPort;

  bool _isReady = false;

  // ── Frame throttle ────────────────────────────────────────────────────────
  // Camera delivers ~30 fps; detection doesn't need more than ~12 fps.
  // Throttling reduces isolate message overhead and CPU heat without any
  // perceptible lag for the user.
  DateTime _lastFrameTime = DateTime.fromMillisecondsSinceEpoch(0);
  static const Duration _minFrameInterval = Duration(milliseconds: 80);

  // ── Stability filter ─────────────────────────────────────────────────────
  // A detection is only emitted if the same label was also seen in the
  // previous frame. Eliminates single-frame phantom detections (false positives
  // that flicker in for one frame). Latency cost: just one frame (~80 ms).
  List<DetectedObjectDm> _previousResults = [];

  final StreamController<List<DetectedObjectDm>> _resultsStreamController =
      StreamController<List<DetectedObjectDm>>();

  Stream<List<DetectedObjectDm>> get resultsStream =>
      _resultsStreamController.stream;

  /// Launch the server on a background isolate..
  static Future<Detector> start({int sensorOrientation = 90}) async {
    final ReceivePort receivePort = ReceivePort();
    // sendPort - To be used by service Isolate to send message to our ReceiverPort
    final Isolate isolate =
        await Isolate.spawn(_DetectorServer._run, receivePort.sendPort);

    final Detector result = Detector._(
      isolate,
      TensorflowService.ssdMobileNet.interpreter,
      TensorflowService.ssdMobileNet.labels,
      sensorOrientation,
    );
    receivePort.listen((message) {
      result._handleCommand(message as _Command);
    });
    if (_enableDebugLogs) debugPrint('[Detector] Detector.start() complete, waiting for handshake...');
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

  /// Handler invoked when a message is received from the port communicating
  /// with the database server.
  void _handleCommand(_Command command) {
    switch (command.processType) {
      case TensorflowProcessType.init:
        _sendPort = command.args?[0] as SendPort;
        // ----------------------------------------------------------------------
        // Before using platform channels and plugins from background isolates we
        // need to register it with its root isolate. This is achieved by
        // acquiring a [RootIsolateToken] which the background isolate uses to
        // invoke [BackgroundIsolateBinaryMessenger.ensureInitialized].
        // ----------------------------------------------------------------------
        RootIsolateToken rootIsolateToken = RootIsolateToken.instance!;
        _sendPort.send(_Command(TensorflowProcessType.init, args: [
          rootIsolateToken,
          _interpreter.address,
          _labels,
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
          final stableResults = _stabilize(rawResults);
          _previousResults = rawResults;
          if (_enableDebugLogs) {
            debugPrint('[Detector] raw=${rawResults.length} stable=${stableResults.length}');
          }
          _resultsStreamController.add(stableResults);
        }
      default:
        if (_enableDebugLogs) debugPrint('Detector unrecognized command: ${command.processType}');
    }
  }

  /// Keeps only detections whose label was also present in the previous frame.
  /// If ALL detections are brand-new (first appearance), passes them through
  /// so the screen doesn't blank out when a new object enters the frame.
  List<DetectedObjectDm> _stabilize(List<DetectedObjectDm> current) {
    if (_previousResults.isEmpty) return current;
    final prevLabels = _previousResults.map((d) => d.label).toSet();
    final stable = current.where((d) => prevLabels.contains(d.label)).toList();
    // Fall back to raw list so the first frame of a new object isn't skipped
    return stable.isEmpty ? current : stable;
  }

  /// Kills the background isolate and its detector server.
  void stop() {
    _resultsStreamController.close();
    _isolate.kill();
  }
}

/// The portion of the [Detector] that runs on the background isolate.
///
/// This is where we use the new feature Background Isolate Channels, which
/// allows us to use plugins from background isolates.
class _DetectorServer {
  _DetectorServer(this._sendPort);

  Interpreter? _interpreter;
  List<String>? _labels;
  int _sensorOrientation = 90;
  final SendPort _sendPort;

  // ----------------------------------------------------------------------
  // Here the plugin is used from the background isolate.
  // ----------------------------------------------------------------------

  /// The main entrypoint for the background isolate sent to [Isolate.spawn].
  static void _run(SendPort sendPort) {
    ReceivePort receivePort = ReceivePort();
    final _DetectorServer server = _DetectorServer(sendPort);
    receivePort.listen((message) async {
      final _Command command = message as _Command;
      await server._handleCommand(command);
    });
    // receivePort.sendPort - used by UI isolate to send commands to the service receiverPort
    sendPort.send(
        _Command(TensorflowProcessType.init, args: [receivePort.sendPort]));
  }

  /// Handle the [command] received from the [ReceivePort].
  Future<void> _handleCommand(_Command command) async {
    switch (command.processType) {
      case TensorflowProcessType.init:
        // ----------------------------------------------------------------------
        // The [RootIsolateToken] is required for
        // [BackgroundIsolateBinaryMessenger.ensureInitialized] and must be
        // obtained on the root isolate and passed into the background isolate via
        // a [SendPort].
        // ----------------------------------------------------------------------
        RootIsolateToken rootIsolateToken =
            command.args?[0] as RootIsolateToken;
        // ----------------------------------------------------------------------
        // [BackgroundIsolateBinaryMessenger.ensureInitialized] for each
        // background isolate that will use plugins. This sets up the
        // [BinaryMessenger] that the Platform Channels will communicate with on
        // the background isolate.
        // ----------------------------------------------------------------------
        BackgroundIsolateBinaryMessenger.ensureInitialized(rootIsolateToken);
        _interpreter = Interpreter.fromAddress(command.args?[1] as int);
        _labels = command.args?[2] as List<String>;
        _sensorOrientation = command.args?[3] as int? ?? 90;
        _sendPort.send(const _Command(TensorflowProcessType.ready));
      case TensorflowProcessType.detect:
        _sendPort.send(const _Command(TensorflowProcessType.busy));
        _convertCameraImage(command.args?[0] as CameraImage);
      default:
        debugPrint(
            '_DetectorService unrecognized command ${command.processType}');
    }
  }

  // Convert Camera Image to Image and run detection.
  void _convertCameraImage(CameraImage cameraImage) {
    try {
      // Convert YUV/NV21/BGRA to RGB image
      var image = ImageUtils.convertCameraImageToImage(cameraImage);
      if (image == null) {
        _sendPort.send(_Command(TensorflowProcessType.result,
            args: [<DetectedObjectDm>[]]));
        return;
      }

      if (_enableDebugLogs) debugPrint('[DetectorServer] Image converted: ${image.width}x${image.height}');

      // Rotate to match display orientation.
      // Android camera sensors produce landscape frames; sensorOrientation
      // tells us how many degrees CW to rotate for correct portrait display.
      if (_sensorOrientation != 0) {
        image = copyRotate(image, angle: _sensorOrientation);
        if (_enableDebugLogs) debugPrint('[DetectorServer] Rotated $_sensorOrientationÂ°: ${image.width}x${image.height}');
      }

      final results = _analyseImageCamera(image);

      _sendPort.send(_Command(TensorflowProcessType.result, args: [results]));
    } catch (e, s) {
      if (_enableDebugLogs) debugPrint('[DetectorServer] ERROR in _convertCameraImage: $e');
      if (_enableDebugLogs) debugPrint('[DetectorServer] Stacktrace: $s');
      // Always send result to restore _isReady
      _sendPort.send(_Command(TensorflowProcessType.result,
          args: [<DetectedObjectDm>[]]));
    }
  }

  /// Run YOLOv8 inference on a camera frame.
  ///
  /// Uses stretch resize (no letterbox) so output bbox coords are
  /// in 640x640 space which maps directly to the camera preview
  /// via [DetectedObjectDm.renderLocation].
  List<DetectedObjectDm> _analyseImageCamera(Image image) {
    if (_interpreter == null) {
      if (_enableDebugLogs) debugPrint('[DetectorServer] _interpreter is null!');
      return [];
    }

    final result = TensorflowHelper.analyseImage(
      image,
      interpreter: _interpreter!,
      label: _labels ?? [],
      drawObjectOnImage: false,
      returnDetectedImage: false,
      // letterbox=true (default) — preserves aspect ratio for better accuracy
    );

    return result.detectedObjects;
  }
}



