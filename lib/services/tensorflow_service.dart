import 'dart:developer';

import 'package:flutter/foundation.dart';
import 'package:flutter/services.dart';
import 'package:image/image.dart';
import 'package:tensorflow_demo/models/detected_object/detected_object_dm.dart';
import 'package:tensorflow_demo/utils/tensorflow_helper.dart';
import 'package:tensorflow_demo/values/app_constants.dart';
import 'package:tensorflow_demo/values/typedefs.dart';
import 'package:tflite_flutter/tflite_flutter.dart';

/// Debug logging flag - set to false in production to avoid console spam
const bool _enableDebugLogs = false;

class TensorflowService {
  const TensorflowService._({required this.modelPath, required this.labelPath});

  // COCO model — 80 classes (person, car, chair, etc.)
  static const ssdMobileNet = TensorflowService._(
    modelPath: AppConstants.cocoModelPath,
    labelPath: AppConstants.cocoLabelPath,
  );

  // Accessibility model — Door, Stair, Window
  static const accessibilityModel = TensorflowService._(
    modelPath: AppConstants.accessibilityModelPath,
    labelPath: AppConstants.accessibilityLabelPath,
  );

  final String modelPath;
  final String labelPath;

  // ── Per-instance state keyed by modelPath ─────────────────────────────────
  static final Map<String, Interpreter> _interpreters = {};
  static final Map<String, List<String>> _labelsMap = {};
  static final Map<String, ModelGeometry> _geometries = {};
  static final Map<String, bool> _initializedMap = {};
  static final Map<String, Future<void>?> _initFutures = {};

  Interpreter get interpreter => _interpreters[modelPath]!;
  List<String> get labels => _labelsMap[modelPath] ?? [];

  /// Input size / class count / anchor count, read from the model's own
  /// tensors. Re-exporting at a different `imgsz` needs no code change.
  ModelGeometry get geometry => _geometries[modelPath]!;

  Future<void> initialize() async {
    if (_initializedMap[modelPath] == true) return;
    if (_initFutures[modelPath] != null) return _initFutures[modelPath]!;

    _initFutures[modelPath] = Future.wait([
      _loadModel(),
      _loadLabels(),
    ]).then((_) {
      _initializedMap[modelPath] = true;
    }).whenComplete(() {
      _initFutures[modelPath] = null;
    });

    return _initFutures[modelPath]!;
  }

  Future<void> ensureInitialized() async {
    await initialize();
  }

  /// Builds interpreter options, preferring hardware acceleration.
  ///
  /// Previously only iOS got a delegate; Android ran on 4 CPU threads with no
  /// acceleration at all. GPU delegation is typically 2-5x on mid-range Android
  /// hardware, which is the single largest speed lever left in the app.
  ///
  /// Not every TFLite op has a GPU kernel, so a failed delegate must NOT be
  /// fatal — we fall back to multi-threaded CPU, which is what shipped before.
  InterpreterOptions _buildOptions({required bool tryDelegate}) {
    final options = InterpreterOptions()..threads = 4;
    if (!tryDelegate) return options;

    try {
      if (defaultTargetPlatform == TargetPlatform.iOS) {
        options.addDelegate(GpuDelegate());
      } else if (defaultTargetPlatform == TargetPlatform.android) {
        options.addDelegate(GpuDelegateV2());
      }
    } catch (e) {
      log('GPU delegate unavailable, using CPU: $e', name: 'TensorflowService');
    }
    return options;
  }

  Future<void> _loadModel() async {
    Interpreter? interpreter;

    // Attempt 1: hardware-accelerated. Attempt 2: plain CPU.
    for (final tryDelegate in [true, false]) {
      try {
        interpreter = await Interpreter.fromAsset(
          modelPath,
          options: _buildOptions(tryDelegate: tryDelegate),
        );
        if (tryDelegate) {
          log('loaded with GPU delegate', name: 'TensorflowService[$modelPath]');
        } else {
          log('loaded on CPU (4 threads)',
              name: 'TensorflowService[$modelPath]');
        }
        break;
      } catch (e) {
        if (!tryDelegate) {
          log('MODEL LOAD FAILED: $e', name: 'TensorflowService[$modelPath]');
          rethrow;
        }
        log('GPU path failed ($e); retrying on CPU',
            name: 'TensorflowService[$modelPath]');
      }
    }

    final loaded = interpreter!;
    loaded.allocateTensors();

    final geometry = ModelGeometry.fromInterpreter(loaded);
    log('$geometry', name: 'TensorflowService[$modelPath]');

    if (_enableDebugLogs) {
      log('in:  ${loaded.getInputTensors().map((e) => e.shape).toList()}',
          name: 'TensorflowService[$modelPath]');
      log('out: ${loaded.getOutputTensors().map((e) => e.shape).toList()}',
          name: 'TensorflowService[$modelPath]');
    }

    _interpreters[modelPath] = loaded;
    _geometries[modelPath] = geometry;
  }

  Future<void> _loadLabels() async {
    final labelsRaw = await rootBundle.loadString(labelPath);
    _labelsMap[modelPath] = labelsRaw
        .split(RegExp(r'\r?\n'))
        .map((label) => label.trim())
        .where((label) => label.isNotEmpty)
        .toList();
  }

  /// Sanity check: the label file and the model head must agree, otherwise
  /// every detection is mislabelled and nothing visibly breaks. This has bitten
  /// this project before (the 5-class model shipped with a 3-class label file).
  String? get labelCountMismatch {
    final g = _geometries[modelPath];
    final l = _labelsMap[modelPath];
    if (g == null || l == null) return null;
    if (g.numClasses == l.length) return null;
    return 'Model head has ${g.numClasses} classes but $labelPath lists '
        '${l.length}. Labels will be wrong.';
  }

  AnalyseImageCallback analyseImage(Uint8List imageData) {
    final image = decodeImage(imageData);
    if (image == null) {
      return (imageBytes: null, detectedObjects: <DetectedObjectDm>[]);
    }
    return TensorflowHelper.analyseImage(
      image,
      interpreter: interpreter,
      label: labels,
      geometry: geometry,
    );
  }
}
