import 'dart:developer';

import 'package:flutter/foundation.dart';
import 'package:flutter/services.dart';
import 'package:image/image.dart';
import 'package:tensorflow_demo/utils/tensorflow_helper.dart';
import 'package:tensorflow_demo/values/app_constants.dart';
import 'package:tensorflow_demo/values/typedefs.dart';
import 'package:tflite_flutter/tflite_flutter.dart';
import 'package:tensorflow_demo/models/detected_object/detected_object_dm.dart';

/// Debug logging flag - set to false in production to avoid console spam
const bool _enableDebugLogs = false;

class TensorflowService {
  const TensorflowService._({required this.modelPath, required this.labelPath});

  static const ssdMobileNet = TensorflowService._(
    modelPath: AppConstants.ssdMobileNetV1,
    labelPath: AppConstants.ssdMobileNetV1LabelPath,
  );

  final String modelPath;
  final String labelPath;

  static late final Interpreter _interpreter;
  static late final List<String>? _labels;
  static bool _isInitialized = false;
  static Future<void>? _initializationFuture;

  Interpreter get interpreter => _interpreter;

  List<String> get labels => _labels ?? [];

  Future<void> initialize() async {
    if (_isInitialized) return;
    if (_initializationFuture != null) return _initializationFuture!;

    _initializationFuture = Future.wait([
      _loadModel(),
      _loadLabels(),
    ]).then((_) {
      _isInitialized = true;
    }).whenComplete(() {
      _initializationFuture = null;
    });

    return _initializationFuture!;
  }

  Future<void> ensureInitialized() async {
    await initialize();
  }

  Future<void> _loadModel() async {
  try {
    if (_enableDebugLogs) print('🔴 STARTING MODEL LOAD');

    // Use 4 threads on Android (Snapdragon/Mali multi-core), 2 elsewhere.
    // XNNPackDelegate with numThreads gives ~2–3× speedup over single-thread.
    final delegate = switch (defaultTargetPlatform) {
      TargetPlatform.iOS => GpuDelegate(),
      TargetPlatform.android => XNNPackDelegate(
        options: XNNPackDelegateOptions(numThreads: 4),
      ),
      _ => XNNPackDelegate(options: XNNPackDelegateOptions(numThreads: 2)),
    };

    if (_enableDebugLogs) print('🔴 LOADING INTERPRETER FROM: $modelPath');

    _interpreter = await Interpreter.fromAsset(
      modelPath,
      options: InterpreterOptions()
        ..threads = 4          // parallelise ops not handled by the delegate
        ..addDelegate(delegate),
    );

    if (_enableDebugLogs) print('🔴 INTERPRETER LOADED SUCCESSFULLY');

    final inputTensors = _interpreter.getInputTensors();
    log(
      'Value: ${inputTensors.map((e) => e.toString()).toList()}',
      name: 'inputTensors',
    );

    final outputTensors = _interpreter.getOutputTensors();
    log(
      'Value: ${outputTensors.map((e) => e.toString()).toList()}',
      name: 'outputTensors',
    );

    _interpreter.allocateTensors();
    
    if (_enableDebugLogs) print('🔴 MODEL LOAD COMPLETE');
  } catch (e, stackTrace) {
    if (_enableDebugLogs) print('🔴🔴🔴 MODEL LOAD FAILED: $e');
    if (_enableDebugLogs) print('🔴🔴🔴 STACK TRACE: $stackTrace');
    rethrow;
  }
}
  Future<void> _loadLabels() async {
    final labelsRaw = await rootBundle.loadString(labelPath);
    _labels = labelsRaw.split('\n');
  }

  AnalyseImageCallback analyseImage(Uint8List imageData) {
    final image = decodeImage(imageData);
    if (image == null) {
      return (
        imageBytes: null,
        detectedObjects: <DetectedObjectDm>[],
      );
    }
    return TensorflowHelper.analyseImage(
      image,
      interpreter: interpreter,
      label: _labels ?? [],
    );
  }
}
