import 'dart:developer';

import 'package:flutter/foundation.dart';
import 'package:flutter/services.dart';
import 'package:image/image.dart';
import 'package:tensorflow_demo/utils/tensorflow_helper.dart';
import 'package:tensorflow_demo/values/app_constants.dart';
import 'package:tensorflow_demo/values/typedefs.dart';
import 'package:tflite_flutter/tflite_flutter.dart';
import 'package:tensorflow_demo/models/detected_object/detected_object_dm.dart';

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

  Interpreter get interpreter => _interpreter;

  List<String> get labels => _labels ?? [];

  Future<void> initialize() async {
    await Future.wait([
      _loadModel(),
      _loadLabels(),
    ]);
  }

  Future<void> _loadModel() async {
  try {
    print('🔴 STARTING MODEL LOAD');
    
    final delegate = switch (defaultTargetPlatform) {
      TargetPlatform.iOS => GpuDelegate(),
      _ => XNNPackDelegate(),
    };

    print('🔴 LOADING INTERPRETER FROM: $modelPath');
    
    _interpreter = await Interpreter.fromAsset(
      modelPath,
      options: InterpreterOptions()..addDelegate(delegate),
    );

    print('🔴 INTERPRETER LOADED SUCCESSFULLY');

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
    
    print('🔴 MODEL LOAD COMPLETE');
  } catch (e, stackTrace) {
    print('🔴🔴🔴 MODEL LOAD FAILED: $e');
    print('🔴🔴🔴 STACK TRACE: $stackTrace');
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
