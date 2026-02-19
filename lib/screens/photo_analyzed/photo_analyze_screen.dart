import 'dart:typed_data';

import 'package:flutter/material.dart';
import 'package:tensorflow_demo/models/detected_object/detected_object_dm.dart';
import 'package:tensorflow_demo/services/snackbar_service.dart';

import 'package:tensorflow_demo/services/tensorflow_service.dart';
import 'package:tensorflow_demo/screens/photo_analyzed/widgets/detected_object_tile.dart';

class PhotoAnalyzedScreen extends StatefulWidget {
  const PhotoAnalyzedScreen({required this.imageBytes, super.key});

  final Uint8List imageBytes;

  @override
  State<PhotoAnalyzedScreen> createState() => _PhotoAnalyzedScreenState();
}

class _PhotoAnalyzedScreenState extends State<PhotoAnalyzedScreen> {
  Uint8List? image;
  List<DetectedObjectDm> detectedObjectList = [];

  @override
  void initState() {
    WidgetsBinding.instance.addPostFrameCallback((_) {
      SnackBarService.show('Finding Objects...');
      _analyzeImage();
    });
    super.initState();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Details'),
      ),
      body: ListView(
        children: [
          AnimatedSwitcher(
            switchInCurve: Curves.easeInOutQuart,
            switchOutCurve: Curves.easeInOutQuart,
            duration: const Duration(milliseconds: 600),
            child: image == null
                ? Image.memory(
                    key: const ValueKey('old_image'),
                    widget.imageBytes,
                  )
                : Image.memory(
                    key: const ValueKey('new_image'),
                    image ?? widget.imageBytes,
                  ),
          ),
          AnimatedSwitcher(
            switchInCurve: Curves.easeInOutQuart,
            switchOutCurve: Curves.easeInOutQuart,
            duration: const Duration(milliseconds: 600),
            child: detectedObjectList.isEmpty
                ? const SizedBox.shrink()
                : Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      const Padding(
                        padding: EdgeInsets.all(8),
                        child: Text(
                          'Detected Objects:',
                          style: TextStyle(
                            fontSize: 20,
                            fontWeight: FontWeight.w600,
                          ),
                        ),
                      ),
                      ...List.generate(
                        detectedObjectList.length,
                        (index) {
                          final detectedObject = detectedObjectList[index];
                          return Padding(
                            padding: const EdgeInsets.only(bottom: 12),
                            child: DetectedObjectTile(
                              label: detectedObject.label,
                              value: detectedObject.score.toString(),
                            ),
                          );
                        },
                      ),
                    ],
                  ),
          ),
        ],
      ),
    );
  }

  void _analyzeImage() {
    Future.delayed(
      const Duration(milliseconds: 600),
      () {
        final output = TensorflowService.ssdMobileNet.analyseImage(
          widget.imageBytes,
        );
        image = output.imageBytes;
        detectedObjectList = output.detectedObjects;
        SnackBarService.remove();
        if (mounted) setState(() {});
      },
    );
  }
}
