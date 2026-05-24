import 'dart:typed_data';

import 'package:flutter/material.dart';
import 'package:tensorflow_demo/screens/live_object_detection/live_object_detection_screen.dart';
import 'package:tensorflow_demo/screens/photo_analyzed/photo_analyze_screen.dart';
import 'package:tensorflow_demo/values/app_routes.dart';

class Routes {
  static Route<dynamic> generateRoute(RouteSettings settings) {
    Route<dynamic> getRoute({
      required Widget widget,
      bool fullscreenDialog = false,
    }) {
      return MaterialPageRoute<void>(
        builder: (context) => widget,
        settings: settings,
        fullscreenDialog: fullscreenDialog,
      );
    }

    switch (settings.name) {
      // '/' — root of the app — goes directly to the camera.
      // HomeScreen (Unsplash API) is removed from the navigation stack entirely.
      case AppRoutes.homeScreen:
        return getRoute(widget: const LiveObjectDetectionScreen());

      case AppRoutes.cameraScreen:
        return getRoute(widget: const LiveObjectDetectionScreen());

      case AppRoutes.photoAnalyzedScreen:
        final imageBytes = settings.arguments as Uint8List?;
        // Guard: if somehow bytes are missing, go back to the camera.
        if (imageBytes == null || imageBytes.isEmpty) {
          return getRoute(widget: const LiveObjectDetectionScreen());
        }
        return getRoute(widget: PhotoAnalyzedScreen(imageBytes: imageBytes));

      default:
        return getRoute(widget: const LiveObjectDetectionScreen());
    }
  }
}
