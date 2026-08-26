import 'package:flutter/material.dart';
import 'package:permission_handler/permission_handler.dart';
import 'package:tensorflow_demo/services/snackbar_service.dart';
import 'package:tensorflow_demo/services/voice_service.dart';

import 'routes.dart';
import 'services/navigation_service.dart';
import 'values/app_routes.dart';

class MyApp extends StatefulWidget {
  const MyApp({super.key});

  @override
  State<MyApp> createState() => _MyAppState();
}

class _MyAppState extends State<MyApp> {
  // TEMPORARILY DISABLED — the always-on wake-word loop was causing a
  // repeating start/stop mic beep even after the permission/backoff fix.
  // Flip this back to true once the root cause is found; tap-to-speak and
  // all other voice buttons still work fine with this off.
  static const bool _ambientVoiceEnabled = false;

  @override
  void initState() {
    super.initState();
    if (_ambientVoiceEnabled) {
      // Enable "envision" wake-word listening app-wide, so voice commands
      // work hands-free on every screen — not just while the camera screen
      // is open. Whichever screen is currently mounted registers the
      // callbacks it can serve (see VoiceService's onXRequested fields), so
      // the same wake-word/command pipeline routes correctly everywhere.
      WidgetsBinding.instance.addPostFrameCallback((_) => _enableAmbientVoice());
    }
  }

  Future<void> _enableAmbientVoice() async {
    try {
      final status = await Permission.microphone.request();
      if (!status.isGranted) {
        // Without mic permission every listen() call fails instantly, which
        // makes the wake-word loop restart in a tight cycle — audible as a
        // start/stop beep every second. Don't enable it until we actually
        // have permission; screens that need the mic (camera, registration)
        // will prompt again themselves when the user reaches them.
        debugPrint('[App] Mic permission not granted — wake word disabled.');
        return;
      }
      await VoiceService.instance.ensureInitialized();
      await VoiceService.instance.setWakeWordEnabled(true);
    } catch (e) {
      debugPrint('[App] Could not enable ambient voice listening: $e');
    }
  }

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Envision',
      navigatorKey: NavigationService.instance.key,
      theme: ThemeData(
        useMaterial3: true,
        colorScheme: ColorScheme.fromSeed(seedColor: Colors.red),
        appBarTheme: const AppBarTheme(scrolledUnderElevation: 0),
      ),
      debugShowCheckedModeBanner: false,
      initialRoute: AppRoutes.homeScreen,
      onGenerateRoute: Routes.generateRoute,
      scaffoldMessengerKey: SnackBarService.scaffoldMessengerKey,
    );
  }
}
