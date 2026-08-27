// This file used to be the stock Flutter template test: it pumped MyApp, then
// looked for a counter showing "0" and an Icons.add button to tap. Envision has
// no counter and no add button, so that test failed every single run — which
// meant `flutter test` was permanently red and nobody could tell a real
// regression from the template noise.
//
// It is replaced with a check that actually holds.
//
// NOTE ON SCOPE: a full `pumpWidget(const MyApp())` is deliberately NOT done
// here. MyApp's initial route builds the camera screen, which needs the camera,
// microphone and TFLite plugins — none of which exist in the `flutter test`
// host VM. That belongs in integration_test/ running on a real device.
//
// The logic worth protecting is pure and lives in:
//   test/letterbox_parity_test.dart  — rotation + letterbox geometry
//   test/detection_logic_test.dart   — pinhole distance, yStretch correction

import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:tensorflow_demo/app.dart';

void main() {
  test('MyApp is a const-constructible widget', () {
    // Cheap, but it does catch the common breakage: a compile error anywhere in
    // the app.dart -> routes.dart -> screens import graph fails this test.
    const app = MyApp();
    expect(app, isA<StatelessWidget>());
  });
}
