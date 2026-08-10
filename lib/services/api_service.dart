import 'package:dio/dio.dart';
import 'package:flutter/foundation.dart';

/// Result from the server recognition endpoint
class ServerRecognitionResult {
  final String name;
  final double confidence;
  final String status;

  const ServerRecognitionResult({
    required this.name,
    required this.confidence,
    required this.status,
  });

  bool get isKnown => name != 'Unknown' && confidence > 0.6;
}

/// Handles all HTTP communication with the Flask face recognition server
class ApiService {
  static final ApiService instance = ApiService._();
  ApiService._();

  // ── Server configuration ──────────────────────────────────────────────────
  // Change this to your PC's local IP address
  static const String _serverIp = '192.168.1.107';
  static const int _serverPort = 5000;
  static String get baseUrl => 'http://$_serverIp:$_serverPort';

  // ── State ─────────────────────────────────────────────────────────────────
  bool _isOnline = false;
  DateTime? _lastCheck;
  static const _checkInterval = Duration(seconds: 10);

  bool get isOnline => _isOnline;

  final Dio _dio = Dio(BaseOptions(
    connectTimeout: const Duration(seconds: 3),
    receiveTimeout: const Duration(seconds: 10),
    sendTimeout: const Duration(seconds: 10),
  ));

  // ── Server Status ─────────────────────────────────────────────────────────

  /// Check if the server is reachable.
  /// Cached for 10 seconds to avoid spamming.
  Future<bool> checkStatus() async {
    final now = DateTime.now();
    if (_lastCheck != null && now.difference(_lastCheck!) < _checkInterval) {
      return _isOnline;
    }

    try {
      final response = await _dio.get('$baseUrl/status');
      _isOnline = response.statusCode == 200;
      _lastCheck = now;
      if (_isOnline) {
        debugPrint('[API] Server online. People: ${response.data['people']}');
      }
    } catch (e) {
      _isOnline = false;
      _lastCheck = now;
      debugPrint('[API] Server offline: $e');
    }
    return _isOnline;
  }

  /// Force a fresh status check (ignores cache)
  Future<bool> forceCheckStatus() async {
    _lastCheck = null;
    return checkStatus();
  }

  // ── Recognition ───────────────────────────────────────────────────────────

  /// Send a face image to the server for recognition.
  /// Returns null if server is unreachable.
  Future<ServerRecognitionResult?> recognize(Uint8List imageBytes) async {
    try {
      final formData = FormData.fromMap({
        'image': MultipartFile.fromBytes(
          imageBytes,
          filename: 'face.jpg',
        ),
      });

      final response = await _dio.post(
        '$baseUrl/recognize',
        data: formData,
      );

      if (response.statusCode == 200) {
        final data = response.data;
        return ServerRecognitionResult(
          name: data['name'] ?? 'Unknown',
          confidence: (data['confidence'] ?? 0.0).toDouble(),
          status: data['status'] ?? 'unknown',
        );
      }
    } catch (e) {
      debugPrint('[API] Recognition error: $e');
      _isOnline = false;
    }
    return null;
  }

  // ── Registration ──────────────────────────────────────────────────────────

  /// Register a new person on the server with multiple face images.
  /// Returns true if successful.
  Future<bool> register(String name, List<Uint8List> images) async {
    try {
      final formData = FormData.fromMap({
        'name': name,
        'images': images.asMap().entries.map((entry) {
          return MultipartFile.fromBytes(
            entry.value,
            filename: 'img${entry.key + 1}.jpg',
          );
        }).toList(),
      });

      final response = await _dio.post(
        '$baseUrl/register',
        data: formData,
        options: Options(
          receiveTimeout: const Duration(seconds: 60), // training takes time
        ),
      );

      if (response.statusCode == 200) {
        final status = response.data['status'];
        debugPrint('[API] Registration result: ${response.data['message']}');
        return status == 'success';
      }
    } catch (e) {
      debugPrint('[API] Registration error: $e');
    }
    return false;
  }

  // ── People ────────────────────────────────────────────────────────────────

  /// Get list of all registered people from server.
  Future<List<String>> getPeople() async {
    try {
      final response = await _dio.get('$baseUrl/people');
      if (response.statusCode == 200) {
        return List<String>.from(response.data['people'] ?? []);
      }
    } catch (e) {
      debugPrint('[API] Get people error: $e');
    }
    return [];
  }

  /// Delete a person from the server.
  Future<bool> deletePerson(String name) async {
    try {
      final response = await _dio.delete('$baseUrl/delete/$name');
      return response.statusCode == 200;
    } catch (e) {
      debugPrint('[API] Delete error: $e');
      return false;
    }
  }
}
