import 'package:dio/dio.dart';
import 'package:flutter/foundation.dart';

class MistralService {
  static final MistralService instance = MistralService._();
  MistralService._();

  static const String _apiKey = String.fromEnvironment('MISTRAL_API_KEY');
  static const String _baseUrl = 'https://api.mistral.ai/v1/chat/completions';
  static const String _model = 'mistral-small-latest';

  bool get isConfigured => _apiKey.isNotEmpty;

  final Dio _dio = Dio(BaseOptions(
    connectTimeout: const Duration(seconds: 5),
    receiveTimeout: const Duration(seconds: 12),
    sendTimeout: const Duration(seconds: 8),
    headers: {'Content-Type': 'application/json'},
  ));

  final List<Map<String, String>> _chatHistory = [];
  static const int _maxHistoryMessages = 8;
  static const List<String> knownCommands = [
    'stop',
    'help',
    'repeat',
    'time',
    'battery',
    'date',
    'read text',
    'describe',
    'take a photo',
    'turn on face recognition',
    'turn off face recognition',
    'register a face',
    'start capture',
    'retake',
    'confirm registration',
    'read again',
    'go back',
    'open camera',
  ];

  Future<String?> classifyCommand(String transcript) async {
    if (!isConfigured) return null;
    try {
      final response = await _dio.post(
        _baseUrl,
        options: Options(headers: {'Authorization': 'Bearer $_apiKey'}),
        data: {
          'model': _model,
          'messages': [
            {
              'role': 'system',
              'content':
                  'You map a spoken command transcript from a blind user\'s '
                      'assistive app to exactly one of these commands: '
                      '${knownCommands.join(", ")}. '
                      'Reply with ONLY the matching command text, exactly as '
                      'written above, and nothing else. If nothing matches '
                      'well, reply with exactly: none',
            },
            {'role': 'user', 'content': transcript},
          ],
          'temperature': 0,
          'max_tokens': 20,
        },
      );
      final content =
          response.data['choices']?[0]?['message']?['content'] as String?;
      final cleaned = content?.trim().toLowerCase();
      if (cleaned == null || cleaned == 'none' || cleaned.isEmpty) {
        return null;
      }
      final match = knownCommands.firstWhere(
        (c) => c.toLowerCase() == cleaned,
        orElse: () => '',
      );
      return match.isEmpty ? null : match;
    } catch (e) {
      debugPrint('[Mistral] classifyCommand error: $e');
      return null;
    }
  }

  Future<String> chat(String message) async {
    if (!isConfigured) {
      return "The assistant isn't set up yet — a Mistral API key needs to "
          'be added to the build.';
    }
    try {
      _chatHistory.add({'role': 'user', 'content': message});
      if (_chatHistory.length > _maxHistoryMessages) {
        _chatHistory.removeRange(0, _chatHistory.length - _maxHistoryMessages);
      }

      final response = await _dio.post(
        _baseUrl,
        options: Options(headers: {'Authorization': 'Bearer $_apiKey'}),
        data: {
          'model': _model,
          'messages': [
            {
              'role': 'system',
              'content':
                  'You are Envision, a spoken voice assistant inside an app '
                      'for blind and visually impaired users. Answer briefly in '
                      'plain spoken sentences — no markdown, no bullet points, '
                      'no lists, nothing that only makes sense visually. Keep '
                      'answers to 2-3 short sentences unless asked for more '
                      'detail.',
            },
            ..._chatHistory,
          ],
          'temperature': 0.4,
          'max_tokens': 200,
        },
      );
      final reply =
          response.data['choices']?[0]?['message']?['content'] as String? ??
              "Sorry, I don't have an answer for that.";
      _chatHistory.add({'role': 'assistant', 'content': reply});
      return reply.trim();
    } catch (e) {
      debugPrint('[Mistral] chat error: $e');
      return "Sorry, I couldn't reach the assistant right now.";
    }
  }

  /// Clears the in-session conversation history. Not required for
  /// persistence (there is none), but useful to reset context explicitly.
  void clearHistory() => _chatHistory.clear();
}
