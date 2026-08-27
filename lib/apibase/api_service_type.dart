import 'dart:io';

import 'package:dio/dio.dart';
import 'package:dio/io.dart';

import 'api_service.dart';

class ApiServiceType {
  const ApiServiceType._();

  static final ApiService unsplashApiService = _getApiService(
    options: BaseOptions(
      baseUrl: 'https://api.unsplash.com',
      headers: {
        'Accept-Version': 'v1',
        'Authorization': 'Client-ID pZeXv9RBiVnjgWdRzEe3t1ruAHDwG7mX0R6geEwlHBI',
      },
    ),
  );

  static ApiService _getApiService({
    required BaseOptions options,
    List<Interceptor>? interceptors,
    bool bypassesSSLCert = false,
  }) {
    final dio = Dio(options);

    /// Configure Dio to use a custom HttpClientAdapter with SSL certificate validation disabled
    ///
    /// if the `bypassesSSLCert` flag is set to true.
    if (bypassesSSLCert) {
      // Set up a custom HttpClientAdapter for Dio.
      dio.httpClientAdapter = IOHttpClientAdapter(
        createHttpClient: () {
          final client = HttpClient(
            context: SecurityContext(withTrustedRoots: false),
          );
          // Accept all certificates.
          client.badCertificateCallback = (cert, host, port) => true;
          return client;
        },
      );
    }

    dio.interceptors.addAll([
      ...?interceptors,
    ]);
    return ApiService(dio);
  }
}
