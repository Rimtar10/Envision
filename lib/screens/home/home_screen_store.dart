import 'dart:async';
import 'dart:developer';

import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:image_picker/image_picker.dart';
import 'package:mobx/mobx.dart';

import '../../apibase/api_service_type.dart';
import '../../models/create_session_dm.dart';
import '../../services/navigation_service.dart';
import '../../values/app_routes.dart';
import '../../values/enumerations.dart';

part 'home_screen_store.g.dart';

class HomeScreenStore = _HomeScreenStore with _$HomeScreenStore;

abstract class _HomeScreenStore with Store {
  @observable
  String searchQuery = 'car ';

  final scrollController = ScrollController();

  final imagePicker = ImagePicker();

  @observable
  NetworkState unsplashPhotosState = NetworkState.idle;

  @observable
  NetworkState paginatedState = NetworkState.idle;

  @observable
  ObservableList<CreateSessionDm> photos = ObservableList();

  int _currentPage = 1;

  final int _totalPages = 10;

  void initialize() {
    unawaited(getUnsplashPhotos());
    scrollController.addListener(
      () {
        final maxScroll = scrollController.position.maxScrollExtent;
        final currentScroll = scrollController.position.pixels;
        if (currentScroll == maxScroll) {
          fetchMore();
        }
      },
    );
  }

  Future<void> refresh() async {
    _currentPage = 1;
    photos.clear();
    await getUnsplashPhotos();
  }

  Future<void> getUnsplashPhotos() async {
    unsplashPhotosState = NetworkState.loading;
    try {
      final result = await ApiServiceType.unsplashApiService.searchPhotos(
        page: _currentPage,
        search: searchQuery,
      );
      _currentPage++;
      photos.addAll(result.results);
      unsplashPhotosState = NetworkState.success;
    } catch (e, s) {
      log('Error: $e', name: 'getUnsplashPhotos()');
      log('Stacktrace: $s', name: 'getUnsplashPhotos()');
      unsplashPhotosState = NetworkState.error;
    }
  }

  Future<void> fetchMore() async {
    if (paginatedState.isLoading || _currentPage > _totalPages) return;

    paginatedState = NetworkState.loading;
    try {
      final result = await ApiServiceType.unsplashApiService.searchPhotos(
        page: _currentPage,
        search: searchQuery,
      );
      _currentPage++;
      result.results
        ..removeAt(0)
        ..removeAt(0);
      photos.addAll(result.results);
      paginatedState = NetworkState.success;
    } catch (e, s) {
      log('Error: $e', name: 'getUnsplashPhotos()');
      log('Stacktrace: $s', name: 'getUnsplashPhotos()');
      paginatedState = NetworkState.error;
    }
  }

  Future<void> analyzeNetworkImage(String url) async {
    final response = await NetworkAssetBundle(Uri.parse(url)).load(url);
    NavigationService.instance.pushNamed(
      AppRoutes.photoAnalyzedScreen,
      arguments: response.buffer.asUint8List(),
    );
  }

  Future<void> pickImageFromCamera() async {
    NavigationService.instance.pushNamed(
      AppRoutes.cameraScreen,
    );
  }

  Future<void> pickImageFromGallery() async {
    final result = await imagePicker.pickImage(
      source: ImageSource.gallery,
    );
    final readAsBytesSync = await result?.readAsBytes();
    if (readAsBytesSync != null) {
      NavigationService.instance.pushNamed(
        AppRoutes.photoAnalyzedScreen,
        arguments: readAsBytesSync,
      );
    }
  }
}
