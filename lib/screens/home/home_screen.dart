import 'package:flutter/material.dart';
import 'package:flutter_mobx/flutter_mobx.dart';
import 'package:flutter_svg/svg.dart';
import 'package:provider/provider.dart';
import 'package:tensorflow_demo/models/screen_params.dart';
import 'package:tensorflow_demo/services/navigation_service.dart';
import 'package:tensorflow_demo/values/app_routes.dart';
import 'package:tensorflow_demo/values/enumerations.dart';

import 'home_screen_store.dart';

class HomeScreen extends StatelessWidget {
  const HomeScreen({super.key});

  @override
  Widget build(BuildContext context) {
    ScreenParams.screenSize = MediaQuery.sizeOf(context);
    final homeScreenStore = context.read<HomeScreenStore>();
    return Scaffold(
      appBar: AppBar(
        title: const Text('Home'),
      ),
      floatingActionButton: FloatingActionButton(
        heroTag: 'live_object_detection',
        onPressed: () => NavigationService.instance.pushNamed(
          AppRoutes.cameraScreen,
        ),
        child: SvgPicture.asset(
          'assets/vectors/camera.svg',
          width: 28,
          height: 28,
        ),
      ),
      body: Observer(
        builder: (_) {
          return switch (homeScreenStore.unsplashPhotosState) {
            NetworkState.idle => const SizedBox.shrink(),
            NetworkState.loading => const Center(
                child: RepaintBoundary(child: CircularProgressIndicator()),
              ),
            NetworkState.error => Center(
                child: Column(
                  mainAxisAlignment: MainAxisAlignment.center,
                  children: [
                    const Text('Something Went Wrong!'),
                    const SizedBox(height: 12),
                    FilledButton(
                      onPressed: homeScreenStore.refresh,
                      child: const Text('Retry'),
                    )
                  ],
                ),
              ),
            NetworkState.success => RefreshIndicator(
                onRefresh: homeScreenStore.refresh,
                child: ListView(
                  padding: const EdgeInsets.all(16),
                  physics: const AlwaysScrollableScrollPhysics(),
                  controller: homeScreenStore.scrollController,
                  children: [
                    Observer(
                      builder: (_) => GridView.builder(
                        shrinkWrap: true,
                        primary: false,
                        itemCount: homeScreenStore.photos.length,
                        gridDelegate:
                            const SliverGridDelegateWithFixedCrossAxisCount(
                          crossAxisCount: 2,
                          childAspectRatio: 3 / 4,
                          crossAxisSpacing: 16,
                          mainAxisSpacing: 16,
                        ),
                        itemBuilder: (_, index) {
                          final urls = homeScreenStore.photos[index].urls;
                          return GestureDetector(
                            onTap: () async =>
                                homeScreenStore.analyzeNetworkImage(
                              urls.small,
                            ),
                            child: ClipRRect(
                              borderRadius: const BorderRadius.all(
                                Radius.circular(16),
                              ),
                              child: Image.network(
                                urls.small,
                                fit: BoxFit.cover,
                              ),
                            ),
                          );
                        },
                      ),
                    ),
                    Observer(
                      builder: (_) => Visibility(
                        visible: homeScreenStore.paginatedState.isLoading,
                        child: const Center(
                          child: RepaintBoundary(
                            child: CircularProgressIndicator(),
                          ),
                        ),
                      ),
                    ),
                  ],
                ),
              ),
          };
        },
      ),
    );
  }
}
