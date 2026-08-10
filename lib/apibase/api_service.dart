import 'package:dio/dio.dart';
import 'package:retrofit/retrofit.dart';
import 'package:tensorflow_demo/models/create_session_dm.dart';
import 'package:tensorflow_demo/models/paginated_res_dm.dart';

part 'api_service.g.dart';

@RestApi()
abstract class ApiService {
  factory ApiService(Dio dio, {String baseUrl}) = _ApiService;

  @GET('/photos')
  Future<List<CreateSessionDm>> getPhotos({
    @Query('page') required int page,
  });

  @GET('/search/photos')
  Future<PaginatedResDm> searchPhotos({
    @Query('page') required int page,
    @Query('query') required String search,
  });
}
