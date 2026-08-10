import 'package:json_annotation/json_annotation.dart';
import 'package:tensorflow_demo/models/create_session_dm.dart';

part 'paginated_res_dm.g.dart';

@JsonSerializable(fieldRename: FieldRename.snake)
class PaginatedResDm {
  const PaginatedResDm({
    required this.total,
    required this.totalPages,
    required this.results,
  });

  final int total;
  final int totalPages;
  final List<CreateSessionDm> results;

  factory PaginatedResDm.fromJson(Map<String, dynamic> json) =>
      _$PaginatedResDmFromJson(json);

  Map<String, dynamic> toJson() => _$PaginatedResDmToJson(this);
}
