import 'package:json_annotation/json_annotation.dart';

part 'current_user_collection_dm.g.dart';

@JsonSerializable(fieldRename: FieldRename.snake)
class CurrentUserCollectionDm {
  const CurrentUserCollectionDm({
    required this.id,
    required this.title,
    required this.publishedAt,
    required this.lastCollectedAt,
    required this.updatedAt,
    required this.coverPhoto,
    required this.user,
  });

  final int id;
  final String title;
  final DateTime publishedAt;
  final DateTime lastCollectedAt;
  final DateTime updatedAt;
  final dynamic coverPhoto;
  final dynamic user;

  factory CurrentUserCollectionDm.fromJson(Map<String, dynamic> json) =>
      _$CurrentUserCollectionDmFromJson(json);

  Map<String, dynamic> toJson() => _$CurrentUserCollectionDmToJson(this);
}
