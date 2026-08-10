import 'package:json_annotation/json_annotation.dart';
import 'package:tensorflow_demo/models/create_session_links_dm.dart';
import 'package:tensorflow_demo/models/urls_dm.dart';
import 'package:tensorflow_demo/models/user_dm.dart';

part 'create_session_dm.g.dart';

@JsonSerializable(explicitToJson: true, fieldRename: FieldRename.snake)
class CreateSessionDm {
  const CreateSessionDm({
    required this.id,
    required this.slug,
    required this.width,
    required this.height,
    required this.color,
    required this.blurHash,
    required this.description,
    required this.urls,
    required this.links,
    required this.currentUserCollections,
    required this.user,
  });

  final String id;
  final String slug;
  final int width;
  final int height;
  final String color;
  final String? blurHash;
  final String? description;
  final UrlsDm urls;
  final CreateSessionLinksDm links;
  final List<dynamic> currentUserCollections;
  final UserDm user;

  factory CreateSessionDm.fromJson(Map<String, dynamic> json) =>
      _$CreateSessionDmFromJson(json);

  Map<String, dynamic> toJson() => _$CreateSessionDmToJson(this);
}
