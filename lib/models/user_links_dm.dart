import 'package:json_annotation/json_annotation.dart';

part 'user_links_dm.g.dart';

@JsonSerializable(fieldRename: FieldRename.snake)
class UserLinksDm {
  const UserLinksDm({
    required this.self,
    required this.html,
    required this.photos,
    required this.likes,
    required this.portfolio,
  });

  final String self;
  final String html;
  final String photos;
  final String likes;
  final String portfolio;

  factory UserLinksDm.fromJson(Map<String, dynamic> json) =>
      _$UserLinksDmFromJson(json);

  Map<String, dynamic> toJson() => _$UserLinksDmToJson(this);
}
