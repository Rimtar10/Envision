import 'package:json_annotation/json_annotation.dart';

part 'create_session_links_dm.g.dart';

@JsonSerializable(fieldRename: FieldRename.snake)
class CreateSessionLinksDm {
  const CreateSessionLinksDm({
    required this.self,
    required this.html,
    required this.download,
    required this.downloadLocation,
  });

  final String self;
  final String html;
  final String download;
  final String downloadLocation;

  factory CreateSessionLinksDm.fromJson(Map<String, dynamic> json) =>
      _$CreateSessionLinksDmFromJson(json);

  Map<String, dynamic> toJson() => _$CreateSessionLinksDmToJson(this);
}
