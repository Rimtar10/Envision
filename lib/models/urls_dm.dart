import 'package:json_annotation/json_annotation.dart';

part 'urls_dm.g.dart';

@JsonSerializable(fieldRename: FieldRename.snake)
class UrlsDm {
  const UrlsDm({
    required this.raw,
    required this.full,
    required this.regular,
    required this.small,
    required this.thumb,
  });

  final String raw;
  final String full;
  final String regular;
  final String small;
  final String thumb;

  factory UrlsDm.fromJson(Map<String, dynamic> json) => _$UrlsDmFromJson(json);

  Map<String, dynamic> toJson() => _$UrlsDmToJson(this);
}
