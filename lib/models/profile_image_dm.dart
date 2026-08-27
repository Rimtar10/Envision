import 'package:json_annotation/json_annotation.dart';

part 'profile_image_dm.g.dart';

@JsonSerializable(fieldRename: FieldRename.snake)
class ProfileImageDm {
  const ProfileImageDm({
    required this.small,
    required this.medium,
    required this.large,
  });

  final String small;
  final String medium;
  final String large;

  factory ProfileImageDm.fromJson(Map<String, dynamic> json) =>
      _$ProfileImageDmFromJson(json);

  Map<String, dynamic> toJson() => _$ProfileImageDmToJson(this);
}
