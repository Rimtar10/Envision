import 'package:json_annotation/json_annotation.dart';
import 'package:tensorflow_demo/models/profile_image_dm.dart';
import 'package:tensorflow_demo/models/user_links_dm.dart';

part 'user_dm.g.dart';

@JsonSerializable(explicitToJson: true, fieldRename: FieldRename.snake)
class UserDm {
  const UserDm({
    required this.id,
    required this.username,
    required this.name,
    required this.firstName,
    required this.links,
    required this.profileImage,
    required this.totalCollections,
    required this.totalLikes,
    required this.totalPhotos,
    this.lastName,
    this.portfolioUrl,
    this.bio,
  });

  final String id;
  final String username;
  final String name;
  final String firstName;
  final String? lastName;
  final String? portfolioUrl;
  final String? bio;
  final UserLinksDm links;
  final ProfileImageDm profileImage;
  final int totalCollections;
  final int totalLikes;
  final int totalPhotos;

  factory UserDm.fromJson(Map<String, dynamic> json) => _$UserDmFromJson(json);

  Map<String, dynamic> toJson() => _$UserDmToJson(this);
}
