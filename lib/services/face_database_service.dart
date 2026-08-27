import 'dart:convert';
import 'dart:io';

import 'package:path/path.dart';
import 'package:sqflite/sqflite.dart';

/// Stores registered faces in a local SQLite database on the phone.
///
/// WHAT IS STORED, AND WHY
/// -----------------------
/// Not photographs — *embeddings*. An embedding is the 192 numbers
/// MobileFaceNet produces for a face crop; two photos of the same person land
/// close together in that space and two different people land far apart.
/// Matching is then a distance comparison over a few hundred floats, which is
/// fast enough to run on every camera frame.
///
/// Storing the photos themselves and comparing images directly would be far
/// slower (you would have to re-run the model over every stored photo on every
/// frame), and it would put other people's faces in the phone's photo gallery,
/// which is not something a user consents to when they register a friend.
///
/// Each person gets ONE row holding:
///   embedding — the mean of all their enrolment shots (the "template")
///   samples   — every individual shot's embedding, so matching can use
///               best-of-N instead of just the average. A person photographed
///               from 7 slightly different angles matches far more reliably
///               against the closest single shot than against the blur of all
///               of them averaged together.
class FaceDatabaseService {
  static final FaceDatabaseService instance = FaceDatabaseService._();
  FaceDatabaseService._();

  Database? _db;

  /// Bumped to 2 when per-sample embeddings were added. The v1 rows were
  /// computed from a centre-crop of the whole photo rather than a detected
  /// face, so they could never match anything — they are dropped rather than
  /// migrated.
  static const int _schemaVersion = 2;

  Future<Database> get database async {
    _db ??= await _initDb();
    return _db!;
  }

  Future<Database> _initDb() async {
    final dbPath = await getDatabasesPath();
    final path = join(dbPath, 'faces.db');

    return openDatabase(
      path,
      version: _schemaVersion,
      onCreate: (db, version) async => _createTable(db),
      onUpgrade: (db, oldVersion, newVersion) async {
        await db.execute('DROP TABLE IF EXISTS faces');
        await _createTable(db);
      },
    );
  }

  Future<void> _createTable(Database db) async {
    await db.execute('''
      CREATE TABLE faces (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        embedding TEXT NOT NULL,
        samples TEXT,
        shots INTEGER NOT NULL DEFAULT 1,
        created_at TEXT NOT NULL
      )
    ''');
  }

  /// Save a person. [samples] is one embedding per enrolment shot;
  /// [embedding] is their mean template.
  ///
  /// Registering a name that already exists REPLACES the old entry — otherwise
  /// re-registering someone to improve their recognition would instead leave
  /// two competing entries, one of them bad.
  Future<void> registerFace(
    String name,
    List<double> embedding, {
    List<List<double>> samples = const [],
  }) async {
    final db = await database;
    await db.delete('faces', where: 'name = ? COLLATE NOCASE', whereArgs: [name]);
    await db.insert('faces', {
      'name': name,
      'embedding': jsonEncode(embedding),
      'samples': jsonEncode(samples),
      'shots': samples.isEmpty ? 1 : samples.length,
      'created_at': DateTime.now().toIso8601String(),
    });
  }

  /// Load every registered face.
  Future<List<RegisteredFace>> getAllFaces() async {
    final db = await database;
    final rows = await db.query('faces', orderBy: 'name COLLATE NOCASE');
    return rows.map(_rowToFace).toList();
  }

  RegisteredFace _rowToFace(Map<String, Object?> row) {
    List<double> decodeVector(Object? raw) {
      if (raw is! String || raw.isEmpty) return const [];
      return (jsonDecode(raw) as List)
          .map((e) => (e as num).toDouble())
          .toList();
    }

    List<List<double>> decodeSamples(Object? raw) {
      if (raw is! String || raw.isEmpty) return const [];
      try {
        return (jsonDecode(raw) as List)
            .map((s) => (s as List).map((e) => (e as num).toDouble()).toList())
            .toList();
      } catch (_) {
        return const [];
      }
    }

    return RegisteredFace(
      id: row['id'] as int,
      name: row['name'] as String,
      embedding: decodeVector(row['embedding']),
      samples: decodeSamples(row['samples']),
      shots: (row['shots'] as int?) ?? 1,
    );
  }

  Future<void> deleteFace(int id) async {
    final db = await database;
    await db.delete('faces', where: 'id = ?', whereArgs: [id]);
  }

  Future<void> clearAll() async {
    final db = await database;
    await db.delete('faces');
  }

  // ── Enrolment crops ───────────────────────────────────────────────────────

  /// Where the aligned 112×112 face crops are kept: a private folder beside
  /// faces.db, NOT the phone's photo gallery.
  ///
  /// These are saved purely so registration is inspectable — if someone is
  /// never recognised, looking at what was actually enrolled tells you
  /// immediately whether the camera captured a face or a blurred ceiling. They
  /// also mean everyone can be re-enrolled without re-photographing them if
  /// the face model is ever swapped.
  Future<Directory> cropsDirectory() async {
    final dir = Directory(join(await getDatabasesPath(), 'face_crops'));
    if (!await dir.exists()) await dir.create(recursive: true);
    return dir;
  }

  /// Save one aligned crop as `<name>_<index>.jpg`. Returns its path, or null.
  Future<String?> saveFaceCrop(String name, int index, List<int> jpeg) async {
    try {
      final safe = name.replaceAll(RegExp(r'[^A-Za-z0-9_-]'), '_');
      final file = File(join((await cropsDirectory()).path, '${safe}_$index.jpg'));
      await file.writeAsBytes(jpeg, flush: true);
      return file.path;
    } catch (_) {
      return null;
    }
  }
}

/// A registered person loaded from the database.
class RegisteredFace {
  final int id;
  final String name;

  /// Mean template across all enrolment shots (L2-normalised).
  final List<double> embedding;

  /// One L2-normalised embedding per enrolment shot. May be empty for rows
  /// written before per-sample storage existed.
  final List<List<double>> samples;

  /// How many shots this person was enrolled from.
  final int shots;

  const RegisteredFace({
    required this.id,
    required this.name,
    required this.embedding,
    this.samples = const [],
    this.shots = 1,
  });
}
