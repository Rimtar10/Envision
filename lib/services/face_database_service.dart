import 'dart:convert';
import 'package:path/path.dart';
import 'package:sqflite/sqflite.dart';

/// Stores registered faces (name + embedding) in a local SQLite database.
class FaceDatabaseService {
  static final FaceDatabaseService instance = FaceDatabaseService._();
  FaceDatabaseService._();

  Database? _db;

  Future<Database> get database async {
    _db ??= await _initDb();
    return _db!;
  }

  Future<Database> _initDb() async {
    final dbPath = await getDatabasesPath();
    final path = join(dbPath, 'faces.db');

    return openDatabase(
      path,
      version: 1,
      onCreate: (db, version) async {
        await db.execute('''
          CREATE TABLE faces (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            embedding TEXT NOT NULL,
            created_at TEXT NOT NULL
          )
        ''');
      },
    );
  }

  /// Save a new face with a name and its embedding vector.
  Future<void> registerFace(String name, List<double> embedding) async {
    final db = await database;
    await db.insert('faces', {
      'name': name,
      'embedding': jsonEncode(embedding),
      'created_at': DateTime.now().toIso8601String(),
    });
  }

  /// Load all registered faces from the database.
  Future<List<RegisteredFace>> getAllFaces() async {
    final db = await database;
    final rows = await db.query('faces');
    return rows.map((row) {
      final embedding = (jsonDecode(row['embedding'] as String) as List)
          .map((e) => (e as num).toDouble())
          .toList();
      return RegisteredFace(
        id: row['id'] as int,
        name: row['name'] as String,
        embedding: embedding,
      );
    }).toList();
  }

  /// Delete a registered face by id.
  Future<void> deleteFace(int id) async {
    final db = await database;
    await db.delete('faces', where: 'id = ?', whereArgs: [id]);
  }

  /// Delete all registered faces.
  Future<void> clearAll() async {
    final db = await database;
    await db.delete('faces');
  }
}

/// A registered face entry from the database.
class RegisteredFace {
  final int id;
  final String name;
  final List<double> embedding;

  const RegisteredFace({
    required this.id,
    required this.name,
    required this.embedding,
  });
}
