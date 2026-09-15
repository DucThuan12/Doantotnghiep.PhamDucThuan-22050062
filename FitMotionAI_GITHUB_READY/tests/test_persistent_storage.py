from __future__ import annotations

import os
import sqlite3
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from storage import backup_database, prepare_persistent_database

try:
    import ultralytics  # noqa: F401
except ModuleNotFoundError:
    sys.modules["ultralytics"] = types.SimpleNamespace(YOLO=lambda *args, **kwargs: None)

from workoutlogic import BaseProcessor, SquatProcessor


class PersistentStorageTests(unittest.TestCase):
    def _legacy_database(self, root: Path, total_rep: int = 17) -> Path:
        path = root / "aifitness.db"
        connection = sqlite3.connect(path)
        try:
            connection.executescript(
                """
                CREATE TABLE users (id INTEGER PRIMARY KEY);
                CREATE TABLE workout_exercises (id INTEGER PRIMARY KEY);
                CREATE TABLE workout_sessions (
                    id INTEGER PRIMARY KEY,
                    total_rep INTEGER,
                    good_rep INTEGER,
                    total_error INTEGER,
                    keypoint_confidence_avg REAL,
                    quality_score_avg REAL,
                    rep_details_json TEXT,
                    error_codes_json TEXT
                );
                """
            )
            connection.execute(
                "INSERT INTO workout_sessions(id,total_rep,good_rep,total_error) VALUES(1,?,?,?)",
                (total_rep, total_rep - 2, 2),
            )
            connection.commit()
        finally:
            connection.close()
        return path

    def test_legacy_database_is_copied_to_stable_data_directory(self):
        with tempfile.TemporaryDirectory() as source_dir, tempfile.TemporaryDirectory() as data_dir:
            source = Path(source_dir)
            legacy = self._legacy_database(source, total_rep=17)
            with patch.dict(os.environ, {"FITMOTION_DATA_DIR": data_dir}, clear=False):
                _, database_path, note = prepare_persistent_database(source)
            self.assertNotEqual(legacy.resolve(), database_path.resolve())
            self.assertIn("Đã sao chép", note)
            connection = sqlite3.connect(database_path)
            try:
                self.assertEqual(
                    17,
                    connection.execute("SELECT total_rep FROM workout_sessions WHERE id=1").fetchone()[0],
                )
            finally:
                connection.close()

    def test_reopening_another_source_copy_reuses_same_database(self):
        with tempfile.TemporaryDirectory() as source_a, tempfile.TemporaryDirectory() as source_b, tempfile.TemporaryDirectory() as data_dir:
            self._legacy_database(Path(source_a), total_rep=11)
            self._legacy_database(Path(source_b), total_rep=2)
            with patch.dict(os.environ, {"FITMOTION_DATA_DIR": data_dir}, clear=False):
                _, first_path, _ = prepare_persistent_database(source_a)
                _, second_path, second_note = prepare_persistent_database(source_b)
            self.assertEqual(first_path, second_path)
            self.assertEqual("", second_note)
            connection = sqlite3.connect(second_path)
            try:
                self.assertEqual(
                    11,
                    connection.execute("SELECT total_rep FROM workout_sessions WHERE id=1").fetchone()[0],
                )
            finally:
                connection.close()

    def test_draft_table_is_created_without_flask(self):
        with tempfile.TemporaryDirectory() as source_dir, tempfile.TemporaryDirectory() as data_dir:
            self._legacy_database(Path(source_dir))
            with patch.dict(os.environ, {"FITMOTION_DATA_DIR": data_dir}, clear=False):
                _, database_path, _ = prepare_persistent_database(source_dir)
            connection = sqlite3.connect(database_path)
            try:
                table = connection.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name='workout_session_drafts'"
                ).fetchone()
            finally:
                connection.close()
            self.assertIsNotNone(table)

    def test_backup_contains_committed_sessions(self):
        with tempfile.TemporaryDirectory() as source_dir, tempfile.TemporaryDirectory() as data_dir:
            self._legacy_database(Path(source_dir), total_rep=23)
            with patch.dict(os.environ, {"FITMOTION_DATA_DIR": data_dir}, clear=False):
                _, database_path, _ = prepare_persistent_database(source_dir)
            backup_path = backup_database(database_path)
            self.assertTrue(backup_path and backup_path.exists())
            connection = sqlite3.connect(backup_path)
            try:
                self.assertEqual(
                    23,
                    connection.execute("SELECT total_rep FROM workout_sessions WHERE id=1").fetchone()[0],
                )
            finally:
                connection.close()

    def test_base_processor_restores_completed_rep_totals(self):
        shared = {
            "total_rep": 6,
            "good_rep": 4,
            "rep_scores": [80, 70, 60, 90, 75, 65],
            "rep_records": [{"confidence": 0.88}],
            "error_code_counts": {"notlow": 2},
            "rep_quality_score": 65,
            "feedback_id": 7,
        }
        processor = BaseProcessor(shared_state=shared, load_model=False)
        self.assertEqual(6, processor.tongsolan)
        self.assertEqual(4, processor.solandung)
        self.assertEqual(6, len(processor.rep_scores))
        self.assertEqual(2, processor.error_code_counts["notlow"])

    def test_squat_processor_restores_calibration_and_rep_count(self):
        shared = {
            "total_rep": 3,
            "good_rep": 2,
            "squat_calibration_ready": True,
            "squat_standing_baseline": 151.0,
        }
        processor = SquatProcessor(shared_state=shared, model=object())
        self.assertEqual(3, processor.tongsolan)
        self.assertEqual(2, processor.solandung)
        self.assertEqual("STANDING", processor.trangthai)
        self.assertAlmostEqual(151.0, processor.baseline_standing_angle)

    def test_app_uses_server_date_and_atomic_draft_finalization(self):
        source = Path("app.py").read_text(encoding="utf-8")
        finish = source[source.index("def finish_workout_api"):source.index("def emergency_status_api")]
        self.assertIn('session_date = datetime.now().date().isoformat()', finish)
        self.assertNotIn('data.get("session_date")', finish)
        self.assertIn("delete_workout_draft", finish)
        self.assertIn("commit=False", finish)
        self.assertIn("backup_database(DB_PATH)", finish)


if __name__ == "__main__":
    unittest.main()
