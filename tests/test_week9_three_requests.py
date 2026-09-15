from __future__ import annotations

import os
import sqlite3
import tempfile
import unittest
from pathlib import Path

import numpy as np

from emergency import EmergencyMonitor
from storage import ensure_storage_schema

ROOT = Path(__file__).resolve().parents[1]


class Week9ProgramDatabaseTests(unittest.TestCase):
    def test_storage_creates_program_community_tables_and_schedule_link(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "fitmotion.db"
            ensure_storage_schema(db_path)
            conn = sqlite3.connect(db_path)
            try:
                tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
                for name in {
                    "workout_programs",
                    "workout_program_items",
                    "workout_program_favorites",
                    "workout_program_reviews",
                    "workout_schedules",
                }:
                    self.assertIn(name, tables)
                schedule_columns = {row[1] for row in conn.execute("PRAGMA table_info(workout_schedules)")}
                self.assertIn("source_program_id", schedule_columns)
            finally:
                conn.close()

    def test_program_routes_cover_expert_share_copy_rating_and_apply(self):
        source = (ROOT / "app.py").read_text(encoding="utf-8")
        self.assertIn('def build_expert_program_blueprint(profile):', source)
        self.assertIn('def user_save_expert_program():', source)
        self.assertIn('def user_toggle_program_visibility(program_id):', source)
        self.assertIn('def user_copy_program(program_id):', source)
        self.assertIn('def user_review_program(program_id):', source)
        self.assertIn('def user_apply_program(program_id):', source)
        self.assertIn('source_program_id=program.id', source)
        self.assertIn('"bmi": round(bmi, 1)', source)
        self.assertIn('health = snapshot["health_note"]', source)

    def test_program_page_exposes_expert_own_and_community_flows(self):
        page = (ROOT / "templates" / "user_programs.html").read_text(encoding="utf-8")
        self.assertIn("Hệ chuyên gia FitMotion", page)
        self.assertIn("Tạo giáo án của tôi", page)
        self.assertIn("Giáo án được chia sẻ", page)
        self.assertIn("Sao chép", page)
        self.assertIn("Gửi đánh giá", page)
        self.assertIn("Áp dụng vào lịch", page)


class Week9DifficultySelectionTests(unittest.TestCase):
    def test_each_exercise_card_requires_level_selection(self):
        page = (ROOT / "templates" / "user_exercises.html").read_text(encoding="utf-8")
        self.assertIn("level-mini-btn", page)
        self.assertIn("data-level=\"{{ level_key }}\"", page)
        self.assertIn("workout-start-disabled", page)
        self.assertIn("?level=${encodeURIComponent(level)}", page)

    def test_workout_route_honors_explicit_level_before_reps(self):
        source = (ROOT / "app.py").read_text(encoding="utf-8")
        self.assertIn('requested_level_raw = str(request.args.get("level", ""))', source)
        self.assertIn('if requested_level and int(shared_state.get("total_rep", 0) or 0) <= 0:', source)
        self.assertIn("Hãy chọn bài tập và cấp độ Dễ / Trung bình / Khó", source)


class Week9EmergencyEvidenceTests(unittest.TestCase):
    def test_emergency_frame_is_saved_to_configured_persistent_folder(self):
        with tempfile.TemporaryDirectory() as tmp:
            previous = os.environ.get("FITMOTION_EMERGENCY_DIR")
            os.environ["FITMOTION_EMERGENCY_DIR"] = tmp
            try:
                state = {}
                monitor = EmergencyMonitor(state)
                frame = np.zeros((120, 160, 3), dtype=np.uint8)
                monitor.trigger("test", raw_frame=frame)
                saved = Path(state["image_path"])
                self.assertTrue(saved.exists())
                self.assertEqual(saved.parent.resolve(), Path(tmp).resolve())
            finally:
                if previous is None:
                    os.environ.pop("FITMOTION_EMERGENCY_DIR", None)
                else:
                    os.environ["FITMOTION_EMERGENCY_DIR"] = previous

    def test_admin_alert_shows_one_image_and_complete_registration_contacts(self):
        page = (ROOT / "templates" / "admin_emergency_alerts.html").read_text(encoding="utf-8")
        self.assertEqual(page.count("admin_emergency_evidence"), 1)
        self.assertIn("Gmail / Email đăng ký", page)
        self.assertIn("SĐT tài khoản", page)
        self.assertIn("SĐT người thân", page)
        self.assertIn("alert.personal_phone_snapshot", page)
        self.assertIn("alert.emergency_phone_snapshot", page)
        self.assertNotIn("<strong>Bằng chứng:</strong> {{ alert.evidence_path }}", page)

    def test_admin_evidence_route_accepts_persistent_emergency_root(self):
        source = (ROOT / "app.py").read_text(encoding="utf-8")
        self.assertIn('def admin_emergency_evidence(alert_id):', source)
        self.assertIn('PERSISTENT_EMERGENCY_DIR', source)
        self.assertIn('(Path(DATA_DIR) / "errorimages").resolve()', source)


if __name__ == "__main__":
    unittest.main()
