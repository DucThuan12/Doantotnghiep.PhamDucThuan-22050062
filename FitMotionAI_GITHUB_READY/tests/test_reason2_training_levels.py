from __future__ import annotations

import sqlite3
import tempfile
import unittest
import sys
import types
from pathlib import Path

try:
    import ultralytics  # noqa: F401
except ModuleNotFoundError:
    sys.modules["ultralytics"] = types.SimpleNamespace(YOLO=lambda *args, **kwargs: None)

from quality_scoring import score_curl, score_pushup, score_squat
from storage import ensure_storage_schema
from training_levels import (
    LEVEL_EASY,
    LEVEL_HARD,
    LEVEL_MEDIUM,
    normalize_training_level,
    scoring_profile,
    threshold_profile,
    training_level_label,
)

ROOT = Path(__file__).resolve().parents[1]


class TrainingLevelProfileTests(unittest.TestCase):
    def test_level_aliases_are_normalized_to_three_canonical_values(self):
        self.assertEqual(normalize_training_level("Dễ"), LEVEL_EASY)
        self.assertEqual(normalize_training_level("Trung bình"), LEVEL_MEDIUM)
        self.assertEqual(normalize_training_level("Nâng cao"), LEVEL_HARD)
        self.assertEqual(normalize_training_level("unexpected"), LEVEL_MEDIUM)
        self.assertEqual(training_level_label(LEVEL_HARD), "Khó")

    def test_each_core_exercise_has_progressive_easy_medium_hard_thresholds(self):
        squat_easy = threshold_profile("squat", LEVEL_EASY)
        squat_medium = threshold_profile("squat", LEVEL_MEDIUM)
        squat_hard = threshold_profile("squat", LEVEL_HARD)
        self.assertLess(squat_easy["minimum_cycle_rom"], squat_medium["minimum_cycle_rom"])
        self.assertLess(squat_medium["minimum_cycle_rom"], squat_hard["minimum_cycle_rom"])
        self.assertLess(squat_easy["good_score_threshold"], squat_hard["good_score_threshold"])

        push_easy = threshold_profile("pushup", LEVEL_EASY)
        push_hard = threshold_profile("pushup", LEVEL_HARD)
        self.assertLess(push_easy["minimum_cycle_rom"], push_hard["minimum_cycle_rom"])
        self.assertLess(push_hard["shallow_limit"], push_easy["shallow_limit"])
        self.assertGreater(push_hard["body_line_limit"], push_easy["body_line_limit"])

        curl_easy = threshold_profile("curl-left", LEVEL_EASY)
        curl_hard = threshold_profile("curl-right", LEVEL_HARD)
        self.assertLess(curl_easy["minimum_range_of_motion"], curl_hard["minimum_range_of_motion"])
        self.assertGreater(curl_easy["elbow_shift_limit"], curl_hard["elbow_shift_limit"])
        self.assertGreater(curl_easy["full_flexion_ceiling"], curl_hard["full_flexion_ceiling"])

    def test_scoring_profile_weights_change_by_level(self):
        easy = scoring_profile("squat", LEVEL_EASY)
        medium = scoring_profile("squat", LEVEL_MEDIUM)
        hard = scoring_profile("squat", LEVEL_HARD)
        self.assertNotEqual(easy["weights"], medium["weights"])
        self.assertNotEqual(medium["weights"], hard["weights"])
        self.assertAlmostEqual(sum(easy["weights"].values()), 1.0)
        self.assertAlmostEqual(sum(hard["weights"].values()), 1.0)

    def test_same_borderline_motion_scores_lower_at_harder_level(self):
        easy_squat = score_squat(120, 140, 0.90, 82, level=LEVEL_EASY)
        hard_squat = score_squat(120, 140, 0.90, 82, level=LEVEL_HARD)
        self.assertGreater(easy_squat.total, hard_squat.total)

        easy_pushup = score_pushup(112, 150, 0.90, 82, level=LEVEL_EASY)
        hard_pushup = score_pushup(112, 150, 0.90, 82, level=LEVEL_HARD)
        self.assertGreater(easy_pushup.total, hard_pushup.total)

        easy_curl = score_curl(82, 148, 25, 0.90, 82, extension_target=150, level=LEVEL_EASY)
        hard_curl = score_curl(82, 148, 25, 0.90, 82, extension_target=150, level=LEVEL_HARD)
        self.assertGreater(easy_curl.total, hard_curl.total)


class ProcessorLevelBindingTests(unittest.TestCase):
    def test_processors_load_thresholds_from_shared_state_level(self):
        import workoutlogic as wl

        dummy_model = object()
        easy_squat = wl.SquatProcessor(shared_state={"training_level": "easy"}, model=dummy_model)
        hard_squat = wl.SquatProcessor(shared_state={"training_level": "hard"}, model=dummy_model)
        self.assertLess(easy_squat.thresholds.minimum_cycle_rom, hard_squat.thresholds.minimum_cycle_rom)
        self.assertLess(easy_squat.thresholds.good_score_threshold, hard_squat.thresholds.good_score_threshold)

        easy_push = wl.PushupProcessor(shared_state={"training_level": "easy"}, model=dummy_model)
        hard_push = wl.PushupProcessor(shared_state={"training_level": "hard"}, model=dummy_model)
        self.assertLess(easy_push.thresholds.minimum_cycle_rom, hard_push.thresholds.minimum_cycle_rom)

        easy_curl = wl.CurlProcessor("left", shared_state={"training_level": "easy"}, model=dummy_model)
        hard_curl = wl.CurlProcessor("right", shared_state={"training_level": "hard"}, model=dummy_model)
        self.assertLess(easy_curl.thresholds.minimum_range_of_motion, hard_curl.thresholds.minimum_range_of_motion)
        self.assertGreater(easy_curl.thresholds.elbow_shift_limit, hard_curl.thresholds.elbow_shift_limit)

    def test_processor_can_switch_level_before_first_rep_but_not_after(self):
        import workoutlogic as wl

        state = {"training_level": "easy"}
        processor = wl.SquatProcessor(shared_state=state, model=object())
        easy_rom = processor.thresholds.minimum_cycle_rom
        state["training_level"] = "hard"
        self.assertTrue(processor.sync_training_level())
        self.assertGreater(processor.thresholds.minimum_cycle_rom, easy_rom)
        self.assertEqual(processor.training_level, "hard")

        processor.tongsolan = 1
        state["training_level"] = "easy"
        self.assertFalse(processor.sync_training_level())
        self.assertEqual(processor.training_level, "hard")
        self.assertEqual(state["training_level"], "hard")


class TrainingLevelStorageTests(unittest.TestCase):
    def test_storage_schema_keeps_level_on_schedule_item_and_draft(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "aifitness.db"
            ensure_storage_schema(db_path)
            conn = sqlite3.connect(db_path)
            try:
                draft_cols = {row[1] for row in conn.execute("PRAGMA table_info(workout_session_drafts)")}
                item_cols = {row[1] for row in conn.execute("PRAGMA table_info(workout_schedule_items)")}
            finally:
                conn.close()
        self.assertIn("training_level", draft_cols)
        self.assertIn("training_level", item_cols)

    def test_storage_migrates_existing_plan_and_session_tables_to_medium(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "aifitness.db"
            conn = sqlite3.connect(db_path)
            try:
                conn.execute("CREATE TABLE workout_plans (id INTEGER PRIMARY KEY, schedule_id INTEGER)")
                conn.execute("CREATE TABLE workout_sessions (id INTEGER PRIMARY KEY, schedule_id INTEGER)")
                conn.commit()
            finally:
                conn.close()
            ensure_storage_schema(db_path)
            conn = sqlite3.connect(db_path)
            try:
                plan_cols = {row[1] for row in conn.execute("PRAGMA table_info(workout_plans)")}
                session_cols = {row[1] for row in conn.execute("PRAGMA table_info(workout_sessions)")}
            finally:
                conn.close()
        self.assertIn("training_level", plan_cols)
        self.assertIn("training_level", session_cols)


class TrainingLevelIntegrationSourceTests(unittest.TestCase):
    def test_models_persist_level_for_plan_schedule_session_and_draft(self):
        source = (ROOT / "models.py").read_text(encoding="utf-8")
        self.assertGreaterEqual(source.count('training_level = db.Column(db.String(20)'), 4)

    def test_schedule_and_workout_forms_expose_level(self):
        schedule = (ROOT / "templates" / "user_schedule.html").read_text(encoding="utf-8")
        workout = (ROOT / "templates" / "user_workout.html").read_text(encoding="utf-8")
        self.assertIn('name="training_level[]"', schedule)
        self.assertIn('id="trainingLevelInput"', workout)
        self.assertIn("Cấp độ AI", workout)

    def test_level_flows_to_runtime_and_is_not_changeable_mid_session(self):
        app_source = (ROOT / "app.py").read_text(encoding="utf-8")
        logic_source = (ROOT / "workoutlogic.py").read_text(encoding="utf-8")
        self.assertIn('shared_state["training_level"] = training_level', app_source)
        self.assertIn('"level_changed": training_level != previous_level', app_source)
        self.assertIn("không thể đổi cấp độ giữa chừng".lower(), app_source.lower())
        self.assertIn("sync_training_level", logic_source)
        self.assertIn('threshold_profile("squat", self.training_level)', logic_source)
        self.assertIn('threshold_profile("pushup", self.training_level)', logic_source)
        self.assertIn('threshold_profile("curl", self.training_level)', logic_source)

    def test_rep_evidence_and_history_include_level(self):
        logic_source = (ROOT / "workoutlogic.py").read_text(encoding="utf-8")
        app_source = (ROOT / "app.py").read_text(encoding="utf-8")
        self.assertIn('"training_level": self.training_level', logic_source)
        self.assertIn('training_level=normalize_training_level(training_level)', app_source)
        self.assertIn('"training_level_label": training_level_label', app_source)

    def test_video_evaluation_uses_same_level_aware_pipeline(self):
        source = (ROOT / "video_processor.py").read_text(encoding="utf-8")
        self.assertIn('state: dict = {"training_level": normalized_level}', source)
        self.assertIn("process_video_file(temp_path, exercise, training_level=training_level)", source)


if __name__ == "__main__":
    unittest.main()
