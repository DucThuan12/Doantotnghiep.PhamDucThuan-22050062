from __future__ import annotations

import sqlite3
import tempfile
import unittest
from pathlib import Path

from storage import ensure_storage_schema
from training_levels import (
    LEVEL_EASY,
    LEVEL_HARD,
    LEVEL_MEDIUM,
    evaluate_session_effectiveness,
    workload_profile,
)

ROOT = Path(__file__).resolve().parents[1]


def rep_records(count: int, stability: float = 80.0) -> list[dict]:
    return [{"index": i + 1, "stability": stability} for i in range(count)]


class MultiCriteriaLevelProfileTests(unittest.TestCase):
    def test_each_core_exercise_has_progressive_workload(self):
        for exercise in ("squat", "pushup", "curl-left", "curl-right"):
            easy = workload_profile(exercise, LEVEL_EASY)
            medium = workload_profile(exercise, LEVEL_MEDIUM)
            hard = workload_profile(exercise, LEVEL_HARD)
            self.assertLess(easy["default_sets"] * easy["reps_per_set"], hard["default_sets"] * hard["reps_per_set"])
            self.assertLessEqual(easy["min_good_rep_ratio"], medium["min_good_rep_ratio"])
            self.assertLessEqual(medium["min_good_rep_ratio"], hard["min_good_rep_ratio"])
            self.assertLessEqual(easy["min_quality_avg"], hard["min_quality_avg"])
            self.assertLessEqual(easy["min_confidence_avg"], hard["min_confidence_avg"])
            self.assertLessEqual(easy["min_stability_avg"], hard["min_stability_avg"])

    def test_pushup_uses_clear_5_10_15_rep_progression(self):
        self.assertEqual(workload_profile("pushup", LEVEL_EASY)["reps_per_set"], 5)
        self.assertEqual(workload_profile("pushup", LEVEL_MEDIUM)["reps_per_set"], 10)
        self.assertEqual(workload_profile("pushup", LEVEL_HARD)["reps_per_set"], 15)

    def test_long_break_threshold_is_configured_and_well_below_half_hour(self):
        for exercise in ("squat", "pushup", "curl-left"):
            for level in (LEVEL_EASY, LEVEL_MEDIUM, LEVEL_HARD):
                self.assertLess(workload_profile(exercise, level)["max_idle_seconds"], 30 * 60)


class SessionEffectivenessTests(unittest.TestCase):
    def setUp(self):
        self.criteria = workload_profile("pushup", LEVEL_MEDIUM)

    def test_volume_alone_does_not_pass_level(self):
        result = evaluate_session_effectiveness(
            total_rep=30,
            good_rep=10,
            quality_score_avg=40,
            confidence_avg=0.90,
            rep_records=rep_records(30, 80),
            tracking_abort_count=0,
            elapsed_seconds=600,
            set_count=3,
            reps_per_set=10,
            criteria=self.criteria,
        )
        self.assertTrue(result["volume_met"])
        self.assertFalse(result["good_ratio_met"])
        self.assertFalse(result["quality_met"])
        self.assertFalse(result["effective"])

    def test_good_technique_without_enough_volume_is_partial(self):
        result = evaluate_session_effectiveness(
            total_rep=12,
            good_rep=12,
            quality_score_avg=90,
            confidence_avg=0.90,
            rep_records=rep_records(12, 90),
            tracking_abort_count=0,
            elapsed_seconds=300,
            set_count=3,
            reps_per_set=10,
            criteria=self.criteria,
        )
        self.assertFalse(result["volume_met"])
        self.assertTrue(result["quality_met"])
        self.assertFalse(result["effective"])

    def test_tracking_stability_and_duration_are_real_gates(self):
        result = evaluate_session_effectiveness(
            total_rep=30,
            good_rep=30,
            quality_score_avg=90,
            confidence_avg=0.90,
            rep_records=rep_records(30, 25),
            tracking_abort_count=99,
            elapsed_seconds=self.criteria["max_session_seconds"] + 1,
            set_count=3,
            reps_per_set=10,
            criteria=self.criteria,
        )
        self.assertFalse(result["stability_met"])
        self.assertFalse(result["tracking_met"])
        self.assertFalse(result["duration_met"])
        self.assertFalse(result["effective"])

    def test_full_multicriteria_session_can_pass(self):
        result = evaluate_session_effectiveness(
            total_rep=30,
            good_rep=25,
            quality_score_avg=82,
            confidence_avg=0.88,
            rep_records=rep_records(30, 82),
            tracking_abort_count=0,
            elapsed_seconds=900,
            set_count=3,
            reps_per_set=10,
            criteria=self.criteria,
        )
        self.assertTrue(result["effective"])
        self.assertEqual(result["target_total_rep"], 30)


class MultiCriteriaStorageTests(unittest.TestCase):
    def test_schema_has_admin_level_config_and_session_evidence(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "aifitness.db"
            conn = sqlite3.connect(db_path)
            try:
                conn.execute("CREATE TABLE workout_sessions (id INTEGER PRIMARY KEY, schedule_id INTEGER)")
                conn.commit()
            finally:
                conn.close()
            ensure_storage_schema(db_path)
            conn = sqlite3.connect(db_path)
            try:
                config_cols = {row[1] for row in conn.execute("PRAGMA table_info(exercise_level_configs)")}
                session_cols = {row[1] for row in conn.execute("PRAGMA table_info(workout_sessions)")}
            finally:
                conn.close()
        self.assertTrue({
            "training_level", "set_count", "rep_target", "rest_seconds",
            "max_idle_seconds", "max_session_seconds", "min_good_rep_ratio",
            "min_quality_score", "min_confidence", "min_stability_score",
            "max_tracking_abort_count",
        }.issubset(config_cols))
        self.assertTrue({
            "set_count", "rep_target", "duration_seconds",
            "effectiveness_status", "session_summary_json",
        }.issubset(session_cols))


class MultiCriteriaIntegrationSourceTests(unittest.TestCase):
    def test_admin_can_configure_many_level_criteria(self):
        template = (ROOT / "templates" / "admin_exercise_detail.html").read_text(encoding="utf-8")
        app = (ROOT / "app.py").read_text(encoding="utf-8")
        for token in (
            "set_count", "rep_target", "rest_seconds", "max_idle_seconds",
            "max_session_seconds", "min_good_rep_percent", "min_quality_score",
            "min_confidence_percent", "min_stability_score", "max_tracking_abort_count",
        ):
            self.assertIn(token, template)
        self.assertIn("def admin_update_level_config", app)

    def test_schedule_level_change_applies_level_set_rep_defaults(self):
        template = (ROOT / "templates" / "user_schedule.html").read_text(encoding="utf-8")
        self.assertIn("const levelDefaults", template)
        self.assertIn("applyLevelDefaults", template)
        self.assertIn("defaults.set_count", template)
        self.assertIn("defaults.rep_target", template)

    def test_long_inactivity_cancels_draft_and_never_finalizes_from_that_path(self):
        app = (ROOT / "app.py").read_text(encoding="utf-8")
        start = app.index("def cancel_inactive_workout_if_needed")
        end = app.index("\ndef normalize_goal", start)
        func = app[start:end]
        self.assertIn("delete_workout_draft", func)
        self.assertIn('"workout_cancelled": True', func)
        self.assertIn('"inactivity_cancelled"', func)
        self.assertNotIn("save_workout_session_result", func)
        self.assertIn("if cancel_inactive_workout_if_needed", app[app.index("def finish_workout_api"):])

    def test_cancelled_workout_ui_requires_fresh_restart(self):
        template = (ROOT / "templates" / "user_workout.html").read_text(encoding="utf-8")
        self.assertIn("cancelledWorkoutOverlay", template)
        self.assertIn("Bắt đầu lại từ 0 rep", template)
        self.assertIn("restartCancelled", template)
        self.assertIn("seconds_until_cancel", template)

    def test_finished_session_is_evaluated_by_whole_session_criteria(self):
        app = (ROOT / "app.py").read_text(encoding="utf-8")
        finish = app[app.index("def finish_workout_api"):]
        self.assertIn("evaluate_session_effectiveness", finish)
        self.assertIn('effectiveness_status="effective" if effectiveness["effective"] else "partial"', finish)
        self.assertIn('status = "completed" if effectiveness["effective"] else "partial"', finish)

    def test_rep_completion_updates_inactivity_anchor(self):
        logic = (ROOT / "workoutlogic.py").read_text(encoding="utf-8")
        self.assertIn('self.shared_state["last_rep_at"] = time.time()', logic)
        self.assertIn('self.shared_state["idle_seconds"] = 0', logic)


if __name__ == "__main__":
    unittest.main()
