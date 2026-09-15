from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from dashboard_activity import build_schedule_adherence, build_week_activity
from emergency import EmergencyMonitor


class _Clock:
    def __init__(self, start=1000.0):
        self.now = float(start)

    def time(self):
        return self.now

    def advance(self, seconds):
        self.now += float(seconds)


def _pose(shoulder_y, hip_y, ankle_y, x0=300, spread=40):
    kp = np.zeros((17, 2), dtype=float)
    kp[5] = (x0 - spread, shoulder_y)
    kp[6] = (x0 + spread, shoulder_y)
    kp[11] = (x0 - spread, hip_y)
    kp[12] = (x0 + spread, hip_y)
    kp[15] = (x0 - spread, ankle_y)
    kp[16] = (x0 + spread, ankle_y)
    return kp, np.ones(17, dtype=float)


def _lying(y=365, jitter_x=0):
    kp = np.zeros((17, 2), dtype=float)
    kp[5] = (120 + jitter_x, y - 15)
    kp[6] = (120 + jitter_x, y + 25)
    kp[11] = (310 + jitter_x, y - 5)
    kp[12] = (310 + jitter_x, y + 35)
    kp[15] = (520 + jitter_x, y + 5)
    kp[16] = (520 + jitter_x, y + 45)
    return kp, np.ones(17, dtype=float)


class EmergencyRound4Tests(unittest.TestCase):
    def _update(self, monitor, clock, pose, advance=0.25, context=None):
        clock.advance(advance)
        kp, conf = pose
        return monitor.update(kp, conf, (480, 640, 3), context=context or {})

    def test_default_alarm_window_is_six_seconds_and_clamped_four_to_ten(self):
        self.assertEqual(EmergencyMonitor({}, immobility_seconds=6).immobility_seconds, 6.0)
        self.assertEqual(EmergencyMonitor({}, immobility_seconds=1).immobility_seconds, 4.0)
        self.assertEqual(EmergencyMonitor({}, immobility_seconds=20).immobility_seconds, 10.0)

    def test_gradual_lie_down_then_six_seconds_immobile_triggers(self):
        clock = _Clock()
        state = {}
        standing = _pose(90, 220, 430)
        # Tilted low frame makes the final transition to lying small enough to
        # miss the old rapid-drop gate.
        almost_lying = _lying(y=330)
        fallen = _lying(y=350)
        with patch("emergency.time.time", side_effect=clock.time):
            monitor = EmergencyMonitor(state, immobility_seconds=6)
            for _ in range(5):
                self._update(monitor, clock, standing)
            self._update(monitor, clock, almost_lying, advance=0.5)
            self._update(monitor, clock, fallen, advance=0.5)
            self.assertTrue(state["emergency_candidate"])
            for index in range(7):
                # One-pixel pose jitter must not cancel verification.
                self._update(monitor, clock, _lying(y=350, jitter_x=index % 2), advance=1.0)
        self.assertTrue(state["active"])
        self.assertIn("6 giây", state["reason"])


class DashboardActivityTests(unittest.TestCase):
    def test_actual_session_appears_without_preexisting_plan(self):
        result = build_week_activity(
            week_dates=["2026-08-04"],
            plans=[],
            sessions=[SimpleNamespace(exercise_id=7, session_date="2026-08-04", total_rep=10)],
            drafts=[],
            live_items=[],
            exercise_calories={7: 0.5},
        )
        self.assertEqual(result["2026-08-04"]["count"], 1)
        self.assertEqual(result["2026-08-04"]["completed_count"], 1)
        self.assertEqual(result["2026-08-04"]["kcal"], 5.0)

    def test_plan_and_session_same_exercise_are_not_double_counted(self):
        result = build_week_activity(
            week_dates=["2026-08-04"],
            plans=[SimpleNamespace(exercise_id=7, workout_date="2026-08-04", rep_target=20)],
            sessions=[SimpleNamespace(exercise_id=7, session_date="2026-08-04", total_rep=8)],
            drafts=[],
            live_items=[],
            exercise_calories={7: 0.5},
        )
        self.assertEqual(result["2026-08-04"]["count"], 1)
        # Actual calories replace planned calories once the exercise was done.
        self.assertEqual(result["2026-08-04"]["kcal"], 4.0)


    def test_registered_schedule_adherence_uses_sessions_and_ignores_manual_plan(self):
        result = build_schedule_adherence(
            plans=[
                SimpleNamespace(schedule_id=9, exercise_id=7, workout_date="2026-08-04"),
                SimpleNamespace(schedule_id=9, exercise_id=8, workout_date="2026-08-06"),
                SimpleNamespace(schedule_id=None, exercise_id=10, workout_date="2026-08-05"),
            ],
            sessions=[SimpleNamespace(exercise_id=7, session_date="2026-08-04", total_rep=10)],
            drafts=[],
            live_items=[],
            today="2026-08-05",
        )
        self.assertEqual(result["planned_count"], 2)
        self.assertEqual(result["performed_count"], 1)
        self.assertEqual(result["missed_count"], 0)
        self.assertEqual(result["upcoming_count"], 1)
        self.assertEqual(result["completion_percent"], 50.0)

    def test_registered_schedule_draft_or_live_rep_counts_as_performed(self):
        result = build_schedule_adherence(
            plans=[SimpleNamespace(schedule_id=4, exercise_id=7, workout_date="2026-08-04")],
            sessions=[],
            drafts=[SimpleNamespace(exercise_id=7, session_date="2026-08-04", total_rep=3)],
            live_items=[],
            today="2026-08-04",
        )
        self.assertEqual(result["performed_count"], 1)
        self.assertEqual(result["completion_percent"], 100.0)

    def test_dashboard_api_and_template_refresh_week_cards(self):
        from pathlib import Path
        root = Path(__file__).resolve().parents[1]
        app_source = (root / "app.py").read_text(encoding="utf-8")
        page = (root / "templates" / "user_dashboard.html").read_text(encoding="utf-8")
        self.assertIn('"week_schedule": week_schedule', app_source)
        self.assertIn('"day_detail": day_detail', app_source)
        self.assertIn('id="weekCount-{{ day.date }}"', page)
        self.assertIn("renderDayPlanDetail", page)
        self.assertIn('cache: "no-store"', page)


if __name__ == "__main__":
    unittest.main()
