from __future__ import annotations

import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

try:
    import ultralytics  # noqa: F401
except ModuleNotFoundError:
    sys.modules["ultralytics"] = types.SimpleNamespace(YOLO=lambda *args, **kwargs: None)

import workoutlogic as wl
from emergency import EmergencyMonitor

ROOT = Path(__file__).resolve().parents[1]


class _FakeResult:
    def __init__(self, frame):
        self.frame = frame

    def plot(self):
        return self.frame.copy()


class _SequenceModel:
    def __init__(self, sequence):
        self.sequence = list(sequence)
        self.index = 0
        self.current = self.sequence[0]
        self.angle_call = 0

    def __call__(self, frame, verbose=False):
        self.current = self.sequence[min(self.index, len(self.sequence) - 1)]
        self.index += 1
        self.angle_call = 0
        return [_FakeResult(frame)]


def _item(angles, confidence=1.0):
    return {"angles": angles, "confidence": confidence}


class ExerciseUsabilityTests(unittest.TestCase):
    def _run(self, processor_cls, sequence, *args):
        model = _SequenceModel(sequence)
        original = (wl.coNguoi, wl.layDiem, wl.tinhgoc, wl.save_bad_rep)
        points = np.zeros((17, 2), dtype=float)
        for index in range(17):
            points[index] = [index * 12.0, index * 8.0]
        wl.coNguoi = lambda results: True
        wl.layDiem = lambda _results: (
            points.copy(),
            np.full(17, float(model.current.get("confidence", 1.0)), dtype=float),
        )

        def fake_angle(*_args):
            angles = model.current["angles"]
            index = model.angle_call
            model.angle_call += 1
            if isinstance(angles, (tuple, list)):
                return float(angles[index])
            return float(angles)

        wl.tinhgoc = fake_angle
        wl.save_bad_rep = lambda *a, **k: 0
        try:
            if processor_cls is wl.CurlProcessor:
                processor = processor_cls(*args, shared_state={}, model=model)
            else:
                processor = processor_cls(shared_state={}, model=model)
            processor.update_emergency = lambda *a, **k: None
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            for _ in sequence:
                processor.process(frame)
            return processor
        finally:
            wl.coNguoi, wl.layDiem, wl.tinhgoc, wl.save_bad_rep = original

    def test_curl_near_180_measurement_is_capped_to_ergonomic_target(self):
        sequence = [_item(179)] * 10 + [_item(70)] * 18 + [_item(150)] * 24
        processor = self._run(wl.CurlProcessor, sequence, "left")
        self.assertEqual(processor.tongsolan, 1)
        self.assertEqual(processor.solandung, 1)
        self.assertLessEqual(processor.shared_state["curl_extension_baseline"], 165.0)
        self.assertNotIn("notstraight", processor.rep_records[0]["error_codes"])

    def test_restored_curl_baseline_from_old_version_is_clamped(self):
        state = {
            "curl_extension_baseline": 179.0,
            "curl_calibration_ready": True,
        }
        processor = wl.CurlProcessor("left", shared_state=state, model=object())
        self.assertEqual(processor.baseline_extension_angle, 165.0)
        self.assertEqual(state["curl_extension_baseline"], 165.0)

    def test_pushup_moderate_camera_angles_count_complete_rep(self):
        sequence = [_item((150, 170))] * 8 + [_item((108, 168))] * 16 + [_item((140, 170))] * 22
        processor = self._run(wl.PushupProcessor, sequence)
        self.assertEqual(processor.tongsolan, 1)
        self.assertEqual(processor.solandung, 1)

    def test_squat_parallelish_depth_can_be_good(self):
        sequence = [_item((160, 160))] * 8 + [_item((128, 150))] * 18 + [_item((150, 160))] * 22
        processor = self._run(wl.SquatProcessor, sequence)
        self.assertEqual(processor.tongsolan, 1)
        self.assertEqual(processor.solandung, 1)
        self.assertGreaterEqual(processor.last_rep_score, 60)


class SafetyAndInterfaceTests(unittest.TestCase):
    def test_emergency_trigger_pauses_workout_immediately(self):
        state = {}
        monitor = EmergencyMonitor(state, immobility_seconds=12)
        with patch("emergency.time.time", return_value=1000.0):
            monitor.trigger("test")
        self.assertTrue(state["active"])
        self.assertTrue(state["workout_paused"])
        self.assertEqual(state["pause_reason"], "emergency")

    def test_one_minute_wellness_check_is_server_authoritative(self):
        source = (ROOT / "app.py").read_text(encoding="utf-8")
        page = (ROOT / "templates" / "user_workout.html").read_text(encoding="utf-8")
        self.assertIn("def update_wellness_check", source)
        self.assertIn('"wellness_check_seconds": 60', source)
        self.assertIn("def acknowledge_wellness_check_api", source)
        self.assertIn("wellnessOverlay", page)
        self.assertIn("Bạn đã tập được khoảng một phút", page)


    def test_live_api_exposes_emergency_active_state_and_ack_is_guarded(self):
        source = (ROOT / "app.py").read_text(encoding="utf-8")
        self.assertIn('"active": bool(shared_state.get("active", False))', source)
        ack = source[source.index("def acknowledge_wellness_check_api"):source.index("def finish_workout_api")]
        self.assertIn('shared_state.get("active", False)', ack)
        self.assertIn('409', ack)

    def test_workout_page_hides_raw_database_and_model_paths(self):
        page = (ROOT / "templates" / "user_workout.html").read_text(encoding="utf-8")
        self.assertNotIn("Mô hình đã gắn:", page)
        self.assertNotIn("Tự lưu SQLite:", page)
        self.assertIn("Xem thông số kỹ thuật", page)


if __name__ == "__main__":
    unittest.main()
