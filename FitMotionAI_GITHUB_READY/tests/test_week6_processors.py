from __future__ import annotations

import sys
import types
import unittest

import numpy as np

# The validation environment does not need the real model weights.
try:
    import ultralytics  # noqa: F401
except ModuleNotFoundError:
    sys.modules["ultralytics"] = types.SimpleNamespace(YOLO=lambda *args, **kwargs: None)

import workoutlogic as wl


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


def _item(angles, confidence=1.0, left_confidence=None, right_confidence=None, elbow_shift=0.0):
    return {
        "angles": angles,
        "confidence": confidence,
        "left_confidence": left_confidence,
        "right_confidence": right_confidence,
        "elbow_shift": elbow_shift,
    }


class ProcessorStateMachineTests(unittest.TestCase):
    def _run(self, processor_cls, sequence, *args):
        model = _SequenceModel(sequence)
        original = (wl.coNguoi, wl.layDiem, wl.tinhgoc, wl.save_bad_rep)

        base_keypoints = np.zeros((17, 2), dtype=float)
        for index in range(17):
            base_keypoints[index] = [index * 12.0, index * 8.0]

        wl.coNguoi = lambda results: True

        def fake_points(_results):
            item = model.current
            keypoints = base_keypoints.copy()
            confidence = np.full(17, float(item.get("confidence", 1.0)), dtype=float)
            if item.get("left_confidence") is not None:
                for idx in [5, 7, 9, 11, 13, 15]:
                    confidence[idx] = float(item["left_confidence"])
            if item.get("right_confidence") is not None:
                for idx in [6, 8, 10, 12, 14, 16]:
                    confidence[idx] = float(item["right_confidence"])
            shift = float(item.get("elbow_shift", 0.0))
            keypoints[7, 0] += shift
            keypoints[8, 0] += shift
            return keypoints, confidence

        wl.layDiem = fake_points

        def fake_angle(*_args):
            item = model.current
            angles = item["angles"]
            call_index = model.angle_call
            model.angle_call += 1
            if isinstance(angles, (tuple, list)):
                return float(angles[call_index])
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

    def test_squat_full_cycle_counts_one_good_rep(self):
        sequence = (
            [_item((170, 170))] * 8
            + [_item((90, 170))] * 20
            + [_item((170, 170))] * 22
        )
        processor = self._run(wl.SquatProcessor, sequence)
        self.assertEqual(processor.tongsolan, 1)
        self.assertEqual(processor.solandung, 1)
        self.assertEqual(processor.trangthai, "STANDING")
        self.assertGreaterEqual(processor.last_rep_score, 70)

    def test_shallow_squat_is_bad_rep(self):
        sequence = (
            [_item((170, 170))] * 8
            + [_item((132, 170))] * 18
            + [_item((170, 170))] * 22
        )
        processor = self._run(wl.SquatProcessor, sequence)
        self.assertEqual(processor.tongsolan, 1)
        self.assertEqual(processor.solandung, 0)
        self.assertEqual(processor.shared_state["last_error_code"], "notlow")

    def test_pushup_full_cycle_counts_one_good_rep(self):
        sequence = (
            [_item((170, 175))] * 8
            + [_item((80, 175))] * 20
            + [_item((170, 175))] * 22
        )
        processor = self._run(wl.PushupProcessor, sequence)
        self.assertEqual(processor.tongsolan, 1)
        self.assertEqual(processor.solandung, 1)
        self.assertEqual(processor.trangthai, "PLANK_UP")

    def test_shallow_pushup_is_counted_but_not_good(self):
        sequence = (
            [_item((170, 175))] * 8
            + [_item((122, 175))] * 18
            + [_item((170, 175))] * 22
        )
        processor = self._run(wl.PushupProcessor, sequence)
        self.assertEqual(processor.tongsolan, 1)
        self.assertEqual(processor.solandung, 0)
        self.assertEqual(processor.shared_state["bad_rep"], 1)
        self.assertEqual(processor.shared_state["last_error_code"], "notlow")

    def test_pushup_bad_body_line_is_recorded(self):
        sequence = (
            [_item((170, 175))] * 8
            + [_item((80, 132))] * 20
            + [_item((170, 175))] * 22
        )
        processor = self._run(wl.PushupProcessor, sequence)
        self.assertEqual(processor.tongsolan, 1)
        self.assertEqual(processor.solandung, 0)
        self.assertGreaterEqual(processor.shared_state["error_code_counts"].get("bodyline", 0), 1)

    def test_curl_full_cycle_counts_one_good_rep(self):
        sequence = (
            [_item(170)] * 10
            + [_item(55)] * 22
            + [_item(170)] * 26
        )
        processor = self._run(wl.CurlProcessor, sequence, "left")
        self.assertEqual(processor.tongsolan, 1)
        self.assertEqual(processor.solandung, 1)
        self.assertEqual(processor.trangthai, "EXTENDED")
        self.assertEqual(processor.shared_state["last_error_code"], "")

    def test_curl_uses_personal_extension_baseline_not_180_degrees(self):
        sequence = (
            [_item(148)] * 12
            + [_item(62)] * 22
            + [_item(146)] * 26
        )
        processor = self._run(wl.CurlProcessor, sequence, "left")
        self.assertEqual(processor.tongsolan, 1)
        self.assertEqual(processor.solandung, 1)
        self.assertTrue(processor.shared_state["curl_calibration_ready"])
        self.assertLess(processor.shared_state["curl_extension_baseline"], 160)
        self.assertNotIn("notstraight", processor.rep_records[0]["error_codes"])

    def test_curl_incomplete_flexion_is_bad_rep(self):
        sequence = (
            [_item(170)] * 10
            + [_item(112)] * 18
            + [_item(170)] * 24
        )
        processor = self._run(wl.CurlProcessor, sequence, "left")
        self.assertEqual(processor.tongsolan, 1)
        self.assertEqual(processor.solandung, 0)
        self.assertEqual(processor.shared_state["last_error_code"], "notbend")

    def test_curl_incomplete_extension_is_observable(self):
        sequence = (
            [_item(170)] * 10
            + [_item(60)] * 22
            + [_item(128)] * 18
            + [_item(70)] * 18
        )
        processor = self._run(wl.CurlProcessor, sequence, "left")
        self.assertEqual(processor.tongsolan, 1)
        self.assertEqual(processor.solandung, 0)
        self.assertIn("notstraight", processor.rep_records[0]["error_codes"])
        self.assertGreaterEqual(processor.shared_state["phase_end_error"], 1)

    def test_low_confidence_during_rep_aborts_without_counting(self):
        sequence = (
            [_item((170, 170))] * 8
            + [_item((110, 170))] * 8
            + [_item((100, 170), confidence=0.05)] * 10
            + [_item((170, 170))] * 10
        )
        processor = self._run(wl.SquatProcessor, sequence)
        self.assertEqual(processor.tongsolan, 0)
        self.assertGreaterEqual(processor.shared_state["tracking_abort_count"], 1)
        self.assertEqual(processor.trangthai, "STANDING")

    def test_side_is_locked_for_active_rep(self):
        sequence = (
            [_item((170, 170), left_confidence=0.95, right_confidence=0.70)] * 8
            + [_item((90, 170), left_confidence=0.90, right_confidence=0.99)] * 20
            + [_item((170, 170), left_confidence=0.90, right_confidence=0.99)] * 22
        )
        processor = self._run(wl.SquatProcessor, sequence)
        self.assertEqual(processor.tongsolan, 1)
        self.assertEqual(processor.rep_records[0]["is_good"], True)

    def test_two_squat_cycles_count_two_reps(self):
        one_cycle = (
            [_item((170, 170))] * 8
            + [_item((90, 170))] * 20
            + [_item((170, 170))] * 22
        )
        processor = self._run(wl.SquatProcessor, one_cycle + one_cycle)
        self.assertEqual(processor.tongsolan, 2)
        self.assertEqual(processor.solandung, 2)
        self.assertEqual(len(processor.rep_records), 2)

    def test_squat_down_only_does_not_count_until_return_to_top(self):
        sequence = (
            [_item((158, 170))] * 10
            + [_item((100, 170))] * 24
        )
        processor = self._run(wl.SquatProcessor, sequence)
        self.assertEqual(processor.tongsolan, 0)
        self.assertIn(processor.trangthai, {"DESCENDING", "BOTTOM"})

    def test_squat_personal_standing_angle_counts_one_down_up_cycle(self):
        sequence = (
            [_item((152, 170))] * 12
            + [_item((105, 170))] * 22
            + [_item((150, 170))] * 26
        )
        processor = self._run(wl.SquatProcessor, sequence)
        self.assertEqual(processor.tongsolan, 1)
        self.assertLess(processor.shared_state["squat_standing_baseline"], 160)

    def test_threshold_noise_does_not_create_false_squat_rep(self):
        sequence = [
            _item((value, 170))
            for value in [170, 160, 150, 148, 146, 150, 147, 151, 149, 155, 160, 170] * 3
        ]
        processor = self._run(wl.SquatProcessor, sequence)
        self.assertEqual(processor.tongsolan, 0)

    def test_pushup_down_only_does_not_count_until_return_to_top(self):
        sequence = (
            [_item((145, 175))] * 12
            + [_item((85, 175))] * 24
        )
        processor = self._run(wl.PushupProcessor, sequence)
        self.assertEqual(processor.tongsolan, 0)
        self.assertIn(processor.trangthai, {"LOWERING", "BOTTOM"})

    def test_pushup_personal_top_angle_counts_one_down_up_cycle(self):
        sequence = (
            [_item((145, 175))] * 12
            + [_item((82, 175))] * 22
            + [_item((143, 175))] * 28
        )
        processor = self._run(wl.PushupProcessor, sequence)
        self.assertEqual(processor.tongsolan, 1)
        self.assertLess(processor.shared_state["pushup_top_baseline"], 155)

    def test_rep_records_include_score_components_and_confidence(self):
        sequence = (
            [_item((170, 175), confidence=0.92)] * 8
            + [_item((80, 175), confidence=0.92)] * 20
            + [_item((170, 175), confidence=0.92)] * 22
        )
        processor = self._run(wl.PushupProcessor, sequence)
        record = processor.rep_records[0]
        self.assertIn("components", record)
        self.assertIn("depth", record["components"])
        self.assertGreater(record["confidence"], 0.8)
        self.assertEqual(processor.shared_state["quality_score_avg"], record["score"])


    def test_unknown_exercise_processor_does_not_load_pose_model(self):
        processor = wl.UnsupportedExerciseProcessor("yoga-tree", shared_state={})
        self.assertIsNone(processor.model)
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        processor.process(frame)
        self.assertEqual(processor.trangthai, "UNSUPPORTED")


if __name__ == "__main__":
    unittest.main()
