from __future__ import annotations

import sys
import types
import unittest
from pathlib import Path

import numpy as np

try:
    import ultralytics  # noqa: F401
except ModuleNotFoundError:
    sys.modules["ultralytics"] = types.SimpleNamespace(YOLO=lambda *args, **kwargs: None)

import helper
import workoutlogic as wl

ROOT = Path(__file__).resolve().parents[1]


class _FakeResult:
    def __init__(self, frame):
        self.frame = frame

    def plot(self):
        return self.frame.copy()


class _NoopModel:
    def __call__(self, frame, verbose=False):
        return [_FakeResult(frame)]


def _vertical_pose():
    kp = np.zeros((17, 2), dtype=np.float32)
    # shoulders 5/6 and hips 11/12 => clearly upright torso
    kp[5] = [220, 120]
    kp[6] = [320, 120]
    kp[11] = [230, 320]
    kp[12] = [310, 320]
    # arms/legs only need valid finite points for confidence checks.
    kp[7], kp[9] = [210, 200], [205, 280]
    kp[8], kp[10] = [330, 200], [335, 280]
    kp[13], kp[15] = [235, 410], [235, 500]
    kp[14], kp[16] = [305, 410], [305, 500]
    return kp, np.ones(17, dtype=np.float32)


def _horizontal_pose():
    kp = np.zeros((17, 2), dtype=np.float32)
    kp[5] = [130, 180]
    kp[6] = [130, 260]
    kp[11] = [430, 185]
    kp[12] = [430, 255]
    kp[7], kp[9] = [180, 155], [220, 145]
    kp[8], kp[10] = [180, 285], [220, 295]
    kp[13], kp[15] = [520, 190], [610, 190]
    kp[14], kp[16] = [520, 250], [610, 250]
    return kp, np.ones(17, dtype=np.float32)


class CameraTextTests(unittest.TestCase):
    def test_overlay_text_is_ascii_only(self):
        value = helper.to_ascii_overlay("CẤP ĐỘ: TRUNG BÌNH – Hít đất đúng, cần chỉnh")
        self.assertEqual(value.encode("ascii", "strict").decode("ascii"), value)
        self.assertIn("CAP DO", value)
        self.assertIn("TRUNG BINH", value)
        self.assertIn("Hit dat dung", value)


class ExerciseIdentityTests(unittest.TestCase):
    def test_identity_gate_distinguishes_upright_from_pushup(self):
        upright_kp, upright_conf = _vertical_pose()
        horizontal_kp, horizontal_conf = _horizontal_pose()

        push = wl.BaseProcessor(
            shared_state={"exercise_identity_gate_enabled": True},
            load_model=False,
        )
        # two matching frames are required before allowing a state machine.
        self.assertFalse(push.posture_identity_gate("pushup", horizontal_kp, horizontal_conf, strict_unknown=True))
        self.assertTrue(push.posture_identity_gate("pushup", horizontal_kp, horizontal_conf, strict_unknown=True))

        wrong = wl.BaseProcessor(
            shared_state={"exercise_identity_gate_enabled": True},
            load_model=False,
        )
        for _ in range(3):
            accepted = wrong.posture_identity_gate("pushup", upright_kp, upright_conf, strict_unknown=True)
        self.assertFalse(accepted)
        self.assertEqual(wrong.shared_state["exercise_identity_reason"], "posture_mismatch")

    def test_upright_motion_cannot_count_on_pushup_page(self):
        model = _NoopModel()
        kp, conf = _vertical_pose()
        original = (wl.coNguoi, wl.layDiem, wl.tinhgoc)
        wl.coNguoi = lambda _results: True
        wl.layDiem = lambda _results: (kp.copy(), conf.copy())
        # Even if elbow values would look like a down/up push-up cycle, the
        # upright body identity gate must stop the push-up state machine first.
        values = iter(([170, 175] * 10) + ([80, 175] * 20) + ([170, 175] * 20))
        wl.tinhgoc = lambda *_args: float(next(values, 170))
        try:
            processor = wl.PushupProcessor(
                shared_state={"exercise_identity_gate_enabled": True}, model=model
            )
            processor.update_emergency = lambda *a, **k: None
            frame = np.zeros((540, 960, 3), dtype=np.uint8)
            for _ in range(30):
                processor.process(frame)
            self.assertEqual(processor.tongsolan, 0)
            self.assertEqual(processor.shared_state["exercise_identity_reason"], "posture_mismatch")
        finally:
            wl.coNguoi, wl.layDiem, wl.tinhgoc = original


class CurlSideTests(unittest.TestCase):
    def test_wrong_arm_feedback_names_selected_and_opposite_side(self):
        processor = wl.CurlProcessor("left", shared_state={}, model=object())
        processor.opposite_extension_baseline = 165.0
        # Force a clear downward trend on the opposite/right elbow signal.
        for value in (165.0, 165.0, 100.0, 90.0):
            processor.temporal.update("curl_opposite_elbow", value)
        # New threshold: need 6 stable frames (opposite drop >= 50 deg) before firing.
        # The first 5 calls accumulate; the 6th triggers the rejection.
        for i in range(5):
            self.assertFalse(processor._detect_pre_rep_wrong_side(155.0, 90.0))
        self.assertTrue(processor._detect_pre_rep_wrong_side(155.0, 90.0))
        self.assertEqual(processor.shared_state["last_error_code"], "wrongside")
        spoken = processor.shared_state["voice_text"].lower()
        self.assertIn("tay trái", spoken)
        self.assertIn("tay phải", spoken)


class ViewerAndAudioWorkflowTests(unittest.TestCase):
    def test_curl_mask_and_pushup_intro_main_sequence_are_present(self):
        source = (ROOT / "static" / "exercise_model_viewer.js").read_text(encoding="utf-8")
        self.assertIn("prepareClipForExercise", source)
        self.assertIn("freezeTrackAtFirstPose", source)
        self.assertIn("curl-left", source)
        self.assertIn("curl-right", source)
        self.assertIn("THREE.LoopOnce", source)
        self.assertIn("startMainLoop", source)
        self.assertIn("THREE.LoopRepeat", source)

    def test_user_3d_missing_asset_comment_is_not_rendered(self):
        page = (ROOT / "templates" / "user_workout.html").read_text(encoding="utf-8")
        viewer = (ROOT / "static" / "exercise_model_viewer.js").read_text(encoding="utf-8")
        self.assertIn('data-viewer-status hidden', page)
        self.assertNotIn("Chưa gắn mô hình Mixamo FBX/GLB có rig + animation", viewer)

    def test_admin_rule_to_generated_audio_to_workout_chain_is_connected(self):
        app = (ROOT / "app.py").read_text(encoding="utf-8")
        workout = (ROOT / "templates" / "user_workout.html").read_text(encoding="utf-8")
        self.assertIn("tts = create_tts_audio", app)
        self.assertIn("refresh_live_exercise_criteria(exercise)", app)
        self.assertIn('"feedback_audio_url": feedback_audio_url', app)
        self.assertIn("await generatedFeedbackAudio.play()", workout)
        self.assertIn("speakWithBrowserTts", workout)


if __name__ == "__main__":
    unittest.main()
