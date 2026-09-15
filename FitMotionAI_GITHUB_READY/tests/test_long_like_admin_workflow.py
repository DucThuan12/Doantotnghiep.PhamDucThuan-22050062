import unittest
from pathlib import Path
import tempfile
from unittest.mock import patch

import sys
import types

try:
    import ultralytics  # noqa: F401
except ModuleNotFoundError:
    sys.modules["ultralytics"] = types.SimpleNamespace(YOLO=lambda *args, **kwargs: None)

from tts_service import TTSResult, synthesize_feedback
from workoutlogic import BaseProcessor, LearnedExerciseProcessor
import numpy as np


class MixamoViewerTests(unittest.TestCase):
    def test_viewer_requires_real_asset_and_has_no_procedural_fallback(self):
        source = Path("static/exercise_model_viewer.js").read_text(encoding="utf-8")
        self.assertIn("FBXLoader", source)
        self.assertIn("GLTFLoader", source)
        self.assertIn("AnimationMixer", source)
        self.assertNotIn("buildProceduralHumanoid", source)
        self.assertNotIn("animateProcedural", source)
        self.assertIn("không dùng mô hình giả", source)

    def test_admin_text_requires_mixamo_asset(self):
        detail = Path("templates/admin_exercise_detail.html").read_text(encoding="utf-8")
        add = Path("templates/admin_exercise_add.html").read_text(encoding="utf-8")
        self.assertIn("Mô hình Mixamo FBX/GLB", detail)
        self.assertIn("mesh, texture, rig và animation", detail)
        self.assertIn("Mixamo FBX/GLB + animation", add)
        self.assertNotIn("dùng humanoid dự phòng", detail.lower())


class WebTrainingWorkflowTests(unittest.TestCase):
    def test_admin_trains_directly_on_web_and_polls_status(self):
        app = Path("app.py").read_text(encoding="utf-8")
        template = Path("templates/admin_exercise_detail.html").read_text(encoding="utf-8")
        train = Path("train.py").read_text(encoding="utf-8")
        self.assertIn('/train-model', app)
        self.assertIn('/training-status', app)
        self.assertIn("_train_exercise_model_job", app)
        self.assertIn("pose_model_status = \"training\"", app)
        self.assertIn("pose_model_status = \"ready\"", app)
        self.assertIn("POSE_MODEL_MIN_TEST_ACCURACY", app)
        self.assertIn("Huấn luyện mô hình trên web", template)
        self.assertIn("trainModelButton", template)
        self.assertIn("--labels", train)
        self.assertIn("split_sources(sources", train)
        self.assertIn("training_validation.required_labels", template)
        self.assertIn('name="joint_a"', template)
        self.assertIn('name="joint_b"', template)
        self.assertIn('name="joint_c"', template)

    def test_custom_exercise_uses_three_state_learned_processor(self):
        app = Path("app.py").read_text(encoding="utf-8")
        logic = Path("workoutlogic.py").read_text(encoding="utf-8")
        self.assertIn('f"{slug}_start"', app)
        self.assertIn('f"{slug}_middle"', app)
        self.assertIn('f"{slug}_end"', app)
        self.assertIn("return LearnedExerciseProcessor(slug, shared_state)", app)
        self.assertIn("start -> middle -> end", logic)
        # No checkpoint => processor must not fake a train result.
        processor = LearnedExerciseProcessor("custom-demo", shared_state={"pose_classifier_model_path": "/tmp/not-exist.pth"}, model=object())
        self.assertFalse(processor.hybrid_guard.available)

    def test_admin_three_joint_rule_is_executable_by_realtime_processor(self):
        shared = {
            "admin_criteria_rules": [{
                "id": 1,
                "error_code": "elbow_open",
                "phase": "middle",
                "joint_indices": [0, 1, 2],
                "operator": ">=",
                "angle_value": 80,
                "message_text": "Góc đang mở quá rộng",
                "advice_text": "Khép góc lại.",
            }]
        }
        processor = BaseProcessor(shared_state=shared, load_model=False)
        kp = np.zeros((17, 2), dtype=np.float32)
        kp[0] = [1.0, 0.0]
        kp[1] = [0.0, 0.0]
        kp[2] = [0.0, 1.0]
        conf = np.ones(17, dtype=np.float32)
        errors = processor.evaluate_admin_criteria(kp, conf, "middle")
        self.assertEqual(len(errors), 1)
        self.assertEqual(errors[0][0], "elbow_open")
        self.assertEqual(errors[0][1], "Góc đang mở quá rộng")

    def test_core_processors_capture_admin_rules_across_rep_phases(self):
        logic = Path("workoutlogic.py").read_text(encoding="utf-8")
        # Squat, Push-up and Curl must execute Admin A-B-C rules in the
        # realtime path, not merely store them in the database.
        self.assertGreaterEqual(logic.count('capture_admin_criteria(kp, conf, "start")'), 3)
        self.assertGreaterEqual(logic.count('capture_admin_criteria(kp, conf, "middle")'), 3)
        self.assertGreaterEqual(logic.count('capture_admin_criteria(kp, conf, "end")'), 4)


class GeneratedAudioWorkflowTests(unittest.TestCase):
    def test_admin_rule_generates_and_runtime_prefers_saved_audio(self):
        app = Path("app.py").read_text(encoding="utf-8")
        detail = Path("templates/admin_exercise_detail.html").read_text(encoding="utf-8")
        workout = Path("templates/user_workout.html").read_text(encoding="utf-8")
        self.assertIn("synthesize_feedback", app)
        self.assertIn("audio_provider", app)
        self.assertIn("audio_error", app)
        self.assertIn("error_code=error_code", app)
        self.assertIn("Tạo lại âm thanh", detail)
        self.assertIn("feedback_audio_url", app)
        self.assertIn("generatedFeedbackAudio", workout)
        self.assertIn("await generatedFeedbackAudio.play()", workout)
        self.assertIn("speakWithBrowserTts", workout)
        self.assertIn("joint_indices_json", app)
        self.assertTrue(Path("scripts/setup_piper_vi.py").is_file())
        setup = Path("scripts/setup_piper_vi.py").read_text(encoding="utf-8")
        self.assertIn("vi_VN-vais1000-medium", setup)
        service = Path("tts_service.py").read_text(encoding="utf-8")
        self.assertIn('"-m",', service)
        self.assertIn('"-f",', service)

    def test_tts_service_does_not_report_fake_wav(self):
        with tempfile.TemporaryDirectory() as td:
            output = Path(td) / "feedback.wav"
            with patch("tts_service._run_piper", return_value=TTSResult(False, "piper", error="no piper")), \
                 patch("tts_service._run_edge_tts", return_value=TTSResult(False, "edge-tts", error="no edge")), \
                 patch("tts_service._run_gtts", return_value=TTSResult(False, "gtts", error="no gtts")), \
                 patch("tts_service._run_pyttsx3", return_value=TTSResult(False, "pyttsx3", error="no voice")):
                result = synthesize_feedback("Giữ lưng thẳng", output)
            self.assertFalse(result.success)
            self.assertFalse(output.exists())
            self.assertIn("Piper", result.error)


if __name__ == "__main__":
    unittest.main()
