import tempfile
import unittest
from pathlib import Path
import sys
import types

import numpy as np
import torch

try:
    import ultralytics  # noqa: F401
except ModuleNotFoundError:
    sys.modules["ultralytics"] = types.SimpleNamespace(YOLO=lambda *args, **kwargs: None)

from pose_learning import (
    BilateralCurlGuard,
    HybridPoseRuntimeGuard,
    build_mlp,
    normalize_yolo_pose,
)
from workoutlogic import CurlProcessor


class PoseLearningFeatureTests(unittest.TestCase):
    def _pose(self, shift=(0.0, 0.0), scale=1.0):
        pts = np.zeros((17, 2), dtype=np.float32)
        for i in range(17):
            pts[i] = [20 + i * 3, 30 + (i % 5) * 7]
        pts[5] = [40, 30]; pts[6] = [70, 30]
        pts[11] = [45, 80]; pts[12] = [65, 80]
        pts = pts * scale + np.asarray(shift, dtype=np.float32)
        conf = np.ones(17, dtype=np.float32) * 0.95
        return pts, conf

    def test_feature_is_51d_and_translation_scale_normalized(self):
        pts, conf = self._pose()
        transformed, _ = self._pose(shift=(123, 77), scale=2.5)
        a = normalize_yolo_pose(pts, conf)
        b = normalize_yolo_pose(transformed, conf)
        self.assertEqual(a.shape, (51,))
        np.testing.assert_allclose(a, b, atol=1e-5)

    def test_optional_guard_never_fakes_prediction_without_checkpoint(self):
        guard = HybridPoseRuntimeGuard(model_path="/tmp/definitely-missing-fitmotion.pth")
        self.assertFalse(guard.available)
        decision = guard.validate_rep("squat")
        self.assertFalse(decision.available)
        self.assertTrue(decision.accepted)

    def test_trained_guard_can_reject_wrong_curl_side(self):
        pts, conf = self._pose()
        with tempfile.TemporaryDirectory() as td:
            checkpoint = Path(td) / "pose.pth"
            model = build_mlp(51, 2)
            with torch.no_grad():
                for param in model.parameters():
                    param.zero_()
                # Force the second label (right curl) with high confidence.
                model[-1].bias[1] = 10.0
            torch.save({
                "model_state": model.state_dict(),
                "labels": ["curl_left_flexed", "curl_right_flexed"],
                "input_dim": 51,
                "feature_version": "yolo17-normalized-xyc-v1",
                "metrics": {"test": {"samples": 20, "accuracy": 1.0}},
            }, checkpoint)
            guard = HybridPoseRuntimeGuard(
                model_path=str(checkpoint), min_confidence=0.5,
                min_predictions=2, expected_ratio=0.55,
            )
            for _ in range(4):
                guard.observe(pts, conf)
            decision = guard.validate_rep("curl", expected_side="left")
            self.assertTrue(decision.available)
            self.assertFalse(decision.accepted)
            self.assertEqual(decision.reason_code, "wrong_side_ml")
            self.assertGreaterEqual(decision.test_accuracy, 0.70)
            self.assertEqual(len(decision.model_sha256), 64)


class CurlWrongSideTests(unittest.TestCase):
    def test_geometry_wrong_side_attempt_does_not_increment_total_rep(self):
        state = {"training_level": "medium"}
        processor = CurlProcessor(side="left", shared_state=state, model=object())
        # Active arm barely changes; opposite arm completes a large curl ROM.
        processor.side_guard.observe(150, 160)
        processor.side_guard.observe(145, 90)
        processor.side_guard.observe(148, 155)
        before = processor.tongsolan
        processor._complete_rep(frame=None)
        self.assertEqual(processor.tongsolan, before)
        self.assertEqual(state.get("last_rejected_attempt", {}).get("error_code"), "wrongside")
        self.assertGreaterEqual(int(state.get("rejected_attempt_count", 0)), 1)
        app = Path("app.py").read_text(encoding="utf-8")
        self.assertIn("rejected_attempt_count", app)
        self.assertIn("last_rejected_attempt", app)


class HybridTrainingPipelineSourceTests(unittest.TestCase):
    def test_train_pipeline_accepts_images_and_videos_and_splits_sources_first(self):
        source = Path("train.py").read_text(encoding="utf-8")
        self.assertIn("IMAGE_EXTENSIONS", source)
        self.assertIn("VIDEO_EXTENSIONS", source)
        self.assertIn("split_sources(sources", source)
        self.assertLess(source.index("split = split_sources"), source.index("expand_split(split[name]"))
        self.assertIn("confusion_matrix", source)
        self.assertIn(".metrics.json", source)
        runtime = Path("pose_learning.py").read_text(encoding="utf-8")
        self.assertIn("min_test_accuracy", runtime)
        self.assertIn("test set độc lập", runtime)

    def test_admin_has_labelled_training_media_upload(self):
        source = Path("app.py").read_text(encoding="utf-8")
        template = Path("templates/admin_exercise_detail.html").read_text(encoding="utf-8")
        self.assertIn("admin_training_media_upload", source)
        self.assertIn("POSE_TRAINING_DIR", source)
        self.assertIn("training_media", template)
        self.assertIn("Huấn luyện mô hình trên web", template)
        self.assertIn("admin_train_exercise_model", source)
        self.assertIn("admin_exercise_training_status", source)
        self.assertIn("pose_model_path", source)


if __name__ == "__main__":
    unittest.main()
