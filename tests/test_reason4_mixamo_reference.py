import json
import tempfile
import unittest
from pathlib import Path

from reference_motion import ReferenceMotionError, ReferenceMotionMatcher, load_reference_file


def payload(exercise="squat"):
    frames = []
    for i in range(10):
        t = i / 9.0
        frames.append({"t": t, "knee_angle": 170 - 80 * (1 - abs(2*t-1)), "torso_angle": 170 - 20 * (1 - abs(2*t-1))})
    return {
        "schema": "fitmotion.reference.v1",
        "exercise": exercise,
        "animation_key": "Squat",
        "version": "mixamo-v1",
        "source": {"type": "mixamo-blender", "action": "Squat"},
        "metrics": ["knee_angle", "torso_angle"],
        "metric_scales": {"knee_angle": 80, "torso_angle": 30},
        "min_score": 60,
        "frames": frames,
    }


class ReferenceValidationTests(unittest.TestCase):
    def _write(self, data):
        td = tempfile.TemporaryDirectory()
        path = Path(td.name) / "ref.json"
        path.write_text(json.dumps(data), encoding="utf-8")
        return td, path

    def test_only_validated_3d_sources_are_accepted(self):
        data = payload()
        data["source"]["type"] = "youtube-video"
        td, path = self._write(data)
        with td, self.assertRaises(ReferenceMotionError):
            load_reference_file(path, "squat")

    def test_curl_reference_must_match_exact_side(self):
        data = payload("curl-right")
        data["metrics"] = ["elbow_angle", "elbow_shift"]
        for i, frame in enumerate(data["frames"]):
            frame.pop("knee_angle", None); frame.pop("torso_angle", None)
            frame["elbow_angle"] = 150 - i * 5
            frame["elbow_shift"] = i
        data["metric_scales"] = {"elbow_angle": 60, "elbow_shift": 20}
        td, path = self._write(data)
        with td, self.assertRaises(ReferenceMotionError):
            load_reference_file(path, "curl-left")

    def test_perfect_sequence_scores_higher_than_bad_sequence(self):
        td, path = self._write(payload())
        with td:
            matcher = ReferenceMotionMatcher(str(path), "squat")
            matcher.start_rep()
            for frame in payload()["frames"]:
                matcher.observe({"knee_angle": frame["knee_angle"], "torso_angle": frame["torso_angle"]})
            good = matcher.finish_rep()
            self.assertTrue(good.available)
            self.assertGreaterEqual(good.score, 99)

            # Same technical trajectory at a different pace should remain high
            # because the matcher aligns a rep against the Mixamo reference.
            varied = payload()["frames"]
            matcher.start_rep()
            for idx in (0, 1, 2, 2, 3, 4, 5, 5, 6, 7, 8, 9):
                frame = varied[idx]
                matcher.observe({"knee_angle": frame["knee_angle"], "torso_angle": frame["torso_angle"]})
            speed_varied = matcher.finish_rep()
            self.assertGreaterEqual(speed_varied.score, 95)

            matcher.start_rep()
            for _ in range(10):
                matcher.observe({"knee_angle": 20, "torso_angle": 40})
            bad = matcher.finish_rep()
            self.assertLess(bad.score, good.score)


class ReferenceIntegrationSourceTests(unittest.TestCase):
    def test_blender_exporter_samples_mixamo_rig(self):
        source = Path("scripts/blender_export_mixamo_reference.py").read_text(encoding="utf-8")
        self.assertIn("mixamorig:LeftArm", source)
        self.assertIn("mixamo-blender", source)
        self.assertIn("fitmotion.reference.v1", source)
        self.assertIn("--exercise", source)

    def test_admin_upload_and_workout_runtime_use_reference(self):
        app = Path("app.py").read_text(encoding="utf-8")
        workout = Path("workoutlogic.py").read_text(encoding="utf-8")
        template = Path("templates/user_workout.html").read_text(encoding="utf-8")
        self.assertIn("admin_reference_upload", app)
        self.assertIn("configure_exercise_ai_resources", app)
        self.assertIn("ReferenceMotionMatcher", workout)
        self.assertIn("reference_match", workout)
        self.assertIn("REFERENCE_BLEND_BY_LEVEL", workout)
        self.assertIn("REFERENCE_MIN_DELTA_BY_LEVEL", workout)
        self.assertIn("DTW", Path("reference_motion.py").read_text(encoding="utf-8"))
        self.assertIn("referenceMatchScore", template)


if __name__ == "__main__":
    unittest.main()
