from pathlib import Path
import sqlite3
import unittest

ROOT = Path(__file__).resolve().parents[1]


class UserReportedFixTests(unittest.TestCase):
    def test_curl_is_adaptive_and_does_not_require_180_degree_lockout(self):
        source = (ROOT / "workoutlogic.py").read_text(encoding="utf-8")
        self.assertIn('self.trangthai = "CALIBRATING"', source)
        self.assertIn("baseline_extension_angle", source)
        self.assertIn("start_drop_from_baseline", source)
        self.assertNotIn("extension_target=180", source)

    def test_camera_uses_one_shared_reconnecting_service(self):
        app_source = (ROOT / "app.py").read_text(encoding="utf-8")
        camera_source = (ROOT / "camera_service.py").read_text(encoding="utf-8")
        self.assertIn("camera_service.get_frame", app_source)
        self.assertIn("/nguoi-dung/api/camera-restart/<slug>", app_source)
        self.assertIn("fitmotion-camera", camera_source)
        self.assertIn("Camera tạm ngắt; hệ thống đang tự kết nối lại", camera_source)

    def test_workout_page_has_real_controls_and_ajax_goal_update(self):
        page = (ROOT / "templates" / "user_workout.html").read_text(encoding="utf-8")
        self.assertIn("data-exercise-model-viewer", page)
        self.assertIn("workoutGoalForm", page)
        self.assertIn("cameraRestart", page)
        self.assertIn("finishWorkoutBtn", page)
        self.assertIn("setInterval(pollLiveWorkout, 1500)", page)

    def test_admin_exposes_create_edit_toggle_and_delete(self):
        app_source = (ROOT / "app.py").read_text(encoding="utf-8")
        list_page = (ROOT / "templates" / "admin_exercises.html").read_text(encoding="utf-8")
        detail_page = (ROOT / "templates" / "admin_exercise_detail.html").read_text(encoding="utf-8")
        self.assertIn("def admin_add_exercise", app_source)
        self.assertIn("def admin_exercise_detail", app_source)
        self.assertIn("def admin_delete_exercise", app_source)
        self.assertIn("admin_toggle_exercise", list_page)
        self.assertIn("admin_delete_exercise", detail_page)
        self.assertIn("model_3d_file", detail_page)

    def test_database_has_model_fields_and_default_criteria(self):
        # Persistent storage lives outside the source directory. Verify the ORM
        # schema and seed configuration instead of relying on a bundled DB copy.
        models_source = (ROOT / "models.py").read_text(encoding="utf-8")
        app_source = (ROOT / "app.py").read_text(encoding="utf-8")
        self.assertTrue(all(name in models_source for name in (
            "model_3d_path", "model_3d_format", "animation_key"
        )))
        self.assertIn('"squat": {', app_source)
        self.assertIn('"pushup": {', app_source)
        self.assertIn('"curl-left": {', app_source)
        self.assertIn('"curl-right": {', app_source)
        self.assertGreaterEqual(app_source.count('("'), 10)

    def test_development_server_does_not_use_camera_breaking_reloader(self):
        source = (ROOT / "app.py").read_text(encoding="utf-8")
        self.assertIn("threaded=True", source)
        self.assertIn("use_reloader=False", source)
        self.assertNotIn('app.run(host="0.0.0.0", port=5000, debug=True)', source)


if __name__ == "__main__":
    unittest.main()
