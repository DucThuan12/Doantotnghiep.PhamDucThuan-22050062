import unittest
from pathlib import Path


class MixamoAdminSessionFixTests(unittest.TestCase):
    def test_admin_json_guard_does_not_flash_or_redirect(self):
        auth = Path("auth.py").read_text(encoding="utf-8")
        self.assertIn("def admin_api_required", auth)
        block = auth.split("def admin_api_required", 1)[1]
        self.assertIn("jsonify", block)
        self.assertNotIn('flash("Bạn không có quyền', block)

    def test_emergency_json_routes_use_api_guard(self):
        app = Path("app.py").read_text(encoding="utf-8")
        for name in ("admin_emergency_notifications_api", "admin_mark_emergency_read", "admin_resolve_emergency_alert"):
            marker = f"@admin_api_required\ndef {name}"
            self.assertIn(marker, app)

    def test_legacy_procedural_model_state_is_hidden(self):
        app = Path("app.py").read_text(encoding="utf-8")
        self.assertIn("model_format not in", app)
        page = Path("templates/admin_exercise_detail.html").read_text(encoding="utf-8")
        self.assertNotIn("Dữ liệu liên quan", page)
        self.assertIn("Chưa gắn mô hình", page)

    def test_bell_stops_polling_after_role_changes(self):
        bell = Path("templates/_admin_notification_bell.html").read_text(encoding="utf-8")
        self.assertIn("res.status === 401 || res.status === 403", bell)
        self.assertIn("pollingEnabled = false", bell)

if __name__ == "__main__":
    unittest.main()
