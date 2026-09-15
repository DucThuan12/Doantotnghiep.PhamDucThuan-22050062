import unittest
from pathlib import Path


class EmergencyContactSourceTests(unittest.TestCase):
    def test_registration_and_profile_collect_contact_fields(self):
        register = Path("templates/auth_register.html").read_text(encoding="utf-8")
        profile = Path("templates/user_profile.html").read_text(encoding="utf-8")
        app = Path("app.py").read_text(encoding="utf-8")
        for field in ("phone", "contact_address", "emergency_contact_name", "emergency_contact_relation", "emergency_contact_phone"):
            self.assertIn(f'name="{field}"', register)
            self.assertIn(f'name="{field}"', profile)
            self.assertIn(field, app)

    def test_emergency_alert_is_persistent_audit_model(self):
        models = Path("models.py").read_text(encoding="utf-8")
        self.assertIn("class EmergencyAlert", models)
        self.assertIn('__tablename__ = "emergency_alerts"', models)
        self.assertIn("event_key", models)
        self.assertIn("status", models)
        self.assertIn("read_at", models)
        self.assertIn("resolved_at", models)
        self.assertIn("emergency_phone_snapshot", models)

    def test_confirmed_emergency_is_persisted_and_exposed_to_admin(self):
        app = Path("app.py").read_text(encoding="utf-8")
        self.assertIn("persist_emergency_alert_if_needed", app)
        self.assertIn('/admin/api/thong-bao-khan-cap', app)
        self.assertIn("admin_mark_emergency_read", app)
        self.assertIn("admin_resolve_emergency_alert", app)
        self.assertIn('shared_state["emergency_alert_id"]', app)
        self.assertIn("stream_with_context(gen_frames", app)
        self.assertIn("persist_emergency_alert_if_needed(user_id, slug, shared_state)", app)

    def test_admin_bell_polls_and_shows_floating_notification(self):
        bell = Path("templates/_admin_notification_bell.html").read_text(encoding="utf-8")
        alerts = Path("templates/admin_emergency_alerts.html").read_text(encoding="utf-8")
        users = Path("templates/admin_users.html").read_text(encoding="utf-8")
        css = Path("static/app.css").read_text(encoding="utf-8")
        self.assertIn("adminBellCount", bell)
        self.assertIn("setInterval(poll, 2500)", bell)
        self.assertIn("adminEmergencyToast", bell)
        self.assertIn("Đã liên hệ / xử lý", alerts)
        self.assertIn("tel:", alerts)
        self.assertIn("tel:", users)
        self.assertIn("SĐT khẩn cấp", users)
        self.assertIn("admin-emergency-toast", css)


if __name__ == "__main__":
    unittest.main()
