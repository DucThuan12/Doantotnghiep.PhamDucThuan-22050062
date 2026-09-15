import ast
import sqlite3
import tempfile
import unittest
from datetime import date
from pathlib import Path
from types import SimpleNamespace

from jinja2 import Environment, FileSystemLoader, StrictUndefined

from storage import ensure_storage_schema
from schedule_logic import (
    build_month_grid,
    month_bounds,
    parse_month,
    recurrence_dates,
    shift_month,
    summarize_weekdays,
    validate_schedule_range,
)


ROOT = Path(__file__).resolve().parents[1]


class ScheduleLogicTests(unittest.TestCase):
    def test_custom_availability_is_not_forced_to_fixed_weekdays(self):
        # User is busy Tuesday/Wednesday/Thursday and chooses Mon/Fri/Sun.
        dates = recurrence_dates(
            date(2026, 8, 3),
            date(2026, 8, 16),
            [0, 4, 6],
        )
        self.assertEqual(
            [item.isoformat() for item in dates],
            [
                "2026-08-03", "2026-08-07", "2026-08-09",
                "2026-08-10", "2026-08-14", "2026-08-16",
            ],
        )

    def test_schedule_can_span_months_and_does_not_reset_after_one_week(self):
        dates = recurrence_dates(
            date(2026, 8, 3),
            date(2026, 10, 31),
            [1, 5],
        )
        self.assertGreater(len(dates), 20)
        self.assertTrue(any(item.month == 9 for item in dates))
        self.assertTrue(any(item.month == 10 for item in dates))

    def test_invalid_range_is_rejected(self):
        with self.assertRaises(ValueError):
            validate_schedule_range(date(2026, 8, 10), date(2026, 8, 9))
        with self.assertRaises(ValueError):
            validate_schedule_range(date(2026, 1, 1), date(2027, 2, 1), max_days=366)

    def test_month_navigation_and_grid(self):
        self.assertEqual(shift_month(2026, 1, -1), (2025, 12))
        self.assertEqual(shift_month(2026, 12, 1), (2027, 1))
        self.assertEqual(parse_month("2026-08"), (2026, 8))
        self.assertEqual(month_bounds(2026, 2), (date(2026, 2, 1), date(2026, 2, 28)))
        grid = build_month_grid(2026, 8)
        self.assertIn(len(grid), {5, 6})
        self.assertTrue(all(len(week) == 7 for week in grid))

    def test_weekday_summary(self):
        self.assertEqual(summarize_weekdays([0, 4, 6]), "Thứ Hai, Thứ Sáu, Chủ Nhật")


class ScheduleIntegrationSourceTests(unittest.TestCase):

    def test_storage_layer_creates_recurring_schedule_tables(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "aifitness.db"
            ensure_storage_schema(db_path)
            conn = sqlite3.connect(db_path)
            try:
                names = {row[0] for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()}
            finally:
                conn.close()
            self.assertIn("workout_schedules", names)
            self.assertIn("workout_schedule_items", names)

    def test_database_models_are_durable(self):
        text = (ROOT / "models.py").read_text(encoding="utf-8")
        self.assertIn('class WorkoutSchedule(db.Model)', text)
        self.assertIn('class WorkoutScheduleItem(db.Model)', text)
        self.assertIn('__tablename__ = "workout_schedules"', text)
        self.assertIn('__tablename__ = "workout_schedule_items"', text)
        self.assertIn('schedule_id = db.Column', text)

    def test_schedule_models_are_declared_exactly_once(self):
        source = (ROOT / "models.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        class_names = [
            node.name for node in tree.body if isinstance(node, ast.ClassDef)
        ]
        self.assertEqual(class_names.count("WorkoutSchedule"), 1)
        self.assertEqual(class_names.count("WorkoutScheduleItem"), 1)

    def test_route_materializes_database_occurrences(self):
        text = (ROOT / "app.py").read_text(encoding="utf-8")
        self.assertIn('@app.route("/nguoi-dung/lich-tap", methods=["GET", "POST"])', text)
        self.assertIn('materialize_schedule(schedule)', text)
        self.assertIn('replace_future_occurrences(schedule, today=today)', text)
        self.assertIn('archive_schedule(schedule)', text)


    def test_schedule_template_renders_saved_dict_without_items_collision(self):
        templates_dir = ROOT / "templates"
        env = Environment(
            loader=FileSystemLoader(str(templates_dir)),
            undefined=StrictUndefined,
            autoescape=True,
        )
        env.globals.update(
            url_for=lambda endpoint, **kwargs: f"/{endpoint}",
            get_flashed_messages=lambda: [],
        )
        exercise = SimpleNamespace(id=1, name="Curl trái")
        context = {
            "title": "Lịch tập cá nhân",
            "user": {"name": "Thuận"},
            "exercises": [exercise],
            "weekday_labels": ["Thứ Hai", "Thứ Ba", "Thứ Tư", "Thứ Năm", "Thứ Sáu", "Thứ Bảy", "Chủ Nhật"],
            "schedules": [{
                "id": 7,
                "name": "Lịch cá nhân",
                "start_date": "2026-08-04",
                "end_date": "2026-10-31",
                "status": "active",
                "weekday_groups": [{
                    "weekday": 0,
                    "weekday_label": "Thứ Hai",
                    "exercises": [{
                        "exercise_id": 1,
                        "exercise_name": "Curl trái",
                        "set_count": 2,
                        "rep_target": 10,
                    }],
                }],
            }],
            "edit_schedule": None,
            "calendar_data": {
                "label": "Tháng 08/2026",
                "month_key": "2026-08",
                "weeks": [[None, None, None, None, None, None, None]],
            },
            "previous_month": "2026-07",
            "next_month": "2026-09",
            "today": "2026-08-04",
            "default_end": "2026-10-26",
        }
        rendered = env.get_template("user_schedule.html").render(**context)
        self.assertIn("Lịch cá nhân", rendered)
        self.assertIn("Curl trái", rendered)
        self.assertNotIn("builtin_function_or_method", rendered)

    def test_template_does_not_use_dict_items_attribute_for_schedule_rows(self):
        template = (ROOT / "templates" / "user_schedule.html").read_text(encoding="utf-8")
        self.assertNotIn("schedule.items", template)
        self.assertNotIn("edit_schedule.items", template)
        self.assertIn("schedule['weekday_groups']", template)
        self.assertIn("edit_schedule['rows']", template)

    def test_schedule_is_connected_to_dashboard_analytics_and_workout(self):
        app_source = (ROOT / "app.py").read_text(encoding="utf-8")
        analytics = (ROOT / "templates" / "user_analytics.html").read_text(encoding="utf-8")
        workout = (ROOT / "templates" / "user_workout.html").read_text(encoding="utf-8")
        self.assertIn("materialize_schedule(schedule)", app_source)
        self.assertIn('get_week_schedule(user["id"]', app_source)
        self.assertIn("schedule_planned_count", app_source)
        self.assertIn("linked_schedule_name", app_source)
        self.assertIn("analyticsSchedulePlanned", analytics)
        self.assertIn("Theo lịch cá nhân", workout)
        self.assertIn('name="workout_date" value="{{ session_date }}" readonly', workout)
        self.assertIn("workout_date = session_date", app_source)

    def test_month_calendar_and_form_exist(self):
        template = (ROOT / "templates" / "user_schedule.html").read_text(encoding="utf-8")
        self.assertIn('name="weekday[]"', template)
        self.assertIn('name="exercise_id[]"', template)
        self.assertIn('name="start_date"', template)
        self.assertIn('name="end_date"', template)
        self.assertIn('month-calendar-grid', template)
        self.assertIn('Ngừng lịch', template)


if __name__ == "__main__":
    unittest.main()
