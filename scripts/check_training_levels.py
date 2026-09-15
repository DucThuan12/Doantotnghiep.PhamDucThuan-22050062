"""Inspect persistent DB support for Easy/Medium/Hard workout levels."""
from __future__ import annotations

import sqlite3
import tempfile
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from storage import ensure_storage_schema


def main() -> int:
    with tempfile.TemporaryDirectory() as temp_dir:
        db_path = Path(temp_dir) / "aifitness.db"
        conn = sqlite3.connect(str(db_path))
        try:
            # Mimic the base tables SQLAlchemy creates before storage migrations.
            conn.execute("CREATE TABLE workout_plans (id INTEGER PRIMARY KEY, schedule_id INTEGER)")
            conn.execute("CREATE TABLE workout_sessions (id INTEGER PRIMARY KEY, schedule_id INTEGER)")
            conn.commit()
        finally:
            conn.close()

        ensure_storage_schema(db_path)
        conn = sqlite3.connect(str(db_path))
        try:
            integrity = conn.execute("PRAGMA integrity_check").fetchone()[0]
            print(f"database=test-fixture:{db_path.name}")
            print(f"integrity={integrity}")
            required = (
                "workout_plans",
                "workout_sessions",
                "workout_session_drafts",
                "workout_schedule_items",
            )
            ok = integrity == "ok"
            for table in required:
                columns = {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}
                has_level = "training_level" in columns
                print(f"{table}.training_level={'OK' if has_level else 'MISSING'}")
                ok = ok and has_level
            level_columns = {row[1] for row in conn.execute("PRAGMA table_info(exercise_level_configs)")}
            required_level = {
                "training_level", "set_count", "rep_target", "rest_seconds",
                "max_idle_seconds", "max_session_seconds", "min_good_rep_ratio",
                "min_quality_score", "min_confidence", "min_stability_score",
                "max_tracking_abort_count",
            }
            level_ok = required_level.issubset(level_columns)
            print(f"exercise_level_configs.multi_criteria={'OK' if level_ok else 'MISSING'}")
            ok = ok and level_ok
            return 0 if ok else 1
        finally:
            conn.close()


if __name__ == "__main__":
    raise SystemExit(main())
