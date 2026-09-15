"""Show the exact SQLite file used by FitMotion AI and its persisted totals."""
from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from storage import prepare_persistent_database, backup_database


def main() -> int:
    _, database_path, migration_note = prepare_persistent_database(ROOT)
    if migration_note:
        print(f"[INFO] {migration_note}")
    print(f"[INFO] DATABASE_PATH={database_path}")

    connection = sqlite3.connect(str(database_path), timeout=30)
    try:
        integrity = connection.execute("PRAGMA integrity_check").fetchone()[0]
        session_count = connection.execute("SELECT COUNT(*) FROM workout_sessions").fetchone()[0]
        draft_count = connection.execute("SELECT COUNT(*) FROM workout_session_drafts").fetchone()[0]
        totals = connection.execute(
            "SELECT COALESCE(SUM(total_rep),0), COALESCE(SUM(good_rep),0), "
            "COALESCE(SUM(total_error),0) FROM workout_sessions"
        ).fetchone()
        draft_totals = connection.execute(
            "SELECT COALESCE(SUM(total_rep),0), COALESCE(SUM(good_rep),0), "
            "COALESCE(SUM(total_error),0) FROM workout_session_drafts"
        ).fetchone()
        schedule_count = connection.execute(
            "SELECT COUNT(*) FROM workout_schedules WHERE status='active'"
        ).fetchone()[0]
        schedule_item_count = connection.execute(
            "SELECT COUNT(*) FROM workout_schedule_items"
        ).fetchone()[0]
        scheduled_occurrence_count = connection.execute(
            "SELECT COUNT(*) FROM workout_plans WHERE schedule_id IS NOT NULL"
        ).fetchone()[0]
    finally:
        connection.close()

    print(f"[{'PASS' if integrity == 'ok' else 'FAIL'}] integrity={integrity}")
    print(f"[INFO] finalized_sessions={session_count}, totals={totals}")
    print(f"[INFO] resumable_drafts={draft_count}, draft_totals={draft_totals}")
    print(
        f"[INFO] active_schedules={schedule_count}, "
        f"schedule_items={schedule_item_count}, "
        f"materialized_occurrences={scheduled_occurrence_count}"
    )
    backup = backup_database(database_path)
    print(f"[PASS] backup={backup}")
    return 0 if integrity == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
