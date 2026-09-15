"""Inspect durable weekly/monthly schedule data without starting Flask."""
from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from storage import prepare_persistent_database


def main() -> int:
    _, database_path, migration_note = prepare_persistent_database(ROOT)
    if migration_note:
        print(f"[INFO] {migration_note}")
    print(f"[INFO] DATABASE_PATH={database_path}")

    connection = sqlite3.connect(str(database_path), timeout=30)
    try:
        integrity = connection.execute("PRAGMA integrity_check").fetchone()[0]
        schedules = connection.execute(
            "SELECT id, user_id, name, start_date, end_date, status "
            "FROM workout_schedules ORDER BY id"
        ).fetchall()
        item_count = connection.execute(
            "SELECT COUNT(*) FROM workout_schedule_items"
        ).fetchone()[0]
        occurrence_count = connection.execute(
            "SELECT COUNT(*) FROM workout_plans WHERE schedule_id IS NOT NULL"
        ).fetchone()[0]
    finally:
        connection.close()

    print(f"[{'PASS' if integrity == 'ok' else 'FAIL'}] integrity={integrity}")
    print(f"[INFO] schedules={len(schedules)}, items={item_count}, occurrences={occurrence_count}")
    for item in schedules:
        print(
            f"[SCHEDULE] id={item[0]} user={item[1]} status={item[5]} "
            f"range={item[3]}..{item[4]} name={item[2]}"
        )
    return 0 if integrity == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
