"""Explicitly replace the persistent database with an existing FitMotion DB.

Use this only when workout history is still located in an older source folder:
    python scripts/import_existing_database.py "C:\\old-source\\aifitness.db"

The current persistent database is backed up before replacement.
"""
from __future__ import annotations

import argparse
import shutil
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from storage import backup_database, ensure_storage_schema, prepare_persistent_database

REQUIRED_TABLES = {"users", "user_profiles", "workout_exercises", "workout_sessions"}


def validate_source(path: Path) -> tuple[bool, str]:
    if not path.is_file() or path.stat().st_size == 0:
        return False, "File nguồn không tồn tại hoặc rỗng."
    connection = sqlite3.connect(str(path), timeout=30)
    try:
        integrity = connection.execute("PRAGMA integrity_check").fetchone()[0]
        tables = {
            row[0]
            for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
        sessions = int(connection.execute("SELECT COUNT(*) FROM workout_sessions").fetchone()[0]) if "workout_sessions" in tables else 0
    finally:
        connection.close()
    missing = sorted(REQUIRED_TABLES - tables)
    if integrity != "ok" or missing:
        return False, f"integrity={integrity}, thiếu bảng={missing}"
    return True, f"integrity=ok, workout_sessions={sessions}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", help="Đường dẫn tới aifitness.db cũ")
    args = parser.parse_args()

    source = Path(args.source).expanduser().resolve()
    ok, detail = validate_source(source)
    if not ok:
        print(f"[FAIL] Database nguồn không hợp lệ: {detail}")
        return 1

    _, target, migration_note = prepare_persistent_database(ROOT)
    if migration_note:
        print(f"[INFO] {migration_note}")
    print(f"[INFO] Nguồn: {source}")
    print(f"[INFO] Đích:  {target}")
    print(f"[INFO] {detail}")

    if source == target.resolve():
        print("[PASS] Database nguồn đã là database bền vững đang sử dụng.")
        return 0

    if target.exists() and target.stat().st_size > 0:
        backup = backup_database(target)
        print(f"[PASS] Đã backup database hiện tại: {backup}")

    # Remove sidecar files belonging to the old target before replacement.
    for suffix in ("-wal", "-shm"):
        sidecar = Path(str(target) + suffix)
        sidecar.unlink(missing_ok=True)
    shutil.copy2(source, target)
    ensure_storage_schema(target)

    ok, detail = validate_source(target)
    print(f"[{'PASS' if ok else 'FAIL'}] Khôi phục hoàn tất: {detail}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
