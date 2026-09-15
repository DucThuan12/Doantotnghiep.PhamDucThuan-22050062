"""Add the optional intro-3D columns to an existing FitMotion SQLite database.

The application also performs these additive migrations on startup. This script
is provided for manual verification on Windows before the Week 8 demo.
"""
from __future__ import annotations

import os
import sqlite3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB = ROOT / "instance" / "aifitness.db"
DB = Path(os.environ.get("FITMOTION_DB_PATH", str(DEFAULT_DB))).expanduser().resolve()

COLUMNS = [
    ("intro_model_3d_path", "TEXT DEFAULT ''"),
    ("intro_model_3d_format", "TEXT DEFAULT 'none'"),
    ("intro_animation_key", "TEXT DEFAULT ''"),
]


def main() -> int:
    if not DB.exists():
        print(f"[FAIL] Không tìm thấy database: {DB}")
        return 1
    con = sqlite3.connect(DB)
    try:
        cols = {row[1] for row in con.execute("PRAGMA table_info(workout_exercises)")}
        for name, definition in COLUMNS:
            if name not in cols:
                con.execute(f"ALTER TABLE workout_exercises ADD COLUMN {name} {definition}")
                print(f"[ADD] workout_exercises.{name}")
            else:
                print(f"[OK] workout_exercises.{name}")
        con.commit()
        result = con.execute("PRAGMA integrity_check").fetchone()[0]
        print(f"[PASS] integrity={result}")
        return 0 if result == "ok" else 1
    finally:
        con.close()


if __name__ == "__main__":
    raise SystemExit(main())
