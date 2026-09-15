"""Persistent storage helpers for FitMotion AI.

The legacy project stored ``aifitness.db`` beside ``app.py``. That works only
while the exact same source directory is reused. Extracting a new ZIP or
running another copy silently opens another database and makes history appear
lost. This module keeps runtime data in a stable operating-system data folder
and performs a one-time migration from the legacy database.
"""
from __future__ import annotations

import os
import shutil
import sqlite3
from datetime import datetime
from pathlib import Path

APP_DIR_NAME = "FitMotionAI"
DB_FILENAME = "aifitness.db"


def get_persistent_data_dir(base_dir: str | os.PathLike[str]) -> Path:
    override = os.getenv("FITMOTION_DATA_DIR", "").strip()
    if override:
        target = Path(override).expanduser()
    elif os.name == "nt":
        local_app_data = os.getenv("LOCALAPPDATA", "").strip()
        if local_app_data:
            target = Path(local_app_data) / APP_DIR_NAME
        else:
            target = Path.home() / "AppData" / "Local" / APP_DIR_NAME
    else:
        xdg_data_home = os.getenv("XDG_DATA_HOME", "").strip()
        root = Path(xdg_data_home).expanduser() if xdg_data_home else Path.home() / ".local" / "share"
        target = root / APP_DIR_NAME.lower()

    target.mkdir(parents=True, exist_ok=True)
    return target.resolve()


def _legacy_database_candidates(base_dir: Path) -> list[Path]:
    return [
        base_dir / DB_FILENAME,
        base_dir / "instance" / DB_FILENAME,
    ]


def prepare_persistent_database(base_dir: str | os.PathLike[str]) -> tuple[Path, Path, str]:
    """Return ``(data_dir, db_path, migration_note)`` and migrate once.

    The source database is copied, never moved, so an interrupted migration
    cannot destroy the user's old data.
    """
    base = Path(base_dir).resolve()
    data_dir = get_persistent_data_dir(base)
    db_path = data_dir / DB_FILENAME
    note = ""

    if not db_path.exists():
        for candidate in _legacy_database_candidates(base):
            if candidate.exists() and candidate.is_file() and candidate.stat().st_size > 0:
                shutil.copy2(candidate, db_path)
                note = f"Đã sao chép database cũ từ {candidate} sang {db_path}"
                break

    ensure_storage_schema(db_path)
    return data_dir, db_path, note


def ensure_storage_schema(db_path: str | os.PathLike[str]) -> None:
    """Create storage-only tables without requiring Flask to be installed."""
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(str(path), timeout=30)
    try:
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=FULL")
        connection.execute("PRAGMA busy_timeout=30000")
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS workout_session_drafts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                exercise_id INTEGER NOT NULL,
                session_date VARCHAR(20) NOT NULL,
                total_rep INTEGER DEFAULT 0,
                good_rep INTEGER DEFAULT 0,
                total_error INTEGER DEFAULT 0,
                keypoint_confidence_avg REAL DEFAULT 0.0,
                quality_score_avg REAL DEFAULT 0.0,
                training_level VARCHAR(20) NOT NULL DEFAULT 'medium',
                state_json TEXT DEFAULT '{}',
                created_at DATETIME,
                updated_at DATETIME,
                CONSTRAINT uq_workout_draft_user_exercise_date
                    UNIQUE (user_id, exercise_id, session_date),
                FOREIGN KEY(user_id) REFERENCES users (id),
                FOREIGN KEY(exercise_id) REFERENCES workout_exercises (id)
            )
            """
        )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS ix_workout_drafts_user_date "
            "ON workout_session_drafts(user_id, session_date)"
        )

        # Week 9: reusable/shared workout programs. These are durable because
        # they are the source for both community sharing and schedule creation.
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS workout_programs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                owner_user_id INTEGER NOT NULL,
                source_program_id INTEGER,
                title VARCHAR(180) NOT NULL,
                description TEXT DEFAULT '',
                goal VARCHAR(100) DEFAULT '',
                health_focus VARCHAR(120) DEFAULT '',
                source_type VARCHAR(30) NOT NULL DEFAULT 'user',
                visibility VARCHAR(20) NOT NULL DEFAULT 'private',
                moderation_status VARCHAR(20) NOT NULL DEFAULT 'visible',
                profile_snapshot_json TEXT DEFAULT '{}',
                created_at DATETIME,
                updated_at DATETIME,
                FOREIGN KEY(owner_user_id) REFERENCES users (id),
                FOREIGN KEY(source_program_id) REFERENCES workout_programs (id)
            )
            """
        )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS ix_workout_programs_owner_user_id "
            "ON workout_programs(owner_user_id)"
        )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS ix_workout_programs_visibility "
            "ON workout_programs(visibility, moderation_status)"
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS workout_program_items (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                program_id INTEGER NOT NULL,
                exercise_id INTEGER NOT NULL,
                weekday INTEGER NOT NULL CHECK (weekday >= 0 AND weekday <= 6),
                set_count INTEGER NOT NULL DEFAULT 1,
                rep_target INTEGER NOT NULL DEFAULT 10,
                training_level VARCHAR(20) NOT NULL DEFAULT 'medium',
                sort_order INTEGER NOT NULL DEFAULT 0,
                created_at DATETIME,
                FOREIGN KEY(program_id) REFERENCES workout_programs (id),
                FOREIGN KEY(exercise_id) REFERENCES workout_exercises (id)
            )
            """
        )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS ix_workout_program_items_program_id "
            "ON workout_program_items(program_id)"
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS workout_program_favorites (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                program_id INTEGER NOT NULL,
                created_at DATETIME,
                CONSTRAINT uq_program_favorite_user_program UNIQUE (user_id, program_id),
                FOREIGN KEY(user_id) REFERENCES users (id),
                FOREIGN KEY(program_id) REFERENCES workout_programs (id)
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS workout_program_reviews (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                program_id INTEGER NOT NULL,
                rating INTEGER NOT NULL DEFAULT 5 CHECK (rating >= 1 AND rating <= 5),
                comment TEXT DEFAULT '',
                created_at DATETIME,
                updated_at DATETIME,
                CONSTRAINT uq_program_review_user_program UNIQUE (user_id, program_id),
                FOREIGN KEY(user_id) REFERENCES users (id),
                FOREIGN KEY(program_id) REFERENCES workout_programs (id)
            )
            """
        )

        # Durable recurring schedules are storage-critical: create them before
        # Flask boots so extracting a new source copy never loses the user's
        # calendar definition.
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS workout_schedules (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                name VARCHAR(160) NOT NULL,
                start_date VARCHAR(20) NOT NULL,
                end_date VARCHAR(20) NOT NULL,
                status VARCHAR(20) NOT NULL DEFAULT 'active',
                created_at DATETIME,
                updated_at DATETIME,
                FOREIGN KEY(user_id) REFERENCES users (id)
            )
            """
        )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS ix_workout_schedules_user_id "
            "ON workout_schedules(user_id)"
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS workout_schedule_items (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                schedule_id INTEGER NOT NULL,
                exercise_id INTEGER NOT NULL,
                weekday INTEGER NOT NULL CHECK (weekday >= 0 AND weekday <= 6),
                set_count INTEGER NOT NULL DEFAULT 1,
                rep_target INTEGER NOT NULL DEFAULT 10,
                training_level VARCHAR(20) NOT NULL DEFAULT 'medium',
                sort_order INTEGER NOT NULL DEFAULT 0,
                created_at DATETIME,
                FOREIGN KEY(schedule_id) REFERENCES workout_schedules (id),
                FOREIGN KEY(exercise_id) REFERENCES workout_exercises (id)
            )
            """
        )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS ix_workout_schedule_items_schedule_id "
            "ON workout_schedule_items(schedule_id)"
        )

        # Reason 2 multi-criteria: admin-editable workload/session rules per
        # exercise and training level, modeled after the reference thesis'
        # dedicated level configuration instead of hard-coding only a label.
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS exercise_level_configs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                exercise_id INTEGER NOT NULL,
                training_level VARCHAR(20) NOT NULL DEFAULT 'medium',
                set_count INTEGER NOT NULL DEFAULT 2,
                rep_target INTEGER NOT NULL DEFAULT 10,
                rest_seconds INTEGER NOT NULL DEFAULT 90,
                max_idle_seconds INTEGER NOT NULL DEFAULT 600,
                max_session_seconds INTEGER NOT NULL DEFAULT 2400,
                min_good_rep_ratio REAL NOT NULL DEFAULT 0.70,
                min_quality_score REAL NOT NULL DEFAULT 60.0,
                min_confidence REAL NOT NULL DEFAULT 0.60,
                min_stability_score REAL NOT NULL DEFAULT 55.0,
                max_tracking_abort_count INTEGER NOT NULL DEFAULT 4,
                created_at DATETIME,
                updated_at DATETIME,
                CONSTRAINT uq_exercise_level_config UNIQUE (exercise_id, training_level),
                FOREIGN KEY(exercise_id) REFERENCES workout_exercises (id)
            )
            """
        )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS ix_exercise_level_configs_exercise_id "
            "ON exercise_level_configs(exercise_id)"
        )

        def table_columns(table_name: str) -> set[str]:
            return {row[1] for row in connection.execute(f"PRAGMA table_info({table_name})").fetchall()}

        schedule_columns = table_columns("workout_schedules")
        if schedule_columns and "source_program_id" not in schedule_columns:
            connection.execute("ALTER TABLE workout_schedules ADD COLUMN source_program_id INTEGER")
        if table_columns("workout_schedules"):
            connection.execute(
                "CREATE INDEX IF NOT EXISTS ix_workout_schedules_source_program_id "
                "ON workout_schedules(source_program_id)"
            )

        for table_name in ("workout_plans", "workout_sessions"):
            columns = table_columns(table_name)
            if columns and "schedule_id" not in columns:
                connection.execute(f"ALTER TABLE {table_name} ADD COLUMN schedule_id INTEGER")

        session_columns = table_columns("workout_sessions")
        if session_columns:
            session_extensions = {
                "set_count": "INTEGER NOT NULL DEFAULT 1",
                "rep_target": "INTEGER NOT NULL DEFAULT 10",
                "duration_seconds": "INTEGER NOT NULL DEFAULT 0",
                "effectiveness_status": "VARCHAR(30) NOT NULL DEFAULT 'partial'",
                "session_summary_json": "TEXT DEFAULT '{}'",
            }
            for column_name, definition in session_extensions.items():
                if column_name not in table_columns("workout_sessions"):
                    connection.execute(
                        f"ALTER TABLE workout_sessions ADD COLUMN {column_name} {definition}"
                    )

        # Reason 2: workout difficulty is per occurrence/session, not the static
        # catalog difficulty of WorkoutExercise. Existing data is migrated to
        # Medium so old history remains valid.
        for table_name in ("workout_plans", "workout_sessions", "workout_session_drafts", "workout_schedule_items"):
            columns = table_columns(table_name)
            if columns and "training_level" not in columns:
                connection.execute(
                    f"ALTER TABLE {table_name} ADD COLUMN training_level VARCHAR(20) NOT NULL DEFAULT 'medium'"
                )
            if columns or table_columns(table_name):
                connection.execute(
                    f"UPDATE {table_name} SET training_level = 'medium' "
                    "WHERE training_level IS NULL OR TRIM(training_level) = ''"
                )

        if table_columns("workout_plans"):
            connection.execute(
                "CREATE INDEX IF NOT EXISTS ix_workout_plans_schedule_id "
                "ON workout_plans(schedule_id)"
            )

        connection.commit()
    finally:
        connection.close()


def sqlite_uri(db_path: str | os.PathLike[str]) -> str:
    # ``as_posix`` also produces the correct C:/... form for SQLite on Windows.
    return f"sqlite:///{Path(db_path).resolve().as_posix()}"


def backup_database(db_path: str | os.PathLike[str], keep: int = 7) -> Path | None:
    """Create a consistent SQLite backup and retain only the newest files."""
    source = Path(db_path)
    if not source.exists() or source.stat().st_size == 0:
        return None

    backup_dir = source.parent / "backups"
    backup_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    target = backup_dir / f"aifitness-{stamp}.db"

    src_conn = sqlite3.connect(str(source), timeout=30)
    dst_conn = sqlite3.connect(str(target), timeout=30)
    try:
        src_conn.backup(dst_conn)
    finally:
        dst_conn.close()
        src_conn.close()

    backups = sorted(backup_dir.glob("aifitness-*.db"), key=lambda item: item.stat().st_mtime, reverse=True)
    for old_file in backups[max(1, int(keep)):]:
        try:
            old_file.unlink()
        except OSError:
            pass
    return target
