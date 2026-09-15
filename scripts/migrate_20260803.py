"""Idempotent migration for the 03/08/2026 FitMotion reliability patch.

Run from the source root:
    python scripts/migrate_20260803.py

The main Flask application also performs these migrations during startup. This
standalone script is provided for users applying the patch without replacing
their existing SQLite database.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import sqlite3
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from storage import prepare_persistent_database

_, DB, MIGRATION_NOTE = prepare_persistent_database(ROOT)

CRITERIA = {
    "squat": [
        ("Độ sâu squat", "góc gối", "<=", 125, "Squat chưa đủ sâu", "Hạ hông thấp hơn nhưng vẫn giữ kiểm soát."),
        ("Kiểm soát thân người", "góc thân", ">=", 135, "Thân người nghiêng quá mức", "Giữ ngực mở và lưng trung lập."),
    ],
    "pushup": [
        ("Độ sâu hít đất", "góc khuỷu", "<=", 115, "Hạ người chưa đủ thấp", "Gập khuỷu thêm trước khi đẩy lên."),
        ("Đường thẳng cơ thể", "vai-hông-cổ chân", ">=", 140, "Thân người chưa thẳng", "Giữ vai, hông và cổ chân thẳng hàng."),
    ],
    "curl-left": [
        ("Biên độ cuốn tạ", "góc khuỷu", "adaptive", 0, "Gập tay chưa đủ", "Gập đủ biên độ so với góc bắt đầu cá nhân."),
        ("Trở về tư thế bắt đầu", "góc khuỷu", "adaptive", 0, "Chưa trở về gần góc bắt đầu", "Hạ tạ về gần góc đã hiệu chuẩn; không khóa cứng 180 độ."),
        ("Kiểm soát khuỷu", "độ lệch khuỷu", "<=", 45, "Khuỷu tay bị lệch", "Giữ khuỷu gần thân và hạn chế vung vai."),
    ],
    "curl-right": [
        ("Biên độ cuốn tạ", "góc khuỷu", "adaptive", 0, "Gập tay chưa đủ", "Gập đủ biên độ so với góc bắt đầu cá nhân."),
        ("Trở về tư thế bắt đầu", "góc khuỷu", "adaptive", 0, "Chưa trở về gần góc bắt đầu", "Hạ tạ về gần góc đã hiệu chuẩn; không khóa cứng 180 độ."),
        ("Kiểm soát khuỷu", "độ lệch khuỷu", "<=", 45, "Khuỷu tay bị lệch", "Giữ khuỷu gần thân và hạn chế vung vai."),
    ],
}


def columns(connection: sqlite3.Connection, table: str) -> set[str]:
    return {row[1] for row in connection.execute(f"PRAGMA table_info({table})")}


def add_column(connection: sqlite3.Connection, table: str, name: str, definition: str) -> None:
    if name not in columns(connection, table):
        connection.execute(f"ALTER TABLE {table} ADD COLUMN {name} {definition}")


def main() -> int:
    if MIGRATION_NOTE:
        print(f"[INFO] {MIGRATION_NOTE}")
    print(f"[INFO] Database đang dùng: {DB}")
    if not DB.exists():
        print(f"[FAIL] Không tìm thấy database: {DB}")
        return 1

    connection = sqlite3.connect(DB)
    try:
        add_column(connection, "workout_exercises", "model_3d_path", "TEXT DEFAULT ''")
        add_column(connection, "workout_exercises", "model_3d_format", "TEXT DEFAULT 'none'")
        add_column(connection, "workout_exercises", "animation_key", "TEXT DEFAULT ''")
        add_column(connection, "workout_exercises", "intro_model_3d_path", "TEXT DEFAULT ''")
        add_column(connection, "workout_exercises", "intro_model_3d_format", "TEXT DEFAULT 'none'")
        add_column(connection, "workout_exercises", "intro_animation_key", "TEXT DEFAULT ''")
        connection.execute("UPDATE workout_exercises SET model_3d_path=COALESCE(NULLIF(model_3d_path,''),fbx_path,'')")
        connection.execute("""
            UPDATE workout_exercises SET model_3d_format=CASE
                WHEN lower(COALESCE(model_3d_path,'')) LIKE '%.fbx' THEN 'fbx'
                WHEN lower(COALESCE(model_3d_path,'')) LIKE '%.glb' OR lower(COALESCE(model_3d_path,'')) LIKE '%.gltf' THEN 'gltf'
                ELSE 'none' END
        """)
        keys = {"squat": "Squat", "pushup": "PushUp", "curl-left": "BicepCurlLeft", "curl-right": "BicepCurlRight"}
        for slug, key in keys.items():
            connection.execute(
                "UPDATE workout_exercises SET animation_key=? WHERE slug=? AND COALESCE(animation_key,'')=''",
                (key, slug),
            )

        now = datetime.now(timezone.utc).replace(tzinfo=None).isoformat(" ")
        for slug, items in CRITERIA.items():
            row = connection.execute("SELECT id FROM workout_exercises WHERE slug=?", (slug,)).fetchone()
            if not row:
                continue
            exercise_id = row[0]
            existing = {r[0] for r in connection.execute("SELECT title FROM exercise_criteria WHERE exercise_id=?", (exercise_id,))}
            for title, joint, operator, angle, message, advice in items:
                if title in existing:
                    continue
                connection.execute(
                    """INSERT INTO exercise_criteria
                    (exercise_id,title,joint_name,operator,angle_value,message_text,advice_text,audio_path,created_at)
                    VALUES(?,?,?,?,?,?,?,?,?)""",
                    (exercise_id, title, joint, operator, angle, message, advice, "", now),
                )
        connection.commit()
        integrity = connection.execute("PRAGMA integrity_check").fetchone()[0]
        count = connection.execute("SELECT COUNT(*) FROM exercise_criteria").fetchone()[0]
        print(f"[PASS] Migration hoàn tất. integrity={integrity}, criteria={count}")
        return 0 if integrity == "ok" else 1
    finally:
        connection.close()


if __name__ == "__main__":
    raise SystemExit(main())
