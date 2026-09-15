from __future__ import annotations

import argparse
import importlib.util
import sqlite3
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from storage import prepare_persistent_database

REQUIRED_FILES = [
    "app.py",
    "INSTALL_TUAN6_WINDOWS.bat",
    "models.py",
    "workoutlogic.py",
    "ai_temporal.py",
    "quality_scoring.py",
    "expert_system.py",
    "pose_model.py",
    "camera_service.py",
    "standalone_runner.py",
    "evaluation_metrics.py",
    "scripts/evaluate_week6.py",
    "scripts/migrate_20260803.py",
    "scripts/static_validate_week6.py",
    "scripts/check_database_persistence.py",
    "CHECK_DATABASE_WINDOWS.bat",
    "scripts/import_existing_database.py",
    "IMPORT_CURRENT_DATABASE_WINDOWS.bat",
    "yolov8n-pose.pt",
    "aifitness.db",
    "templates/user_workout.html",
    "static/exercise_model_viewer.js",
    "templates/user_analytics.html",
    "data/validation/week6/annotation_template.csv",
    "data/validation/week6/real_video_protocol.csv",
]

REQUIRED_DB_COLUMNS = {
    "total_rep",
    "good_rep",
    "total_error",
    "phase_start_error",
    "phase_middle_error",
    "phase_end_error",
    "keypoint_confidence_avg",
    "quality_score_avg",
    "rep_details_json",
    "error_codes_json",
}

DEPENDENCIES = {
    "flask": "Flask",
    "flask_sqlalchemy": "Flask-SQLAlchemy",
    "cv2": "opencv-python",
    "numpy": "numpy",
    "ultralytics": "ultralytics",
    "torch": "torch",
    "PIL": "pillow",
}


def print_result(ok: bool, label: str, detail: str = "") -> None:
    status = "PASS" if ok else "FAIL"
    suffix = f" - {detail}" if detail else ""
    print(f"[{status}] {label}{suffix}")


def check_files() -> bool:
    ok = True
    for relative in REQUIRED_FILES:
        exists = (ROOT / relative).is_file()
        print_result(exists, f"File {relative}")
        ok = ok and exists
    return ok


def check_dependencies() -> bool:
    ok = True
    for module, package in DEPENDENCIES.items():
        exists = importlib.util.find_spec(module) is not None
        print_result(exists, f"Dependency {package}")
        ok = ok and exists
    return ok


def check_database() -> bool:
    _, database_path, migration_note = prepare_persistent_database(ROOT)
    if migration_note:
        print(f"[INFO] {migration_note}")
    if not database_path.is_file():
        print_result(False, "SQLite database", f"not found: {database_path}")
        return False

    connection = sqlite3.connect(database_path)
    try:
        integrity = connection.execute("PRAGMA integrity_check").fetchone()[0]
        session_columns = {row[1] for row in connection.execute("PRAGMA table_info(workout_sessions)")}
        exercise_columns = {row[1] for row in connection.execute("PRAGMA table_info(workout_exercises)")}
        criterion_count = int(connection.execute("SELECT COUNT(*) FROM exercise_criteria").fetchone()[0])
        draft_table = connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='workout_session_drafts'"
        ).fetchone()
    finally:
        connection.close()

    missing_sessions = sorted(REQUIRED_DB_COLUMNS - session_columns)
    required_model_columns = {"model_3d_path", "model_3d_format", "animation_key"}
    missing_models = sorted(required_model_columns - exercise_columns)
    ok = (
        integrity == "ok"
        and not missing_sessions
        and not missing_models
        and criterion_count >= 10
        and bool(draft_table)
    )
    detail = (
        f"integrity={integrity}, missing_session={missing_sessions}, "
        f"missing_model={missing_models}, criteria={criterion_count}, "
        f"draft_table={bool(draft_table)}, path={database_path}"
    )
    print_result(ok, "SQLite schema and seed data", detail)
    return ok


def run_tests() -> bool:
    command = [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"]
    result = subprocess.run(command, cwd=ROOT, check=False)
    print_result(result.returncode == 0, "Week-6 automated tests")
    return result.returncode == 0


def check_camera() -> bool:
    try:
        import cv2
    except Exception as exc:
        print_result(False, "Webcam", f"OpenCV unavailable: {exc}")
        return False

    candidates = [(0, getattr(cv2, "CAP_DSHOW", 0)), (0, getattr(cv2, "CAP_ANY", 0)), (1, getattr(cv2, "CAP_DSHOW", 0))]
    for index, backend in candidates:
        camera = cv2.VideoCapture(index, backend)
        try:
            if camera.isOpened():
                success, frame = camera.read()
                if success and frame is not None:
                    print_result(True, "Webcam", f"camera index {index}, frame {frame.shape[1]}x{frame.shape[0]}")
                    return True
        finally:
            camera.release()

    print_result(False, "Webcam", "No readable camera found or camera is in use")
    return False


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-dependencies", action="store_true")
    parser.add_argument("--camera", action="store_true", help="Also test the physical webcam")
    args = parser.parse_args()

    checks = [check_files(), check_database(), run_tests()]
    if not args.skip_dependencies:
        checks.append(check_dependencies())
    if args.camera:
        checks.append(check_camera())

    all_ok = all(checks)
    print("\nWEEK-6 PREFLIGHT:", "PASS" if all_ok else "FAIL")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
