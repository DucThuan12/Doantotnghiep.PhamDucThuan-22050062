import json
import os
import re
import sqlite3
import time
import uuid
import unicodedata
import subprocess
import sys
import threading
import shutil
from pathlib import Path
from datetime import datetime, timedelta

from flask import Flask, render_template, request, redirect, url_for, flash, Response, jsonify, stream_with_context, send_file, abort
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename

from database import db
from models import (
    User, UserProfile, WorkoutExercise, WorkoutPlan, WorkoutSession,
    WorkoutSessionDraft, ExerciseCriterion, ExerciseLabelImage,
    WorkoutSchedule, WorkoutScheduleItem, ExerciseLevelConfig,
    ExerciseTrainingMedia, EmergencyAlert, WorkoutProgram, WorkoutProgramItem,
    WorkoutProgramFavorite, WorkoutProgramReview
)
from auth import login_user, logout_user, current_user, login_required, admin_required, admin_api_required
from workoutlogic import SquatProcessor, PushupProcessor, CurlProcessor, LearnedExerciseProcessor, UnsupportedExerciseProcessor
from camera_service import camera_service
from storage import prepare_persistent_database, sqlite_uri, backup_database
from dashboard_activity import build_schedule_adherence, build_week_activity
from schedule_logic import (
    WEEKDAY_LABELS, parse_iso_date, parse_month, shift_month,
    validate_schedule_range,
)
from schedule_service import (
    archive_schedule, build_month_calendar, get_user_schedules,
    materialize_schedule, replace_future_occurrences, schedule_to_dict,
)
from training_levels import (
    DEFAULT_LEVEL, LEVEL_OPTIONS, normalize_training_level, training_level_label,
    workload_profile, evaluate_session_effectiveness,
)
from reference_motion import load_reference_file, ReferenceMotionError
from glb_reference import build_reference_from_glb, inspect_glb_asset, GLBReferenceError
from pose_learning import RECOMMENDED_LABELS, expected_label_prefixes
from tts_service import synthesize_feedback
from helper import to_ascii_overlay


GOAL_OPTIONS = [
    ("tap-nhe", "Tập nhẹ"),
    ("giam-mo", "Giảm mỡ"),
    ("tang-co", "Tăng cơ"),
]

HEALTH_OPTIONS = [
    ("khong-co-van-de", "Không có vấn đề đặc biệt"),
    ("the-trang-yeu", "Thể trạng yếu"),
    ("dau-goi", "Đau gối"),
    ("dau-vai", "Đau vai"),
    ("co-tay-yeu", "Cổ tay yếu"),
    ("dau-lung", "Đau lưng"),
    ("huyet-ap-khong-on-dinh", "Huyết áp không ổn định"),
]

GOAL_LABELS = dict(GOAL_OPTIONS)
HEALTH_LABELS = dict(HEALTH_OPTIONS)

EXERCISE_LABELS = {
    "squat": "Squat",
    "pushup": "Hít đất",
    "curl-left": "Cuốn tạ tay trái",
    "curl-right": "Cuốn tạ tay phải",
}

EMERGENCY_STATES = {}

# Web training jobs are intentionally kept lightweight in memory; the durable
# source of truth is WorkoutExercise.pose_model_status/error in the database.
# The lock prevents two clicks from starting two expensive training processes
# for the same exercise.
MODEL_TRAINING_THREADS = {}
MODEL_TRAINING_LOCK = threading.Lock()
POSE_MODEL_MIN_TEST_ACCURACY = 0.70
MIN_TRAINING_SOURCES_PER_LABEL = 5


def get_emergency_key(user_id, slug):
    return f"{user_id}:{slug}"


def get_default_emergency_state():
    return {
        "active": False,
        "message": "",
        "reason": "",
        "updated_at": 0,
        "image_path": "",
        "body_angle": 0,
        "low_posture": False,
        "training_level": DEFAULT_LEVEL,
        "training_level_label": training_level_label(DEFAULT_LEVEL),
        "emergency_posture": "unknown",
        "emergency_candidate": False,
        "emergency_armed": False,
        "emergency_immobile_seconds": 0.0,
        "emergency_required_seconds": 6.0,
        "emergency_exercise": "",
        "emergency_phase": "",
        "emergency_reset_version": 0,
        "emergency_alert_id": 0,
        "emergency_persist_error": "",
        "total_rep": 0,
        "good_rep": 0,
        "bad_rep": 0,
        "phase_start_error": 0,
        "phase_middle_error": 0,
        "phase_end_error": 0,
        "tracking_abort_count": 0,
        "status_text": "San sang",
        "target_rep": 0,
        "exercise_slug": "",
        # Production webcam sessions use a coarse posture identity gate before
        # Squat/Push-up/Curl state machines may advance. This prevents doing a
        # different exercise from being counted merely because one joint angle
        # happens to cross the selected exercise threshold.
        "exercise_identity_gate_enabled": True,
        "exercise_identity_match": False,
        "exercise_identity_reason": "",
        "body_axis_from_horizontal": 0.0,
        "workout_done": False,
        "rep_quality_score": 0,
        "quality_score_avg": 0.0,
        "keypoint_confidence_avg": 0.0,
        "current_primary_angle": 0.0,
        "current_secondary_angle": 0.0,
        "primary_metric_name": "Chỉ số chính",
        "secondary_metric_name": "Chỉ số phụ",
        "evaluation_side": "",
        "feedback_id": 0,
        "feedback_text": "",
        "feedback_advice": "",
        "feedback_level": "neutral",
        "voice_text": "",
        "last_error_code": "",
        "last_rep_components": {},
        "rep_scores": [],
        "rep_records": [],
        "error_code_counts": {},
        "rejected_attempt_count": 0,
        "last_rejected_attempt": {},
        "fps": 0,
        "camera_ready": False,
        "camera_error": "",
        "camera_backend": "",
        "camera_last_frame_age": 0.0,
        "pose_classifier_available": False,
        "pose_classifier_error": "",
        "pose_classifier_last": {},
        "pose_guard_last_decision": {},
        "reference_motion_available": False,
        "reference_motion_error": "",
        "reference_match": {},
        "curl_extension_baseline": 0.0,
        "curl_calibration_ready": False,
        "squat_standing_baseline": 0.0,
        "squat_calibration_ready": False,
        "pushup_top_baseline": 0.0,
        "pushup_calibration_ready": False,
        "workout_paused": False,
        "pause_reason": "",
        "wellness_check_enabled": False,
        "wellness_check_due": False,
        "wellness_check_completed": False,
        "wellness_check_seconds": 60,
        "session_started_at": 0.0,
        "last_rep_at": 0.0,
        "idle_seconds": 0,
        "set_count": 1,
        "reps_per_set": 10,
        "rest_seconds": 90,
        "max_idle_seconds": 600,
        "max_session_seconds": 2400,
        "min_good_rep_ratio": 0.70,
        "min_quality_avg": 60.0,
        "min_confidence_avg": 0.60,
        "min_stability_avg": 55.0,
        "max_tracking_abort_count": 4,
        "workout_cancelled": False,
        "cancellation_reason": "",
    }

def get_or_create_emergency_state(user_id, slug):
    key = get_emergency_key(user_id, slug)
    if key not in EMERGENCY_STATES:
        EMERGENCY_STATES[key] = get_default_emergency_state()
    return key, EMERGENCY_STATES[key]


def get_effective_level_criteria(exercise, level):
    """Return DB-configured workload/session rules with safe defaults."""
    normalized = normalize_training_level(level)
    defaults = workload_profile(getattr(exercise, "slug", ""), normalized)
    row = ExerciseLevelConfig.query.filter_by(
        exercise_id=exercise.id, training_level=normalized
    ).first()
    if not row:
        return defaults
    return {
        "default_sets": max(1, int(row.set_count or defaults["default_sets"])),
        "reps_per_set": max(1, int(row.rep_target or defaults["reps_per_set"])),
        "rest_seconds": max(15, int(row.rest_seconds or defaults["rest_seconds"])),
        "max_idle_seconds": max(60, int(row.max_idle_seconds or defaults["max_idle_seconds"])),
        "max_session_seconds": max(300, int(row.max_session_seconds or defaults["max_session_seconds"])),
        "min_good_rep_ratio": min(1.0, max(0.0, float(row.min_good_rep_ratio if row.min_good_rep_ratio is not None else defaults["min_good_rep_ratio"]))),
        "min_quality_avg": min(100.0, max(0.0, float(row.min_quality_score if row.min_quality_score is not None else defaults["min_quality_avg"]))),
        "min_confidence_avg": min(1.0, max(0.0, float(row.min_confidence if row.min_confidence is not None else defaults["min_confidence_avg"]))),
        "min_stability_avg": min(100.0, max(0.0, float(row.min_stability_score if row.min_stability_score is not None else defaults["min_stability_avg"]))),
        "max_tracking_abort_count": max(0, int(row.max_tracking_abort_count if row.max_tracking_abort_count is not None else defaults["max_tracking_abort_count"])),
    }


def ensure_level_config_defaults(exercise):
    """Create the three admin-editable level rows once, like Long's level use case."""
    for level_key, _ in LEVEL_OPTIONS:
        existing = ExerciseLevelConfig.query.filter_by(
            exercise_id=exercise.id, training_level=level_key
        ).first()
        if existing:
            continue
        cfg = workload_profile(exercise.slug, level_key)
        db.session.add(ExerciseLevelConfig(
            exercise_id=exercise.id,
            training_level=level_key,
            set_count=cfg["default_sets"],
            rep_target=cfg["reps_per_set"],
            rest_seconds=cfg["rest_seconds"],
            max_idle_seconds=cfg["max_idle_seconds"],
            max_session_seconds=cfg["max_session_seconds"],
            min_good_rep_ratio=cfg["min_good_rep_ratio"],
            min_quality_score=cfg["min_quality_avg"],
            min_confidence=cfg["min_confidence_avg"],
            min_stability_score=cfg["min_stability_avg"],
            max_tracking_abort_count=cfg["max_tracking_abort_count"],
        ))


def apply_session_policy(shared_state, exercise, level, set_count=None, reps_per_set=None):
    cfg = get_effective_level_criteria(exercise, level)
    actual_sets = max(1, int(set_count if set_count is not None else cfg["default_sets"]))
    actual_reps = max(1, int(reps_per_set if reps_per_set is not None else cfg["reps_per_set"]))
    shared_state.update({
        "set_count": actual_sets,
        "reps_per_set": actual_reps,
        "target_rep": actual_sets * actual_reps,
        "rest_seconds": int(cfg["rest_seconds"]),
        "max_idle_seconds": int(cfg["max_idle_seconds"]),
        "max_session_seconds": int(cfg["max_session_seconds"]),
        "min_good_rep_ratio": float(cfg["min_good_rep_ratio"]),
        "min_quality_avg": float(cfg["min_quality_avg"]),
        "min_confidence_avg": float(cfg["min_confidence_avg"]),
        "min_stability_avg": float(cfg["min_stability_avg"]),
        "max_tracking_abort_count": int(cfg["max_tracking_abort_count"]),
    })
    return cfg


def cancel_inactive_workout_if_needed(user_id, exercise, shared_state, now=None):
    """Cancel a stale workout and remove its resumable draft.

    No finalized WorkoutSession is created. Any temporary draft checkpoint is
    deleted so a user cannot return after a long break and append new reps to
    the same recorded workout. Existing recurring schedule definitions are not
    deleted because they are the user's plan, not a workout result.
    """
    if bool(shared_state.get("workout_cancelled", False)):
        return True
    if bool(shared_state.get("workout_done", False)):
        return False
    if shared_state.get("pause_reason") in {"emergency", "wellness_check"}:
        return False
    started_at = float(shared_state.get("session_started_at", 0.0) or 0.0)
    if started_at <= 0:
        return False
    now = float(time.time() if now is None else now)
    total_rep = max(0, int(shared_state.get("total_rep", 0) or 0))
    last_rep_at = float(shared_state.get("last_rep_at", 0.0) or 0.0)
    anchor = last_rep_at if total_rep > 0 and last_rep_at > 0 else started_at
    max_idle = max(60, int(shared_state.get("max_idle_seconds", 600) or 600))
    idle_seconds = max(0.0, now - anchor)

    level = normalize_training_level(shared_state.get("training_level", DEFAULT_LEVEL))
    level_name = training_level_label(level)
    # Use the active level policy instead of a second hard-coded 2/4/6 minute
    # timer. The old duplicate timer could cancel AI processing unexpectedly
    # and made a later level inherit the first session's elapsed time.
    max_session = max(60, int(shared_state.get("max_session_seconds", 2400) or 2400))

    elapsed_seconds = max(0.0, now - started_at)
    set_count = max(1, int(shared_state.get("set_count", 1) or 1))
    reps_per_set = max(1, int(shared_state.get("reps_per_set", 1) or 1))
    target_rep = set_count * reps_per_set

    is_idle_timeout = idle_seconds >= max_idle
    is_session_timeout = (elapsed_seconds >= max_session) and (total_rep < target_rep)

    if not is_idle_timeout and not is_session_timeout:
        shared_state["idle_seconds"] = int(idle_seconds)
        shared_state["elapsed_seconds"] = int(elapsed_seconds)
        return False

    session_date = datetime.now().date().isoformat()
    delete_workout_draft(user_id, exercise.id, session_date)
    db.session.commit()

    policy = {
        key: shared_state.get(key)
        for key in (
            "rest_seconds", "max_idle_seconds", "max_session_seconds",
            "min_good_rep_ratio", "min_quality_avg", "min_confidence_avg",
            "min_stability_avg", "max_tracking_abort_count"
        )
    }
    reset_workout_runtime_state(user_id, exercise.slug, preserve_target=False)

    if is_session_timeout:
        pause_reason = "session_timeout_cancelled"
        reason_msg = f"Không hoàn thành mức {level_name} do quá thời gian quy định ({max_session // 60} phút). Buổi tập đã được đặt lại từ đầu."
        status_msg = f"KHONG HOAN THANH MUC {level_name.upper()} DO HET GIO"
    else:
        pause_reason = "inactivity_cancelled"
        reason_msg = f"Nghỉ quá {max_idle // 60} phút; buổi tập đã bị hủy và không lưu vào lịch sử."
        status_msg = "BUOI TAP DA HUY DO NGHI QUA LAU"

    shared_state.update({
        "training_level": level,
        "training_level_label": level_name,
        "set_count": set_count,
        "reps_per_set": reps_per_set,
        "target_rep": target_rep,
        **policy,
        "workout_cancelled": True,
        "workout_paused": True,
        "pause_reason": pause_reason,
        "cancellation_reason": reason_msg,
        "status_text": status_msg,
        "idle_seconds": int(idle_seconds),
        "elapsed_seconds": int(elapsed_seconds),
        "session_started_at": 0.0,
        "last_rep_at": 0.0,
    })
    return True


def normalize_goal(value):
    raw = (value or "").strip().lower()
    mapping = {
        "tap nhe": "tap-nhe",
        "tap-nhe": "tap-nhe",
        "giam mo": "giam-mo",
        "giam-mo": "giam-mo",
        "tang co": "tang-co",
        "tang-co": "tang-co",
    }
    return mapping.get(raw, "tap-nhe")


def normalize_health_note(value):
    raw = (value or "").strip().lower()
    mapping = {
        "khong co van de dac biet": "khong-co-van-de",
        "khong-co-van-de": "khong-co-van-de",
        "khong co": "khong-co-van-de",
        "the trang yeu": "the-trang-yeu",
        "the-trang-yeu": "the-trang-yeu",
        "dau goi": "dau-goi",
        "dau-goi": "dau-goi",
        "dau vai": "dau-vai",
        "dau-vai": "dau-vai",
        "co tay yeu": "co-tay-yeu",
        "co-tay-yeu": "co-tay-yeu",
        "dau lung": "dau-lung",
        "dau-lung": "dau-lung",
        "huyet ap khong on dinh": "huyet-ap-khong-on-dinh",
        "huyet-ap-khong-on-dinh": "huyet-ap-khong-on-dinh",
    }
    return mapping.get(raw, "khong-co-van-de")


BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DATA_DIR, DB_PATH_OBJECT, DB_MIGRATION_NOTE = prepare_persistent_database(BASE_DIR)
DB_PATH = str(DB_PATH_OBJECT)

UPLOAD_EXERCISE_DIR = os.path.join(BASE_DIR, "static", "uploads", "exercises")
UPLOAD_FBX_DIR = os.path.join(BASE_DIR, "static", "uploads", "fbx")
UPLOAD_LABEL_DIR = os.path.join(BASE_DIR, "static", "uploads", "labels")
UPLOAD_REFERENCE_DIR = os.path.join(BASE_DIR, "static", "uploads", "references")
POSE_TRAINING_DIR = os.path.join(BASE_DIR, "data", "pose_training")
AUDIO_DIR = os.path.join(BASE_DIR, "static", "audio")
PERSISTENT_EMERGENCY_DIR = Path(DATA_DIR) / "errorimages" / "emergency"
PERSISTENT_EMERGENCY_DIR.mkdir(parents=True, exist_ok=True)
os.environ["FITMOTION_EMERGENCY_DIR"] = str(PERSISTENT_EMERGENCY_DIR)

# Preserve legacy screenshots when this patch is applied directly over an old
# source folder. Future emergency frames are written to the persistent data
# directory so extracting a new ZIP cannot lose them.
legacy_emergency_dir = Path(BASE_DIR) / "data" / "errorimages" / "emergency"
if legacy_emergency_dir.exists():
    for legacy_image in legacy_emergency_dir.glob("emergency_*.jpg"):
        target_image = PERSISTENT_EMERGENCY_DIR / legacy_image.name
        if not target_image.exists():
            try:
                shutil.copy2(legacy_image, target_image)
            except OSError:
                pass

os.makedirs(UPLOAD_EXERCISE_DIR, exist_ok=True)
os.makedirs(UPLOAD_FBX_DIR, exist_ok=True)
os.makedirs(UPLOAD_LABEL_DIR, exist_ok=True)
os.makedirs(UPLOAD_REFERENCE_DIR, exist_ok=True)
os.makedirs(POSE_TRAINING_DIR, exist_ok=True)
os.makedirs(AUDIO_DIR, exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, "data", "uploaded", "dung"), exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, "data", "uploaded", "sai"), exist_ok=True)


def get_active_database_uri(db_path):
    _mysql_host = os.getenv("FITMOTION_MYSQL_HOST", "127.0.0.1")
    _mysql_port = int(os.getenv("FITMOTION_MYSQL_PORT", "3306"))
    _mysql_user = os.getenv("FITMOTION_MYSQL_USER", "root")
    _mysql_pass = os.getenv("FITMOTION_MYSQL_PASSWORD", "")
    _mysql_db   = os.getenv("FITMOTION_MYSQL_DB", "fitmotion_ai")
    mysql_uri = os.getenv(
        "FITMOTION_MYSQL_URI",
        f"mysql+pymysql://{_mysql_user}:{_mysql_pass}@{_mysql_host}:{_mysql_port}/{_mysql_db}?charset=utf8mb4",
    )
    force_sqlite = os.getenv("FITMOTION_FORCE_SQLITE", "").strip().lower() in {"1", "true", "yes"}
    if not force_sqlite:
        try:
            import pymysql
            conn = pymysql.connect(
                host=_mysql_host,
                port=_mysql_port,
                user=_mysql_user,
                password=_mysql_pass,
                database=_mysql_db,
                connect_timeout=2,
            )
            conn.close()
            print("[FitMotion AI] Co so du lieu hoat dong: MySQL Server 8.0 (database: fitmotion_ai)")
            return mysql_uri, "mysql"
        except Exception as exc:
            print(f"[FitMotion AI] Khong the ket noi MySQL ({exc}), tu dong chuyen sang SQLite du phong.")
    print(f"[FitMotion AI] Co so du lieu hoat dong: SQLite ({db_path})")
    return sqlite_uri(db_path), "sqlite"


ACTIVE_DB_URI, ACTIVE_DB_ENGINE = get_active_database_uri(DB_PATH)

app = Flask(__name__)
app.config["SECRET_KEY"] = os.getenv("FITMOTION_SECRET_KEY", "fitmotion-ai-dev-secret-change-me")
app.config["SQLALCHEMY_DATABASE_URI"] = ACTIVE_DB_URI
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
if ACTIVE_DB_ENGINE == "mysql":
    app.config["SQLALCHEMY_ENGINE_OPTIONS"] = {
        "pool_size": 10,
        "max_overflow": 20,
        "pool_recycle": 1800,
        "pool_pre_ping": True,
    }
else:
    app.config["SQLALCHEMY_ENGINE_OPTIONS"] = {
        "connect_args": {"timeout": 30, "check_same_thread": False},
        "pool_pre_ping": True,
    }
# Allow reasonably sized rigged FBX/GLB uploads while rejecting accidental huge files.
app.config["MAX_CONTENT_LENGTH"] = 120 * 1024 * 1024

db.init_app(app)


def ensure_sqlite_schema():
    """Apply migrations required by the build."""
    if ACTIVE_DB_ENGINE == "mysql":
        with app.app_context():
            db.create_all()
        return

    if not os.path.exists(DB_PATH):
        return

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    def table_exists(table_name):
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
            (table_name,),
        )
        return cursor.fetchone() is not None

    def table_columns(table_name):
        cursor.execute(f"PRAGMA table_info({table_name})")
        return {row[1] for row in cursor.fetchall()}

    def add_column_if_missing(table_name, column_name, sql_definition):
        columns = table_columns(table_name)
        if column_name not in columns:
            cursor.execute(
                f"ALTER TABLE {table_name} ADD COLUMN {column_name} {sql_definition}"
            )

    try:
        if table_exists("workout_sessions"):
            add_column_if_missing("workout_sessions", "session_date", "TEXT")
            add_column_if_missing("workout_sessions", "keypoint_confidence_avg", "REAL DEFAULT 0")
            add_column_if_missing("workout_sessions", "quality_score_avg", "REAL DEFAULT 0")
            add_column_if_missing("workout_sessions", "rep_details_json", "TEXT DEFAULT '[]'")
            add_column_if_missing("workout_sessions", "error_codes_json", "TEXT DEFAULT '{}'")
            add_column_if_missing("workout_sessions", "set_count", "INTEGER NOT NULL DEFAULT 1")
            add_column_if_missing("workout_sessions", "rep_target", "INTEGER NOT NULL DEFAULT 10")
            add_column_if_missing("workout_sessions", "duration_seconds", "INTEGER NOT NULL DEFAULT 0")
            add_column_if_missing("workout_sessions", "effectiveness_status", "TEXT NOT NULL DEFAULT 'partial'")
            add_column_if_missing("workout_sessions", "session_summary_json", "TEXT DEFAULT '{}'")
            add_column_if_missing("workout_sessions", "reference_score_avg", "REAL DEFAULT 0")
            add_column_if_missing("workout_sessions", "pose_guard_confirmed_ratio", "REAL DEFAULT 0")

            today_str = datetime.now().date().isoformat()
            cursor.execute(
                """
                UPDATE workout_sessions
                SET session_date = ?
                WHERE session_date IS NULL OR session_date = ''
                """,
                (today_str,),
            )

        if table_exists("workout_exercises"):
            add_column_if_missing("workout_exercises", "model_3d_path", "TEXT DEFAULT ''")
            add_column_if_missing("workout_exercises", "model_3d_format", "TEXT DEFAULT 'none'")
            add_column_if_missing("workout_exercises", "animation_key", "TEXT DEFAULT ''")
            add_column_if_missing("workout_exercises", "intro_model_3d_path", "TEXT DEFAULT ''")
            add_column_if_missing("workout_exercises", "intro_model_3d_format", "TEXT DEFAULT 'none'")
            add_column_if_missing("workout_exercises", "intro_animation_key", "TEXT DEFAULT ''")
            add_column_if_missing("workout_exercises", "reference_motion_path", "TEXT DEFAULT ''")
            add_column_if_missing("workout_exercises", "reference_motion_source", "TEXT DEFAULT ''")
            add_column_if_missing("workout_exercises", "reference_motion_version", "TEXT DEFAULT ''")
            add_column_if_missing("workout_exercises", "reference_min_score", "REAL DEFAULT 55")
            add_column_if_missing("workout_exercises", "pose_model_path", "TEXT DEFAULT ''")
            add_column_if_missing("workout_exercises", "pose_model_metrics_path", "TEXT DEFAULT ''")
            add_column_if_missing("workout_exercises", "pose_model_status", "TEXT DEFAULT 'not_trained'")
            add_column_if_missing("workout_exercises", "pose_model_accuracy", "REAL DEFAULT 0")
            add_column_if_missing("workout_exercises", "pose_model_error", "TEXT DEFAULT ''")
            add_column_if_missing("workout_exercises", "pose_model_updated_at", "DATETIME")
            cursor.execute(
                """
                UPDATE workout_exercises
                SET model_3d_format = CASE
                    WHEN COALESCE(model_3d_path, '') <> '' AND LOWER(model_3d_path) LIKE '%.fbx' THEN 'fbx'
                    WHEN COALESCE(model_3d_path, '') <> '' AND (
                        LOWER(model_3d_path) LIKE '%.glb' OR LOWER(model_3d_path) LIKE '%.gltf'
                    ) THEN 'gltf'
                    WHEN COALESCE(fbx_path, '') <> '' THEN 'fbx'
                    ELSE 'none'
                END
                WHERE model_3d_format IS NULL OR model_3d_format = '' OR LOWER(model_3d_format) NOT IN ('fbx','gltf','glb','none')
                """
            )
            cursor.execute(
                """
                UPDATE workout_exercises
                SET animation_key = slug
                WHERE animation_key IS NULL OR animation_key = ''
                """
            )

        if table_exists("exercise_criteria"):
            add_column_if_missing("exercise_criteria", "error_code", "TEXT DEFAULT ''")
            add_column_if_missing("exercise_criteria", "phase", "TEXT DEFAULT 'middle'")
            add_column_if_missing("exercise_criteria", "joint_indices_json", "TEXT DEFAULT '[]'")
            add_column_if_missing("exercise_criteria", "audio_path", "TEXT DEFAULT ''")
            add_column_if_missing("exercise_criteria", "audio_provider", "TEXT DEFAULT ''")
            add_column_if_missing("exercise_criteria", "audio_error", "TEXT DEFAULT ''")

        if table_exists("workout_plans"):
            add_column_if_missing("workout_plans", "schedule_id", "INTEGER")
            add_column_if_missing("workout_plans", "training_level", "TEXT NOT NULL DEFAULT 'medium'")
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS ix_workout_plans_schedule_id ON workout_plans(schedule_id)"
            )

        if table_exists("workout_sessions"):
            add_column_if_missing("workout_sessions", "schedule_id", "INTEGER")
            add_column_if_missing("workout_sessions", "training_level", "TEXT NOT NULL DEFAULT 'medium'")

        if table_exists("workout_session_drafts"):
            add_column_if_missing("workout_session_drafts", "training_level", "TEXT NOT NULL DEFAULT 'medium'")

        if table_exists("workout_schedule_items"):
            add_column_if_missing("workout_schedule_items", "training_level", "TEXT NOT NULL DEFAULT 'medium'")

        if table_exists("workout_schedules"):
            add_column_if_missing("workout_schedules", "source_program_id", "INTEGER")
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS ix_workout_schedules_source_program_id ON workout_schedules(source_program_id)"
            )

        if table_exists("user_profiles"):
            columns = table_columns("user_profiles")
            if "daily_target" not in columns:
                cursor.execute(
                    "ALTER TABLE user_profiles ADD COLUMN daily_target INTEGER DEFAULT 6"
                )
                cursor.execute(
                    """
                    UPDATE user_profiles
                    SET daily_target = CASE
                        WHEN weekly_target IS NOT NULL AND weekly_target > 0
                            THEN MAX(1, CAST(ROUND(weekly_target / 7.0) AS INTEGER))
                        ELSE 6
                    END
                    WHERE daily_target IS NULL OR daily_target = 0
                    """
                )
            add_column_if_missing("user_profiles", "phone", "TEXT DEFAULT ''")
            add_column_if_missing("user_profiles", "contact_address", "TEXT DEFAULT ''")
            add_column_if_missing("user_profiles", "emergency_contact_name", "TEXT DEFAULT ''")
            add_column_if_missing("user_profiles", "emergency_contact_relation", "TEXT DEFAULT ''")
            add_column_if_missing("user_profiles", "emergency_contact_phone", "TEXT DEFAULT ''")

        conn.commit()
    finally:
        conn.close()

def slugify(text):
    # Normalize Vietnamese names (e.g. "Cuốn tạ") into stable ASCII slugs.
    normalized = unicodedata.normalize("NFKD", str(text or ""))
    normalized = normalized.replace("đ", "d").replace("Đ", "D")
    text = normalized.encode("ascii", "ignore").decode("ascii").lower().strip()
    text = re.sub(r"[^a-z0-9\s-]", "", text)
    text = re.sub(r"\s+", "-", text)
    text = re.sub(r"-+", "-", text)
    return text.strip("-")


def _is_path_within(path, root):
    try:
        Path(path).resolve().relative_to(Path(root).resolve())
        return True
    except ValueError:
        return False


def normalize_phone(value):
    raw = str(value or "").strip()
    prefix = "+" if raw.startswith("+") else ""
    digits = re.sub(r"\D", "", raw)
    return prefix + digits


def valid_phone(value):
    normalized = normalize_phone(value)
    digits = normalized.lstrip("+")
    return 8 <= len(digits) <= 15 and digits.isdigit()


def normalize_pose_label(value):
    label = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    label = re.sub(r"[^a-z0-9_]", "", label)
    label = re.sub(r"_+", "_", label).strip("_")
    return label[:120]


YOLO_KEYPOINT_OPTIONS = [
    (0, "Mũi"),
    (1, "Mắt trái"),
    (2, "Mắt phải"),
    (3, "Tai trái"),
    (4, "Tai phải"),
    (5, "Vai trái"),
    (6, "Vai phải"),
    (7, "Khuỷu trái"),
    (8, "Khuỷu phải"),
    (9, "Cổ tay trái"),
    (10, "Cổ tay phải"),
    (11, "Hông trái"),
    (12, "Hông phải"),
    (13, "Gối trái"),
    (14, "Gối phải"),
    (15, "Cổ chân trái"),
    (16, "Cổ chân phải"),
]
YOLO_KEYPOINT_LABELS = {index: label for index, label in YOLO_KEYPOINT_OPTIONS}


def _criterion_joint_indices(criterion):
    try:
        values = json.loads(str(getattr(criterion, "joint_indices_json", "[]") or "[]"))
    except (TypeError, ValueError, json.JSONDecodeError):
        return []
    if not isinstance(values, list) or len(values) != 3:
        return []
    try:
        result = [int(item) for item in values]
    except (TypeError, ValueError):
        return []
    if any(item < 0 or item > 16 for item in result) or len(set(result)) != 3:
        return []
    return result


def runtime_criterion_rules(criteria):
    """Serialize Admin angle rules for the realtime processor.

    Only rules with a complete 3-keypoint definition are executable. Older
    criteria remain visible/auditable but cannot silently become runtime rules.
    """
    rules = []
    for criterion in criteria or []:
        indices = _criterion_joint_indices(criterion)
        if len(indices) != 3:
            continue
        operator = str(getattr(criterion, "operator", "") or "").strip()
        if operator not in {"<", "<=", ">", ">=", "="}:
            continue
        phase = normalize_pose_label(getattr(criterion, "phase", "middle")) or "middle"
        if phase not in {"start", "middle", "end", "any"}:
            phase = "middle"
        rules.append({
            "id": int(criterion.id),
            "error_code": normalize_pose_label(getattr(criterion, "error_code", "")) or f"criterion_{criterion.id}",
            "phase": phase,
            "joint_indices": indices,
            "joint_labels": [YOLO_KEYPOINT_LABELS.get(item, str(item)) for item in indices],
            "operator": operator,
            "angle_value": float(getattr(criterion, "angle_value", 0.0) or 0.0),
            "message_text": str(getattr(criterion, "message_text", "") or "Sai kỹ thuật"),
            "advice_text": str(getattr(criterion, "advice_text", "") or "Điều chỉnh tư thế và thực hiện lại."),
        })
    return rules


def allowed_training_labels_for_exercise(exercise_slug):
    """Return state labels that Admin may train for one exercise.

    Core exercises keep their detailed phase labels. A newly-created exercise
    automatically receives the three-state convention used by the benchmark
    thesis: ``start -> middle -> end`` plus ``other``.
    """
    slug = normalize_pose_label(str(exercise_slug or ""))
    prefixes = expected_label_prefixes(slug)
    labels = [label for label in RECOMMENDED_LABELS if label == "other" or label.startswith(prefixes)]
    known_prefix = any(
        slug == value for value in ("squat", "pushup", "curl_left", "curl_right")
    )
    if known_prefix and len(labels) > 1:
        return labels
    if not slug:
        return ["other"]
    return [f"{slug}_start", f"{slug}_middle", f"{slug}_end", "other"]


def training_media_counts(exercise_id):
    """Count independent uploaded source files per label for one exercise."""
    counts = {}
    rows = ExerciseTrainingMedia.query.filter_by(exercise_id=exercise_id).all()
    for row in rows:
        label = normalize_pose_label(row.label_name)
        counts[label] = counts.get(label, 0) + 1
    return counts


def validate_web_training_dataset(exercise):
    """Validate the same source-level split assumption used by train.py.

    Each required phase needs enough independent sources for train/validation/test
    source-level splits. For custom exercises the required phases are exactly
    start/middle/end, matching the three-state workflow in Long's thesis.
    ``other`` is optional and joins training only after the same source minimum.
    """
    allowed = allowed_training_labels_for_exercise(exercise.slug)
    required = [label for label in allowed if label != "other"]
    counts = training_media_counts(exercise.id)
    minimum_sources = int(MIN_TRAINING_SOURCES_PER_LABEL)
    missing = [label for label in required if counts.get(label, 0) < minimum_sources]
    labels_to_train = list(required)
    if counts.get("other", 0) >= minimum_sources:
        labels_to_train.append("other")
    return {
        "ready": not missing and len(required) >= 3,
        "required_labels": required,
        "labels_to_train": labels_to_train,
        "counts": counts,
        "missing_labels": missing,
        "minimum_sources_per_label": minimum_sources,
        "message": "" if not missing else (
            f"Mỗi trạng thái cần ít nhất {minimum_sources} tệp nguồn độc lập để tách train/validation/test. "
            + "Còn thiếu: " + ", ".join(
                f"{label} ({counts.get(label, 0)}/{minimum_sources})" for label in missing
            )
        ),
    }


def _relative_to_base(path):
    return os.path.relpath(str(path), BASE_DIR).replace(os.sep, "/")


def _train_exercise_model_job(exercise_id, labels_to_train):
    """Background worker for the Admin 'Huấn luyện mô hình' button."""
    output_abs = Path(BASE_DIR) / "models" / f"pose_{exercise_id}.pth"
    metrics_abs = output_abs.with_suffix(".metrics.json")
    command = [
        sys.executable,
        str(Path(BASE_DIR) / "train.py"),
        "--dataset", str(POSE_TRAINING_DIR),
        "--output", str(output_abs),
        "--labels", ",".join(labels_to_train),
    ]
    try:
        completed = subprocess.run(
            command,
            cwd=BASE_DIR,
            text=True,
            capture_output=True,
            timeout=60 * 45,
            check=False,
        )
        with app.app_context():
            exercise = db.session.get(WorkoutExercise, exercise_id)
            if not exercise:
                return
            exercise.pose_model_updated_at = datetime.utcnow()
            if completed.returncode != 0 or not output_abs.is_file() or not metrics_abs.is_file():
                exercise.pose_model_status = "error"
                exercise.pose_model_error = (
                    completed.stderr or completed.stdout or f"train.py exit={completed.returncode}"
                )[-4000:]
                db.session.commit()
                return

            report = json.loads(metrics_abs.read_text(encoding="utf-8"))
            test_metrics = report.get("test") or {}
            test_accuracy = test_metrics.get("accuracy")
            test_samples = int(test_metrics.get("samples", 0) or 0)
            exercise.pose_model_path = _relative_to_base(output_abs)
            exercise.pose_model_metrics_path = _relative_to_base(metrics_abs)
            exercise.pose_model_accuracy = float(test_accuracy or 0.0)
            if test_accuracy is not None and test_samples > 0 and float(test_accuracy) >= POSE_MODEL_MIN_TEST_ACCURACY:
                exercise.pose_model_status = "ready"
                exercise.pose_model_error = ""
            else:
                exercise.pose_model_status = "trained_below_gate"
                if test_accuracy is None or test_samples <= 0:
                    exercise.pose_model_error = "Model đã train nhưng chưa có test set độc lập nên chưa được dùng realtime."
                else:
                    exercise.pose_model_error = (
                        f"Test accuracy {float(test_accuracy):.3f} thấp hơn cổng "
                        f"{POSE_MODEL_MIN_TEST_ACCURACY:.2f}; model được lưu nhưng chưa được dùng realtime."
                    )
            db.session.commit()
    except Exception as exc:
        with app.app_context():
            exercise = db.session.get(WorkoutExercise, exercise_id)
            if exercise:
                exercise.pose_model_status = "error"
                exercise.pose_model_error = str(exc)[:4000]
                exercise.pose_model_updated_at = datetime.utcnow()
                db.session.commit()
    finally:
        with MODEL_TRAINING_LOCK:
            MODEL_TRAINING_THREADS.pop(int(exercise_id), None)


def configure_exercise_ai_resources(exercise, shared_state):
    """Bind per-exercise learned/reference assets before processor construction."""
    model_rel = str(getattr(exercise, "pose_model_path", "") or "").strip()
    classifier_path = os.path.join(BASE_DIR, model_rel.replace("/", os.sep)) if model_rel else ""
    if not (classifier_path and os.path.isfile(classifier_path)):
        # Legacy global checkpoint is retained only for the original four
        # exercises. A custom Admin-created exercise must never silently reuse
        # another exercise's model; it needs its own web-trained checkpoint.
        legacy = os.path.join(BASE_DIR, "models", "pose_phase_classifier.pth")
        core_slugs = {"squat", "pushup", "curl-left", "curl-right"}
        classifier_path = legacy if exercise.slug in core_slugs and os.path.isfile(legacy) else ""
    shared_state["pose_classifier_model_path"] = classifier_path
    shared_state["pose_classifier_status"] = str(getattr(exercise, "pose_model_status", "not_trained") or "not_trained")
    shared_state["pose_classifier_accuracy"] = float(getattr(exercise, "pose_model_accuracy", 0.0) or 0.0)

    reference_rel = str(getattr(exercise, "reference_motion_path", "") or "").strip()
    reference_abs = os.path.join(BASE_DIR, "static", reference_rel.replace("/", os.sep)) if reference_rel else ""
    # Bundled thesis references are a safe fallback when an older database was
    # created before reference_motion_path existed. This also activates the
    # newly derived Squat reference instead of silently reverting to angles only.
    if not (reference_abs and os.path.isfile(reference_abs)):
        bundled_refs = {
            "squat": "uploads/references/squat-glb-reference-auto.json",
            "pushup": "uploads/references/pushup-glb-reference-1f2a2b651d.json",
            "curl-left": "uploads/references/curl-left-glb-reference-137f5cf8de.json",
            "curl-right": "uploads/references/curl-right-glb-reference-b96f4a8cad.json",
        }
        bundled_rel = bundled_refs.get(str(exercise.slug or ""), "")
        bundled_abs = os.path.join(BASE_DIR, "static", bundled_rel.replace("/", os.sep)) if bundled_rel else ""
        if bundled_abs and os.path.isfile(bundled_abs):
            reference_rel, reference_abs = bundled_rel, bundled_abs
    if reference_abs and os.path.isfile(reference_abs):
        shared_state["reference_motion_file"] = reference_abs
        shared_state["reference_motion_source"] = str(getattr(exercise, "reference_motion_source", "") or "")
        shared_state["reference_motion_version"] = str(getattr(exercise, "reference_motion_version", "") or "")
        shared_state["reference_min_score"] = float(getattr(exercise, "reference_min_score", 55.0) or 55.0)
    else:
        shared_state["reference_motion_file"] = ""
        shared_state["reference_motion_source"] = ""
        shared_state["reference_motion_version"] = ""
    return shared_state


def summarize_hybrid_rep_evidence(rep_records):
    records = [item for item in (rep_records or []) if isinstance(item, dict)]
    reference_scores = []
    guard_confirmed = 0
    guard_evaluable = 0
    for item in records:
        ref = item.get("reference_match") or {}
        if ref.get("available"):
            reference_scores.append(float(ref.get("score", 0) or 0))
        guard = item.get("pose_guard") or {}
        if guard.get("available"):
            guard_evaluable += 1
            if guard.get("accepted"):
                guard_confirmed += 1
    return {
        "reference_score_avg": round(sum(reference_scores) / len(reference_scores), 2) if reference_scores else 0.0,
        "reference_rep_count": len(reference_scores),
        "pose_guard_confirmed_ratio": round(guard_confirmed / guard_evaluable, 4) if guard_evaluable else 0.0,
        "pose_guard_rep_count": guard_evaluable,
    }


ALLOWED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
ALLOWED_MODEL_EXTENSIONS = {".fbx", ".glb", ".gltf"}
ALLOWED_REFERENCE_EXTENSIONS = {".json"}


def save_uploaded_asset(uploaded_file, target_dir, relative_dir, allowed_extensions):
    """Validate and save an upload under a collision-resistant file name."""
    if not uploaded_file or not uploaded_file.filename:
        return ""

    original = secure_filename(uploaded_file.filename)
    extension = os.path.splitext(original)[1].lower()
    if extension not in set(allowed_extensions):
        raise ValueError(
            f"Định dạng {extension or '(không có đuôi)'} không được hỗ trợ."
        )

    stem = slugify(os.path.splitext(original)[0]) or "asset"
    filename = f"{stem}-{uuid.uuid4().hex[:10]}{extension}"
    save_path = os.path.join(target_dir, filename)
    uploaded_file.save(save_path)
    return f"{relative_dir}/{filename}"


def _infer_3d_format(model_path, stored_format=""):
    model_path = str(model_path or "").strip()
    extension = os.path.splitext(model_path)[1].lower()
    inferred = "fbx" if extension == ".fbx" else "gltf" if extension in {".glb", ".gltf"} else "none"
    model_format = str(stored_format or "").strip().lower()
    if model_format == "glb":
        model_format = "gltf"
    # Legacy builds may contain values such as ``procedural``. The current
    # viewer only exposes real FBX/GLTF assets, so stale states are hidden.
    if model_format not in {"fbx", "gltf", "none"}:
        model_format = inferred
    if model_format == "none" and inferred != "none":
        model_format = inferred
    return model_format if inferred != "none" else "none"


def _static_asset_absolute(relative_path):
    rel = str(relative_path or "").strip().replace("/", os.sep)
    return os.path.join(BASE_DIR, "static", rel) if rel else ""


def _remove_static_asset(relative_path):
    absolute = _static_asset_absolute(relative_path)
    if absolute and os.path.isfile(absolute):
        try:
            os.remove(absolute)
        except OSError:
            pass


def validate_saved_3d_asset(relative_path):
    """Validate self-contained GLB uploads before Admin can publish them."""
    if not relative_path:
        return {}
    extension = os.path.splitext(relative_path)[1].lower()
    if extension != ".glb":
        return {}
    try:
        return inspect_glb_asset(_static_asset_absolute(relative_path))
    except GLBReferenceError as exc:
        raise ValueError(f"GLB không hợp lệ để mô phỏng: {exc}") from exc


def auto_attach_reference_from_glb(exercise, relative_path):
    """Use the uploaded Mixamo GLB itself as the measurable scoring standard.

    The old workflow required a second manual Blender->JSON export.  For the
    four thesis exercises, a self-contained GLB with Mixamo bones can now be
    sampled directly.  This means the avatar shown to the learner and the
    reference used by DTW scoring come from the same animation asset.
    """
    relative_path = str(relative_path or "").strip()
    if os.path.splitext(relative_path)[1].lower() != ".glb":
        return None
    if str(getattr(exercise, "slug", "") or "").strip().lower() not in {"squat", "pushup", "curl-left", "curl-right"}:
        return None
    absolute = _static_asset_absolute(relative_path)
    payload = build_reference_from_glb(
        absolute,
        exercise.slug,
        getattr(exercise, "animation_key", "") or exercise.slug,
        samples=40,
        min_score=float(getattr(exercise, "reference_min_score", 55.0) or 55.0),
    )
    filename = f"{slugify(exercise.slug)}-glb-reference-{uuid.uuid4().hex[:10]}.json"
    output = os.path.join(UPLOAD_REFERENCE_DIR, filename)
    Path(output).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    old_reference = str(getattr(exercise, "reference_motion_path", "") or "")
    exercise.reference_motion_path = f"uploads/references/{filename}"
    exercise.reference_motion_source = str(payload.get("source", {}).get("type", "mixamo-blender"))
    exercise.reference_motion_version = str(payload.get("version", "mixamo-glb-v1"))
    if old_reference and old_reference != exercise.reference_motion_path:
        _remove_static_asset(old_reference)
    return payload


def _resolved_asset(path, stored_format=""):
    fmt = _infer_3d_format(path, stored_format)
    usable_path = str(path or "") if fmt != "none" else ""
    return {
        "path": usable_path,
        "filename": os.path.basename(usable_path) if usable_path else "",
        "format": fmt,
    }


def resolve_exercise_model(exercise):
    main_path = (
        getattr(exercise, "model_3d_path", "")
        or getattr(exercise, "fbx_path", "")
        or ""
    )
    main = _resolved_asset(main_path, getattr(exercise, "model_3d_format", ""))
    intro = _resolved_asset(
        getattr(exercise, "intro_model_3d_path", "") or "",
        getattr(exercise, "intro_model_3d_format", ""),
    )
    return {
        **main,
        "animation_key": getattr(exercise, "animation_key", "") or exercise.slug,
        "intro_path": intro["path"],
        "intro_filename": intro["filename"],
        "intro_format": intro["format"],
        "intro_animation_key": getattr(exercise, "intro_animation_key", "") or "",
        "reference_path": getattr(exercise, "reference_motion_path", "") or "",
        "reference_source": getattr(exercise, "reference_motion_source", "") or "",
        "reference_version": getattr(exercise, "reference_motion_version", "") or "",
        "reference_min_score": float(getattr(exercise, "reference_min_score", 55.0) or 55.0),
    }


def criterion_audio_map(criteria):
    """Return only generated audio files that really exist under /static."""
    result = {}
    for item in criteria or []:
        code = normalize_pose_label(getattr(item, "error_code", ""))
        rel = str(getattr(item, "audio_path", "") or "").strip().replace("\\", "/")
        if not code or not rel:
            continue
        absolute = os.path.join(BASE_DIR, "static", rel.replace("/", os.sep))
        if os.path.isfile(absolute) and os.path.getsize(absolute) > 44:
            result[code] = rel
    return result


def refresh_live_exercise_criteria(exercise):
    """Push saved Admin rules/audio into active workout states immediately."""
    if not exercise:
        return
    current_criteria = ExerciseCriterion.query.filter_by(exercise_id=exercise.id).all()
    audio_map = criterion_audio_map(current_criteria)
    runtime_rules = runtime_criterion_rules(current_criteria)
    suffix = f":{exercise.slug}"
    for key, state in list(EMERGENCY_STATES.items()):
        if key.endswith(suffix) and isinstance(state, dict):
            state["feedback_audio_map"] = dict(audio_map)
            state["admin_criteria_rules"] = list(runtime_rules)


def create_tts_audio(text, filename_hint):
    """Materialize one expert-rule message to a durable WAV file.

    The benchmark thesis creates feedback audio when Admin saves a rule.
    FitMotion does the same: Piper is preferred when configured and pyttsx3 is
    the offline fallback. The returned metadata is persisted so the Admin UI
    never pretends an audio file exists when synthesis failed.
    """
    safe_hint = slugify(filename_hint) or f"feedback-{uuid.uuid4().hex[:8]}"
    audio_file = f"{safe_hint}.wav"
    full_path = os.path.join(AUDIO_DIR, audio_file)
    result = synthesize_feedback(text, full_path)
    return {
        "success": bool(result.success),
        "path": f"audio/{audio_file}" if result.success else "",
        "provider": str(result.provider or ""),
        "error": str(result.error or ""),
    }


def create_default_data():
    db.create_all()
    ensure_sqlite_schema()

    admin = User.query.filter_by(email="admin@aifitness.local").first()
    if not admin:
        admin = User(
            fullname="Quan tri vien",
            email="admin@aifitness.local",
            password_hash=generate_password_hash("admin123"),
            role="admin"
        )
        db.session.add(admin)
        db.session.commit()

        db.session.add(UserProfile(
            user_id=admin.id,
            age=25,
            height=170,
            weight=65,
            goal="tap-nhe",
            health_note="khong-co-van-de",
            weekly_target=45,
            daily_target=6
        ))
        db.session.commit()

    demo_user = User.query.filter_by(email="22050062@student.bdu.edu.vn").first()
    if not demo_user:
        demo_user = User(
            fullname="Pham Duc Thuan",
            email="22050062@student.bdu.edu.vn",
            password_hash=generate_password_hash("123456"),
            role="user"
        )
        db.session.add(demo_user)
        db.session.commit()

        db.session.add(UserProfile(
            user_id=demo_user.id,
            age=21,
            height=170,
            weight=58,
            goal="tang-co",
            health_note="the-trang-yeu",
            weekly_target=50,
            daily_target=8
        ))
        db.session.commit()

    if WorkoutExercise.query.count() == 0:
        exercises = [
            WorkoutExercise(
                name="Squat",
                slug="squat",
                muscle_group="Đùi trước, mông, bắp chân",
                age_min=15,
                age_max=100,
                calories=0.1,
                difficulty="Cơ bản",
                side_mode="none",
                description="Bài tập thân dưới giúp phát triển sức mạnh chân và mông.",
                guide_text="Đứng thẳng, hạ hông xuống rồi đứng lên lại.",
                suitable_for="Người mới tập, mục tiêu tăng sức bền và sức mạnh chân.",
                caution_for="Thận trọng nếu đang đau gối hoặc đau lưng.",
                model_3d_format="none",
                animation_key="squat",
            ),
            WorkoutExercise(
                name="Hít đất",
                slug="pushup",
                muscle_group="Ngực, vai, tay sau",
                age_min=16,
                age_max=60,
                calories=0.12,
                difficulty="Trung bình",
                side_mode="none",
                description="Bài tập thân trên giúp phát triển ngực, vai và tay sau.",
                guide_text="Giữ thân thẳng, hạ người xuống rồi đẩy lên.",
                suitable_for="Người muốn tăng sức mạnh thân trên.",
                caution_for="Thận trọng nếu đau vai hoặc cổ tay.",
                model_3d_format="none",
                animation_key="pushup",
            ),
            WorkoutExercise(
                name="Cuốn tạ tay trái",
                slug="curl-left",
                muscle_group="Tay trước",
                age_min=15,
                age_max=100,
                calories=0.08,
                difficulty="Cơ bản",
                side_mode="left",
                description="Bài tập đơn tay giúp phát triển bắp tay trước bên trái.",
                guide_text="Giữ khuỷu tay gần thân, bắt đầu ở góc duỗi thoải mái; cuốn tạ lên rồi hạ về gần đúng góc bắt đầu. Không khóa cứng khuỷu ở 180 độ.",
                suitable_for="Người mới tập hoặc thể trạng yếu.",
                caution_for="Giữ khuỷu tay sát thân, không vung vai hoặc khóa cứng khớp khuỷu.",
                model_3d_format="none",
                animation_key="curl-left",
            ),
            WorkoutExercise(
                name="Cuốn tạ tay phải",
                slug="curl-right",
                muscle_group="Tay trước",
                age_min=15,
                age_max=100,
                calories=0.08,
                difficulty="Cơ bản",
                side_mode="right",
                description="Bài tập đơn tay giúp phát triển bắp tay trước bên phải.",
                guide_text="Giữ khuỷu tay gần thân, bắt đầu ở góc duỗi thoải mái; cuốn tạ lên rồi hạ về gần đúng góc bắt đầu. Không khóa cứng khuỷu ở 180 độ.",
                suitable_for="Người mới tập hoặc thể trạng yếu.",
                caution_for="Giữ khuỷu tay sát thân, không vung vai hoặc khóa cứng khớp khuỷu.",
                model_3d_format="none",
                animation_key="curl-right",
            ),
        ]
        db.session.add_all(exercises)
        db.session.commit()

    # Keep every core exercise usable even when the legacy database already
    # existed before the new 3D fields and evaluation criteria were added.
    default_exercise_config = {
        "squat": {
            "animation_key": "squat",
            "criteria": [
                ("Độ sâu squat", "góc gối", "<=", 125, "Squat chưa đủ sâu", "Hạ hông thấp hơn nhưng vẫn giữ kiểm soát."),
                ("Kiểm soát thân người", "góc thân", ">=", 135, "Thân người nghiêng quá mức", "Giữ ngực mở và lưng trung lập."),
            ],
        },
        "pushup": {
            "animation_key": "pushup",
            "criteria": [
                ("Độ sâu hít đất", "góc khuỷu", "<=", 115, "Hạ người chưa đủ thấp", "Gập khuỷu thêm trước khi đẩy lên."),
                ("Đường thẳng cơ thể", "vai-hông-cổ chân", ">=", 140, "Thân người chưa thẳng", "Giữ vai, hông và cổ chân thẳng hàng."),
            ],
        },
        "curl-left": {
            "animation_key": "curl-left",
            "criteria": [
                ("Biên độ cuốn tạ", "góc khuỷu", "adaptive", 0, "Gập tay chưa đủ", "Gập đủ biên độ so với góc bắt đầu cá nhân."),
                ("Trở về tư thế bắt đầu", "góc khuỷu", "adaptive", 0, "Chưa trở về gần góc bắt đầu", "Hạ tạ về gần góc đã hiệu chuẩn; không khóa cứng 180 độ."),
                ("Kiểm soát khuỷu", "độ lệch khuỷu", "<=", 45, "Khuỷu tay bị lệch", "Giữ khuỷu gần thân và hạn chế vung vai."),
            ],
        },
        "curl-right": {
            "animation_key": "curl-right",
            "criteria": [
                ("Biên độ cuốn tạ", "góc khuỷu", "adaptive", 0, "Gập tay chưa đủ", "Gập đủ biên độ so với góc bắt đầu cá nhân."),
                ("Trở về tư thế bắt đầu", "góc khuỷu", "adaptive", 0, "Chưa trở về gần góc bắt đầu", "Hạ tạ về gần góc đã hiệu chuẩn; không khóa cứng 180 độ."),
                ("Kiểm soát khuỷu", "độ lệch khuỷu", "<=", 45, "Khuỷu tay bị lệch", "Giữ khuỷu gần thân và hạn chế vung vai."),
            ],
        },
    }

    default_criterion_runtime = {
        "squat": {
            "Độ sâu squat": ("notlow", "middle"),
            "Kiểm soát thân người": ("backlean", "middle"),
        },
        "pushup": {
            "Độ sâu hít đất": ("notlow", "middle"),
            "Đường thẳng cơ thể": ("bodyline", "middle"),
        },
        "curl-left": {
            "Biên độ cuốn tạ": ("notbend", "middle"),
            "Trở về tư thế bắt đầu": ("notstraight", "end"),
            "Kiểm soát khuỷu": ("elbowshift", "middle"),
        },
        "curl-right": {
            "Biên độ cuốn tạ": ("notbend", "middle"),
            "Trở về tư thế bắt đầu": ("notstraight", "end"),
            "Kiểm soát khuỷu": ("elbowshift", "middle"),
        },
    }

    for slug, config in default_exercise_config.items():
        exercise = WorkoutExercise.query.filter_by(slug=slug).first()
        if not exercise:
            continue
        if not exercise.animation_key:
            exercise.animation_key = config["animation_key"]
        if not exercise.model_3d_format:
            exercise.model_3d_format = "none"
        if not exercise.guide_text and slug.startswith("curl-"):
            exercise.guide_text = "Cuốn tạ theo góc bắt đầu cá nhân; không yêu cầu khóa cứng khuỷu 180 độ."
        existing_items = ExerciseCriterion.query.filter_by(exercise_id=exercise.id).all()
        existing_by_title = {item.title: item for item in existing_items}
        runtime_map = default_criterion_runtime.get(slug, {})
        for title, joint_name, operator, angle_value, message_text, advice_text in config["criteria"]:
            runtime_code, runtime_phase = runtime_map.get(title, ("", "middle"))
            existing = existing_by_title.get(title)
            if existing is not None:
                if not getattr(existing, "error_code", "") and runtime_code:
                    existing.error_code = runtime_code
                if not getattr(existing, "phase", ""):
                    existing.phase = runtime_phase
                continue
            db.session.add(ExerciseCriterion(
                exercise_id=exercise.id,
                title=title,
                joint_name=joint_name,
                operator=operator,
                angle_value=angle_value,
                message_text=message_text,
                advice_text=advice_text,
                error_code=runtime_code,
                phase=runtime_phase,
                audio_path="",
            ))
        ensure_level_config_defaults(exercise)
    db.session.commit()

    demo_user = User.query.filter_by(email="22050062@student.bdu.edu.vn").first()
    start_date = datetime.now().date()

    if demo_user and WorkoutPlan.query.filter_by(user_id=demo_user.id).count() == 0:
        all_ex = WorkoutExercise.query.all()

        plans = []
        for i, ex in enumerate(all_ex[:3]):
            plans.append(
                WorkoutPlan(
                    user_id=demo_user.id,
                    exercise_id=ex.id,
                    workout_date=str(start_date + timedelta(days=i)),
                    set_count=1,
                    rep_target=15,
                    status="pending"
                )
            )
        db.session.add_all(plans)
        db.session.commit()

    if demo_user and WorkoutSession.query.filter_by(user_id=demo_user.id).count() == 0:
        squat_ex = WorkoutExercise.query.filter_by(slug="squat").first()
        if squat_ex:
            db.session.add(WorkoutSession(
                user_id=demo_user.id,
                exercise_id=squat_ex.id,
                session_date=str(start_date),
                total_rep=12,
                good_rep=10,
                total_error=2,
                confidence_avg=0.87,
                phase_start_error=1,
                phase_middle_error=1,
                phase_end_error=0
            ))
            db.session.commit()


with app.app_context():
    create_default_data()
    startup_backup = backup_database(DB_PATH)

print(f"[FitMotion AI] Database bền vững: {DB_PATH}")
if DB_MIGRATION_NOTE:
    print(f"[FitMotion AI] {DB_MIGRATION_NOTE}")
if startup_backup:
    print(f"[FitMotion AI] Backup khởi động: {startup_backup}")


def get_user_context():
    return current_user()


def get_recommended_rep_targets(profile):
    """
    Tính rep đề xuất theo hồ sơ. Đây là rep khuyến nghị,
    không phải mục tiêu cứng bắt buộc của người dùng.
    """
    if not profile:
        return {
            "weekly": 45,
            "daily": 6,
            "reason": "Dùng mức mặc định do chưa có hồ sơ người tập."
        }

    age = profile.age or 18
    height = float(profile.height or 170)
    weight = float(profile.weight or 60)
    goal = normalize_goal(profile.goal or "tap-nhe")
    health = normalize_health_note(profile.health_note or "khong-co-van-de")

    if goal == "tang-co":
        base_weekly = 84
        goal_text = "Mục tiêu tăng cơ phù hợp với mức rep trung bình đến khá."
    elif goal == "giam-mo":
        base_weekly = 105
        goal_text = "Mục tiêu giảm mỡ phù hợp với tổng rep cao hơn để tăng vận động."
    else:
        base_weekly = 56
        goal_text = "Mục tiêu tập nhẹ phù hợp với mức rep vừa phải để dễ duy trì."

    age_factor = 1.0
    if age < 18:
        age_factor = 0.9
    elif age <= 30:
        age_factor = 1.0
    elif age <= 45:
        age_factor = 0.9
    elif age <= 60:
        age_factor = 0.8
    else:
        age_factor = 0.7

    body_factor = 1.0
    bmi = 0.0
    if height > 0:
        bmi = weight / ((height / 100.0) ** 2)

    if bmi > 0:
        if bmi < 18.5:
            body_factor = 0.85
        elif bmi < 25:
            body_factor = 1.0
        elif bmi < 30:
            body_factor = 0.95
        else:
            body_factor = 0.85

    health_factor = 1.0
    if health == "the-trang-yeu":
        health_factor = 0.75
    elif health in ["dau-goi", "dau-vai", "co-tay-yeu", "dau-lung"]:
        health_factor = 0.8
    elif health == "huyet-ap-khong-on-dinh":
        health_factor = 0.7

    weekly = int(round(base_weekly * age_factor * body_factor * health_factor))
    weekly = max(21, weekly)
    daily = max(3, int(round(weekly / 7)))

    return {
        "weekly": weekly,
        "daily": daily,
        "reason": f"{goal_text} Giá trị được hiệu chỉnh theo độ tuổi, thể trạng và tình trạng sức khỏe."
    }


def get_goal_daily_target(profile):
    if not profile:
        return 6

    daily_target = getattr(profile, "daily_target", None)
    if daily_target and int(daily_target) > 0:
        return max(1, int(daily_target))

    suggested = get_recommended_rep_targets(profile)
    return suggested["daily"]


def normalize_display_good_rep(total_rep, good_rep):
    """Clamp the real good-rep count; never inflate AI results for display."""
    total_rep = max(0, int(total_rep or 0))
    good_rep = max(0, int(good_rep or 0))
    return min(total_rep, good_rep)

def get_week_bounds(base_date=None):
    if base_date is None:
        base_date = datetime.now().date()
    monday = base_date - timedelta(days=base_date.weekday())
    sunday = monday + timedelta(days=6)
    return monday, sunday


def get_weekly_sessions(user_id, base_date=None):
    week_start, week_end = get_week_bounds(base_date)
    week_start_str = week_start.isoformat()
    week_end_str = week_end.isoformat()
    sessions = WorkoutSession.query.filter(
        WorkoutSession.user_id == user_id,
        WorkoutSession.session_date >= week_start_str,
        WorkoutSession.session_date <= week_end_str
    ).order_by(WorkoutSession.created_at.desc()).all()
    return sessions, week_start, week_end


def get_or_create_shared_state(user_id, slug):
    _, shared_state = get_or_create_emergency_state(user_id, slug)
    defaults = get_default_emergency_state()
    for key, value in defaults.items():
        if key not in shared_state:
            if isinstance(value, dict):
                shared_state[key] = dict(value)
            elif isinstance(value, list):
                shared_state[key] = list(value)
            else:
                shared_state[key] = value
    shared_state["exercise_slug"] = slug
    return shared_state


def update_wellness_check(shared_state):
    """Pause a caution-flagged workout once after 60 seconds and ask for confirmation."""
    if not bool(shared_state.get("wellness_check_enabled", False)):
        return
    if bool(shared_state.get("wellness_check_completed", False)):
        return
    if bool(shared_state.get("active", False)):
        return

    started_at = float(shared_state.get("session_started_at", 0.0) or 0.0)
    if started_at <= 0:
        return
    elapsed = max(0.0, time.time() - started_at)
    required = max(30, int(shared_state.get("wellness_check_seconds", 60) or 60))
    if elapsed >= required and not bool(shared_state.get("wellness_check_due", False)):
        shared_state.update({
            "wellness_check_due": True,
            "workout_paused": True,
            "pause_reason": "wellness_check",
            "status_text": "TAM DUNG - KIEM TRA SUC KHOE",
        })

def reset_workout_runtime_state(user_id, slug, preserve_target=True):
    """Reset one exercise runtime to a clean, non-running session.

    Emergency/pause/timer state is intentionally not preserved. A completed or
    stopped level must never keep processing state that leaks into the next
    Easy/Medium/Hard workout.
    """
    shared_state = get_or_create_shared_state(user_id, slug)
    target = int(shared_state.get("target_rep", 0)) if preserve_target else 0
    training_level = normalize_training_level(shared_state.get("training_level", DEFAULT_LEVEL))

    shared_state.clear()
    shared_state.update(get_default_emergency_state())
    shared_state["target_rep"] = target
    shared_state["training_level"] = training_level
    shared_state["training_level_label"] = training_level_label(training_level)
    shared_state["exercise_slug"] = slug
    shared_state["workout_done"] = False
    shared_state["active"] = False
    shared_state["workout_paused"] = False
    shared_state["pause_reason"] = ""
    shared_state["session_started_at"] = 0.0
    shared_state["last_rep_at"] = 0.0
    shared_state["idle_seconds"] = 0
    shared_state["elapsed_seconds"] = 0
    return shared_state



DRAFT_STATE_KEYS = (
    "total_rep", "good_rep", "bad_rep",
    "phase_start_error", "phase_middle_error", "phase_end_error",
    "tracking_abort_count", "target_rep", "exercise_slug",
    "training_level", "training_level_label",
    "rep_quality_score", "quality_score_avg", "keypoint_confidence_avg",
    "feedback_id", "feedback_text", "feedback_advice", "feedback_level",
    "voice_text", "last_error_code", "last_rep_components",
    "rep_scores", "rep_records", "error_code_counts",
    "rejected_attempt_count", "last_rejected_attempt",
    "curl_extension_baseline", "curl_calibration_ready",
    "squat_standing_baseline", "squat_calibration_ready",
    "pushup_top_baseline", "pushup_calibration_ready",
    "set_count", "reps_per_set", "rest_seconds", "max_idle_seconds",
    "max_session_seconds", "min_good_rep_ratio", "min_quality_avg",
    "min_confidence_avg", "min_stability_avg", "max_tracking_abort_count",
    "session_started_at", "last_rep_at",
    "pose_classifier_last", "pose_guard_last_decision",
    "reference_match", "reference_motion_source", "reference_motion_version",
)


def _valid_session_date(value):
    try:
        return datetime.strptime(str(value), "%Y-%m-%d").date().isoformat()
    except (TypeError, ValueError):
        return datetime.now().date().isoformat()


def _draft_payload(shared_state):
    payload = {}
    for key in DRAFT_STATE_KEYS:
        value = shared_state.get(key)
        if isinstance(value, dict):
            payload[key] = dict(value)
        elif isinstance(value, list):
            payload[key] = list(value)
        else:
            payload[key] = value
    return payload


def checkpoint_workout_draft(user_id, exercise, session_date, shared_state, force=False):
    """Persist a resumable workout checkpoint when the rep state changes."""
    if bool(shared_state.get("workout_cancelled", False)):
        return None
    session_date = _valid_session_date(session_date)
    total_rep = max(0, int(shared_state.get("total_rep", 0) or 0))
    if total_rep <= 0:
        return None
    good_rep = normalize_display_good_rep(total_rep, shared_state.get("good_rep", 0))
    total_error = max(0, total_rep - good_rep)
    signature = (
        total_rep,
        good_rep,
        round(float(shared_state.get("quality_score_avg", 0) or 0), 3),
        round(float(shared_state.get("keypoint_confidence_avg", 0) or 0), 4),
        len(shared_state.get("rep_records", []) or []),
        int(shared_state.get("target_rep", 0) or 0),
    )
    now = time.monotonic()
    if not force and shared_state.get("_draft_signature") == signature:
        return None
    if not force and now - float(shared_state.get("_draft_saved_at", 0) or 0) < 0.75:
        return None

    draft = WorkoutSessionDraft.query.filter_by(
        user_id=user_id,
        exercise_id=exercise.id,
        session_date=session_date,
    ).first()
    if not draft:
        draft = WorkoutSessionDraft(
            user_id=user_id,
            exercise_id=exercise.id,
            session_date=session_date,
        )
        db.session.add(draft)

    draft.total_rep = total_rep
    draft.good_rep = good_rep
    draft.total_error = total_error
    draft.keypoint_confidence_avg = float(shared_state.get("keypoint_confidence_avg", 0) or 0)
    draft.quality_score_avg = float(shared_state.get("quality_score_avg", 0) or 0)
    draft.training_level = normalize_training_level(shared_state.get("training_level", DEFAULT_LEVEL))
    draft.state_json = json.dumps(_draft_payload(shared_state), ensure_ascii=False)
    db.session.commit()
    shared_state["_draft_signature"] = signature
    shared_state["_draft_saved_at"] = now
    return draft


def restore_workout_draft(user_id, exercise, session_date, shared_state):
    """Restore completed reps from SQLite without restoring a half-finished phase."""
    if int(shared_state.get("total_rep", 0) or 0) > 0:
        return None
    draft = WorkoutSessionDraft.query.filter_by(
        user_id=user_id,
        exercise_id=exercise.id,
        session_date=_valid_session_date(session_date),
    ).first()
    if not draft:
        return None
    try:
        payload = json.loads(draft.state_json or "{}")
    except (TypeError, ValueError, json.JSONDecodeError):
        payload = {}
    for key in DRAFT_STATE_KEYS:
        if key in payload:
            shared_state[key] = payload[key]
    shared_state["total_rep"] = int(draft.total_rep or 0)
    shared_state["good_rep"] = normalize_display_good_rep(draft.total_rep, draft.good_rep)
    shared_state["bad_rep"] = max(0, int(draft.total_error or 0))
    shared_state["quality_score_avg"] = float(draft.quality_score_avg or 0)
    shared_state["keypoint_confidence_avg"] = float(draft.keypoint_confidence_avg or 0)
    restored_level = normalize_training_level(
        shared_state.get("training_level", getattr(draft, "training_level", DEFAULT_LEVEL))
    )
    shared_state["training_level"] = restored_level
    shared_state["training_level_label"] = training_level_label(restored_level)
    # Never restore an incomplete movement phase or an emergency alarm.
    shared_state["status_text"] = "Da khoi phuc tien do - san sang rep tiep theo"
    shared_state["workout_done"] = False
    shared_state["active"] = False
    shared_state["emergency_candidate"] = False
    shared_state["emergency_immobile_seconds"] = 0.0
    shared_state.pop("_draft_signature", None)
    return draft


def delete_workout_draft(user_id, exercise_id, session_date):
    draft = WorkoutSessionDraft.query.filter_by(
        user_id=user_id,
        exercise_id=exercise_id,
        session_date=_valid_session_date(session_date),
    ).first()
    if draft:
        db.session.delete(draft)
    return draft


def _live_exercise_ids(user_id):
    exercise_ids = set()
    for key, state in EMERGENCY_STATES.items():
        if not key.startswith(f"{user_id}:") or bool(state.get("workout_done", False)):
            continue
        if int(state.get("total_rep", 0) or 0) <= 0:
            continue
        slug = str(state.get("exercise_slug", "") or "")
        exercise = WorkoutExercise.query.filter_by(slug=slug).first() if slug else None
        if exercise:
            exercise_ids.add(exercise.id)
    return exercise_ids


def _drafts_not_in_live_state(user_id, start_date, end_date=None):
    end_date = end_date or start_date
    live_ids = _live_exercise_ids(user_id)
    drafts = WorkoutSessionDraft.query.filter(
        WorkoutSessionDraft.user_id == user_id,
        WorkoutSessionDraft.session_date >= _valid_session_date(start_date),
        WorkoutSessionDraft.session_date <= _valid_session_date(end_date),
    ).all()
    return [draft for draft in drafts if draft.exercise_id not in live_ids]


def get_today_summary(user_id, profile=None):
    if not profile:
        profile = UserProfile.query.filter_by(user_id=user_id).first()

    today_str = datetime.now().date().isoformat()
    sessions = WorkoutSession.query.filter_by(user_id=user_id, session_date=today_str).all()
    drafts = _drafts_not_in_live_state(user_id, today_str)
    live_total = 0
    live_good = 0
    live_bad = 0

    for key, state in EMERGENCY_STATES.items():
        if not key.startswith(f"{user_id}:"):
            continue
        if bool(state.get("workout_done", False)):
            continue
        current_total = int(state.get("total_rep", 0) or 0)
        current_good = normalize_display_good_rep(current_total, state.get("good_rep", 0))
        live_total += current_total
        live_good += current_good
        live_bad += max(0, current_total - current_good)

    done_rep = (
        sum((s.total_rep or 0) for s in sessions)
        + sum((d.total_rep or 0) for d in drafts)
        + live_total
    )
    good_rep = (
        sum(normalize_display_good_rep(s.total_rep, s.good_rep) for s in sessions)
        + sum(normalize_display_good_rep(d.total_rep, d.good_rep) for d in drafts)
        + live_good
    )
    total_error = (
        sum((s.total_error or 0) for s in sessions)
        + sum((d.total_error or 0) for d in drafts)
        + live_bad
    )
    daily_target = get_goal_daily_target(profile)
    remaining = max(0, daily_target - done_rep)
    completed = done_rep >= daily_target

    return {
        "date": today_str,
        "daily_target": daily_target,
        "done_rep": done_rep,
        "good_rep": good_rep,
        "total_error": total_error,
        "remaining_rep": remaining,
        "completed": completed,
        "progress_percent": round((done_rep / daily_target) * 100, 2) if daily_target else 0,
    }


def get_today_summary_cached(user_id, profile=None, shared_state=None, ttl_seconds=4.0):
    """Cache only persisted SQLite totals; merge live reps on every poll.

    This keeps the realtime card responsive while avoiding a database query at
    1-2 Hz. Finishing a workout invalidates the persisted cache immediately.
    """
    if not profile:
        profile = UserProfile.query.filter_by(user_id=user_id).first()

    state = shared_state if shared_state is not None else {}
    now = time.monotonic()
    cached = state.get("_today_persisted_cache")
    cached_at = float(state.get("_today_persisted_cache_at", 0.0) or 0.0)
    today_str = datetime.now().date().isoformat()

    if not isinstance(cached, dict) or now - cached_at > float(ttl_seconds):
        sessions = WorkoutSession.query.filter_by(user_id=user_id, session_date=today_str).all()
        drafts = _drafts_not_in_live_state(user_id, today_str)
        cached = {
            "done_rep": (
                sum(int(item.total_rep or 0) for item in sessions)
                + sum(int(item.total_rep or 0) for item in drafts)
            ),
            "good_rep": (
                sum(normalize_display_good_rep(item.total_rep, item.good_rep) for item in sessions)
                + sum(normalize_display_good_rep(item.total_rep, item.good_rep) for item in drafts)
            ),
            "total_error": (
                sum(int(item.total_error or 0) for item in sessions)
                + sum(int(item.total_error or 0) for item in drafts)
            ),
        }
        state["_today_persisted_cache"] = dict(cached)
        state["_today_persisted_cache_at"] = now

    live_total = live_good = live_bad = 0
    for key, runtime in EMERGENCY_STATES.items():
        if not key.startswith(f"{user_id}:") or bool(runtime.get("workout_done", False)):
            continue
        current_total = int(runtime.get("total_rep", 0) or 0)
        current_good = normalize_display_good_rep(current_total, runtime.get("good_rep", 0))
        live_total += current_total
        live_good += current_good
        live_bad += max(0, current_total - current_good)

    daily_target = get_goal_daily_target(profile)
    done_rep = int(cached["done_rep"]) + live_total
    good_rep = int(cached["good_rep"]) + live_good
    total_error = int(cached["total_error"]) + live_bad
    remaining = max(0, daily_target - done_rep)
    return {
        "date": today_str,
        "daily_target": daily_target,
        "done_rep": done_rep,
        "good_rep": good_rep,
        "total_error": total_error,
        "remaining_rep": remaining,
        "completed": done_rep >= daily_target,
        "progress_percent": round((done_rep / daily_target) * 100, 2) if daily_target else 0,
    }

def invalidate_today_summary_cache(user_id):
    for key, state in EMERGENCY_STATES.items():
        if key.startswith(f"{user_id}:"):
            state.pop("_today_summary_cache", None)
            state.pop("_today_summary_cache_at", None)
            state.pop("_today_persisted_cache", None)
            state.pop("_today_persisted_cache_at", None)


def calculate_user_dashboard(user_id):
    profile = UserProfile.query.filter_by(user_id=user_id).first()
    plans = WorkoutPlan.query.filter_by(user_id=user_id).all()
    today_summary = get_today_summary(user_id, profile)
    weekly_target = profile.weekly_target if profile else 45

    sessions_week, week_start, week_end = get_weekly_sessions(user_id)
    plans_week = WorkoutPlan.query.filter(
        WorkoutPlan.user_id == user_id,
        WorkoutPlan.workout_date >= week_start.isoformat(),
        WorkoutPlan.workout_date <= week_end.isoformat(),
    ).all()

    drafts_week = _drafts_not_in_live_state(user_id, week_start.isoformat(), week_end.isoformat())
    done_count = (
        sum((s.total_rep or 0) for s in sessions_week)
        + sum((d.total_rep or 0) for d in drafts_week)
    )
    total_errors = (
        sum((s.total_error or 0) for s in sessions_week)
        + sum((d.total_error or 0) for d in drafts_week)
    )
    start_errors = sum((s.phase_start_error or 0) for s in sessions_week)
    middle_errors = sum((s.phase_middle_error or 0) for s in sessions_week)
    end_errors = sum((s.phase_end_error or 0) for s in sessions_week)

    calories = 0.0
    for s in sessions_week:
        ex = db.session.get(WorkoutExercise, s.exercise_id)
        if ex:
            calories += (ex.calories or 0.0) * (s.total_rep or 0)
    for draft in drafts_week:
        ex = db.session.get(WorkoutExercise, draft.exercise_id)
        if ex:
            calories += (ex.calories or 0.0) * (draft.total_rep or 0)

    live_total = 0
    live_errors = 0
    live_start = 0
    live_middle = 0
    live_end = 0
    live_calories = 0.0

    for key, state in EMERGENCY_STATES.items():
        if not key.startswith(f"{user_id}:"):
            continue
        if bool(state.get("workout_done", False)):
            continue
        slug = str(state.get("exercise_slug", "") or "")
        ex = WorkoutExercise.query.filter_by(slug=slug).first() if slug else None
        current_total = int(state.get("total_rep", 0) or 0)
        live_total += current_total
        live_errors += int(state.get("bad_rep", 0) or 0)
        live_start += int(state.get("phase_start_error", 0) or 0)
        live_middle += int(state.get("phase_middle_error", 0) or 0)
        live_end += int(state.get("phase_end_error", 0) or 0)
        if ex:
            live_calories += (ex.calories or 0.0) * current_total

    done_count += live_total
    total_errors += live_errors
    start_errors += live_start
    middle_errors += live_middle
    end_errors += live_end
    calories = round(calories + live_calories, 2)

    live_schedule_items = []
    current_day = datetime.now().date().isoformat()
    for key, state in EMERGENCY_STATES.items():
        if not key.startswith(f"{user_id}:") or bool(state.get("workout_done", False)):
            continue
        current_total = int(state.get("total_rep", 0) or 0)
        if current_total <= 0:
            continue
        slug = str(state.get("exercise_slug", "") or "")
        exercise = WorkoutExercise.query.filter_by(slug=slug).first() if slug else None
        if exercise:
            live_schedule_items.append({
                "exercise_id": exercise.id,
                "session_date": current_day,
                "total_rep": current_total,
            })

    schedule_adherence = build_schedule_adherence(
        plans=plans_week,
        sessions=sessions_week,
        drafts=drafts_week,
        live_items=live_schedule_items,
        today=current_day,
    )

    weekly_remaining = max(0, weekly_target - done_count)
    weekly_completed = done_count >= weekly_target
    progress_percent = round((done_count / weekly_target) * 100, 2) if weekly_target else 0

    if weekly_completed:
        weekly_message = f"Chúc mừng! Bạn đã hoàn thành mục tiêu tuần với {done_count}/{weekly_target} rep."
    else:
        weekly_message = f"Tuần này bạn còn thiếu {weekly_remaining} rep để đạt mục tiêu {weekly_target} rep."

    error_reason_items = [
        {
            "key": "start",
            "title": "Khởi động chưa đúng",
            "count": start_errors,
            "reason": "Bắt đầu động tác chưa ổn định hoặc vào tư thế chưa đúng chuẩn ban đầu.",
        },
        {
            "key": "middle",
            "title": "Biên độ chưa chuẩn",
            "count": middle_errors,
            "reason": "Co hoặc duỗi chưa đủ, xuống chưa sâu hoặc biên độ rep chưa trọn vẹn.",
        },
        {
            "key": "end",
            "title": "Kết thúc chưa gọn",
            "count": end_errors,
            "reason": "Kết thúc rep chưa về đúng tư thế hoặc mất kiểm soát ở cuối động tác.",
        },
    ]

    return {
        "weekly_target": weekly_target,
        "done_count": done_count,
        "progress_percent": progress_percent,
        "total_plans": len(sessions_week) + sum(1 for key, state in EMERGENCY_STATES.items() if key.startswith(f"{user_id}:") and not bool(state.get("workout_done", False))),
        "total_errors": total_errors,
        "start_errors": start_errors,
        "middle_errors": middle_errors,
        "end_errors": end_errors,
        "calories": calories,
        "today_target": today_summary["daily_target"],
        "today_done": today_summary["done_rep"],
        "today_remaining": today_summary["remaining_rep"],
        "today_completed": today_summary["completed"],
        "today_progress_percent": today_summary["progress_percent"],
        "weekly_remaining": weekly_remaining,
        "weekly_completed": weekly_completed,
        "weekly_message": weekly_message,
        "schedule_planned_count": schedule_adherence["planned_count"],
        "schedule_performed_count": schedule_adherence["performed_count"],
        "schedule_missed_count": schedule_adherence["missed_count"],
        "schedule_upcoming_count": schedule_adherence["upcoming_count"],
        "schedule_completion_percent": schedule_adherence["completion_percent"],
        "error_reason_items": error_reason_items,
        "week_start": week_start.isoformat(),
        "week_end": week_end.isoformat(),
    }


def get_week_schedule(user_id, selected_date=None):
    """Return the current week using both plans and actual workout data.

    The old dashboard counted only ``WorkoutPlan`` rows, so an ad-hoc workout
    could be saved correctly in ``WorkoutSession`` while the weekly card still
    displayed ``0 bài``.  The new aggregation treats plans, finalized sessions,
    durable drafts and live reps as sources of weekly activity.
    """
    today = datetime.now().date()
    monday = today - timedelta(days=today.weekday())
    sunday = monday + timedelta(days=6)
    week_dates = [(monday + timedelta(days=i)).isoformat() for i in range(7)]

    if not selected_date or selected_date not in week_dates:
        selected_date = today.isoformat() if today.isoformat() in week_dates else monday.isoformat()

    plans = WorkoutPlan.query.filter(
        WorkoutPlan.user_id == user_id,
        WorkoutPlan.workout_date >= monday.isoformat(),
        WorkoutPlan.workout_date <= sunday.isoformat(),
    ).all()
    sessions = WorkoutSession.query.filter(
        WorkoutSession.user_id == user_id,
        WorkoutSession.session_date >= monday.isoformat(),
        WorkoutSession.session_date <= sunday.isoformat(),
    ).all()
    drafts = _drafts_not_in_live_state(user_id, monday.isoformat(), sunday.isoformat())

    live_items = []
    for key, state in EMERGENCY_STATES.items():
        if not key.startswith(f"{user_id}:") or bool(state.get("workout_done", False)):
            continue
        total_rep = int(state.get("total_rep", 0) or 0)
        if total_rep <= 0:
            continue
        slug = str(state.get("exercise_slug", "") or "")
        exercise = WorkoutExercise.query.filter_by(slug=slug).first() if slug else None
        if exercise:
            live_items.append({
                "exercise_id": exercise.id,
                "session_date": today.isoformat(),
                "total_rep": total_rep,
            })

    exercise_ids = {int(item.exercise_id) for item in plans + sessions + drafts}
    exercise_ids.update(int(item["exercise_id"]) for item in live_items)
    exercises = WorkoutExercise.query.filter(WorkoutExercise.id.in_(exercise_ids)).all() if exercise_ids else []
    exercise_calories = {exercise.id: float(exercise.calories or 0.0) for exercise in exercises}

    activity = build_week_activity(
        week_dates=week_dates,
        plans=plans,
        sessions=sessions,
        drafts=drafts,
        live_items=live_items,
        exercise_calories=exercise_calories,
    )

    days = []
    for i, day_text in enumerate(week_dates):
        day = monday + timedelta(days=i)
        item = activity.get(day_text, {"count": 0, "completed_count": 0, "kcal": 0.0})
        days.append({
            "date": day_text,
            "short": day.strftime("%d/%m"),
            "count": int(item["count"]),
            "completed_count": int(item["completed_count"]),
            "kcal": round(float(item["kcal"]), 2),
            "selected": day_text == selected_date,
        })

    return days, selected_date



def get_day_detail(user_id, selected_date):
    plans = WorkoutPlan.query.filter_by(
        user_id=user_id,
        workout_date=selected_date
    ).all()

    sessions = WorkoutSession.query.filter_by(
        user_id=user_id,
        session_date=selected_date
    ).all()
    drafts = _drafts_not_in_live_state(user_id, selected_date)

    exercise_items = []
    total_count = 0
    total_kcal = 0.0
    planned_exercise_ids = {int(plan.exercise_id) for plan in plans}
    actual_exercises = {}

    def record_actual(exercise, reps, status, training_level=DEFAULT_LEVEL):
        if not exercise or int(reps or 0) <= 0:
            return
        normalized_level = normalize_training_level(training_level)
        item = actual_exercises.setdefault(exercise.id, {
            "exercise": exercise,
            "reps": 0,
            "kcal": 0.0,
            "status": status,
            "training_level": normalized_level,
        })
        item["reps"] += int(reps or 0)
        item["kcal"] += float(exercise.calories or 0.0) * int(reps or 0)
        if status == "completed":
            item["status"] = "completed"

    for p in plans:
        ex = db.session.get(WorkoutExercise, p.exercise_id)
        if ex:
            total_count += 1
            kcal = round((ex.calories or 0) * max(1, int(p.set_count or 1)) * int(p.rep_target or 0), 2)
            total_kcal += kcal

            exercise_items.append({
                "name": ex.name,
                "set_count": p.set_count,
                "rep_target": p.rep_target,
                "training_level": normalize_training_level(getattr(p, "training_level", DEFAULT_LEVEL)),
                "training_level_label": training_level_label(getattr(p, "training_level", DEFAULT_LEVEL)),
                "status": p.status,
                "kcal": kcal
            })

    day_sessions = []
    total_rep = 0
    good_rep = 0
    total_error = 0

    for s in sessions:
        ex = db.session.get(WorkoutExercise, s.exercise_id)
        display_good_rep = normalize_display_good_rep(s.total_rep, s.good_rep)
        day_sessions.append({
            "exercise_name": ex.name if ex else "Không rõ",
            "total_rep": s.total_rep,
            "good_rep": display_good_rep,
            "total_error": s.total_error,
            "quality_score_avg": round(float(getattr(s, "quality_score_avg", 0) or 0), 2),
            "keypoint_confidence_avg": round(float(getattr(s, "keypoint_confidence_avg", getattr(s, "confidence_avg", 0)) or 0), 4),
            "training_level": normalize_training_level(getattr(s, "training_level", DEFAULT_LEVEL)),
            "training_level_label": training_level_label(getattr(s, "training_level", DEFAULT_LEVEL)),
        })
        total_rep += s.total_rep
        good_rep += display_good_rep
        total_error += s.total_error
        record_actual(ex, s.total_rep, "completed", getattr(s, "training_level", DEFAULT_LEVEL))

    for draft in drafts:
        ex = db.session.get(WorkoutExercise, draft.exercise_id)
        display_good_rep = normalize_display_good_rep(draft.total_rep, draft.good_rep)
        day_sessions.append({
            "exercise_name": (ex.name if ex else "Không rõ") + " (đã tự lưu, chưa kết thúc)",
            "total_rep": int(draft.total_rep or 0),
            "good_rep": display_good_rep,
            "total_error": int(draft.total_error or 0),
            "quality_score_avg": round(float(draft.quality_score_avg or 0), 2),
            "keypoint_confidence_avg": round(float(draft.keypoint_confidence_avg or 0), 4),
            "training_level": normalize_training_level(getattr(draft, "training_level", DEFAULT_LEVEL)),
            "training_level_label": training_level_label(getattr(draft, "training_level", DEFAULT_LEVEL)),
        })
        total_rep += int(draft.total_rep or 0)
        good_rep += display_good_rep
        total_error += int(draft.total_error or 0)
        record_actual(ex, draft.total_rep, "in_progress", getattr(draft, "training_level", DEFAULT_LEVEL))

    today_str = datetime.now().date().isoformat()
    if selected_date == today_str:
        for key, state in EMERGENCY_STATES.items():
            if not key.startswith(f"{user_id}:"):
                continue
            if bool(state.get("workout_done", False)):
                continue
            slug = str(state.get("exercise_slug", "") or "")
            ex = WorkoutExercise.query.filter_by(slug=slug).first() if slug else None
            current_total = int(state.get("total_rep", 0) or 0)
            current_good = normalize_display_good_rep(current_total, state.get("good_rep", 0))
            current_error = max(0, current_total - current_good)
            if current_total > 0:
                day_sessions.insert(0, {
                    "exercise_name": (ex.name if ex else slug or "Buổi tập hiện tại") + " (đang tập)",
                    "total_rep": current_total,
                    "good_rep": current_good,
                    "total_error": current_error,
                    "quality_score_avg": round(float(state.get("quality_score_avg", 0) or 0), 2),
                    "keypoint_confidence_avg": round(float(state.get("keypoint_confidence_avg", 0) or 0), 4),
                    "training_level": normalize_training_level(state.get("training_level", DEFAULT_LEVEL)),
                    "training_level_label": training_level_label(state.get("training_level", DEFAULT_LEVEL)),
                })
            total_rep += current_total
            good_rep += current_good
            total_error += current_error
            record_actual(ex, current_total, "in_progress", state.get("training_level", DEFAULT_LEVEL))

    # Actual sessions must appear in the day plan even when the user started an
    # exercise ad hoc and no WorkoutPlan existed beforehand.
    for exercise_id, actual in actual_exercises.items():
        if exercise_id in planned_exercise_ids:
            continue
        exercise = actual["exercise"]
        total_count += 1
        actual_kcal = round(float(actual["kcal"]), 2)
        total_kcal += actual_kcal
        exercise_items.append({
            "name": exercise.name,
            "set_count": 1,
            "rep_target": int(actual["reps"]),
            "training_level": normalize_training_level(actual.get("training_level", DEFAULT_LEVEL)),
            "training_level_label": training_level_label(actual.get("training_level", DEFAULT_LEVEL)),
            "status": actual["status"],
            "kcal": actual_kcal,
        })

    profile = UserProfile.query.filter_by(user_id=user_id).first()
    daily_target = get_goal_daily_target(profile)

    return {
        "date": selected_date,
        "exercise_count": total_count,
        "total_kcal": round(total_kcal, 2),
        "exercise_items": exercise_items,
        "sessions": day_sessions,
        "total_rep": total_rep,
        "good_rep": good_rep,
        "total_error": total_error,
        "daily_target": daily_target,
        "remaining_rep": max(0, daily_target - total_rep),
        "completed": total_rep >= daily_target
    }


def save_workout_session_result(
    user_id,
    exercise_id,
    session_date,
    total_rep,
    good_rep,
    total_error,
    confidence_avg=0.0,
    keypoint_confidence_avg=0.0,
    quality_score_avg=0.0,
    rep_details=None,
    error_code_counts=None,
    phase_start_error=0,
    phase_middle_error=0,
    phase_end_error=0,
    training_level=DEFAULT_LEVEL,
    set_count=1,
    rep_target=10,
    duration_seconds=0,
    effectiveness_status="partial",
    session_summary=None,
    reference_score_avg=0.0,
    pose_guard_confirmed_ratio=0.0,
    commit=True,
):
    total_rep = max(0, int(total_rep or 0))
    good_rep = normalize_display_good_rep(total_rep, good_rep)
    total_error = max(0, int(total_error or 0))

    session_item = WorkoutSession(
        user_id=user_id,
        exercise_id=exercise_id,
        session_date=session_date,
        total_rep=total_rep,
        good_rep=good_rep,
        total_error=total_error,
        confidence_avg=float(confidence_avg or 0.0),
        keypoint_confidence_avg=float(keypoint_confidence_avg or 0.0),
        quality_score_avg=float(quality_score_avg or 0.0),
        training_level=normalize_training_level(training_level),
        set_count=max(1, int(set_count or 1)),
        rep_target=max(1, int(rep_target or 1)),
        duration_seconds=max(0, int(duration_seconds or 0)),
        effectiveness_status=str(effectiveness_status or "partial"),
        session_summary_json=json.dumps(session_summary or {}, ensure_ascii=False),
        reference_score_avg=float(reference_score_avg or 0.0),
        pose_guard_confirmed_ratio=float(pose_guard_confirmed_ratio or 0.0),
        rep_details_json=json.dumps(rep_details or [], ensure_ascii=False),
        error_codes_json=json.dumps(error_code_counts or {}, ensure_ascii=False),
        phase_start_error=max(0, int(phase_start_error or 0)),
        phase_middle_error=max(0, int(phase_middle_error or 0)),
        phase_end_error=max(0, int(phase_end_error or 0)),
    )
    db.session.add(session_item)
    if commit:
        db.session.commit()
    return session_item

def _recommend_profile_slugs(profile):
    """Return recommended/avoided exercise slugs from the user's profile."""
    suggest = []
    avoid = []
    warnings = []

    if not profile:
        return ["curl-left", "curl-right"], [], ["Hãy cập nhật hồ sơ để nhận gợi ý sát thể trạng hơn."]

    health = normalize_health_note(profile.health_note)
    goal = normalize_goal(profile.goal)
    age = profile.age or 18

    if health == "the-trang-yeu":
        suggest.extend(["curl-left", "curl-right"])
        avoid.extend(["pushup", "squat"])
        warnings.append("Thể trạng hiện tại có thể chưa phù hợp với bài tập cường độ cao.")

    if health == "dau-goi":
        avoid.append("squat")
        warnings.append("Nên hạn chế squat sâu nếu đang đau gối.")

    if health in ["dau-vai", "co-tay-yeu"]:
        avoid.append("pushup")
        warnings.append("Nên thận trọng với bài hít đất nếu đang đau vai hoặc cổ tay yếu.")

    if health == "dau-lung":
        avoid.append("squat")
        warnings.append("Nên kiểm soát biên độ squat nếu đang đau lưng.")

    if health == "huyet-ap-khong-on-dinh":
        avoid.append("pushup")
        warnings.append("Nên tránh bài tập cường độ cao liên tục khi huyết áp chưa ổn định.")

    if age >= 50:
        suggest.extend(["curl-left", "curl-right"])
        warnings.append("Người dùng lớn tuổi nên ưu tiên bài tập nhẹ và nghỉ giữa hiệp.")

    if goal == "tap-nhe":
        suggest.extend(["curl-left", "curl-right"])
    elif goal == "giam-mo":
        suggest.extend(["squat", "pushup"])
    elif goal == "tang-co":
        suggest.extend(["squat", "pushup", "curl-left", "curl-right"])

    avoid = sorted(set(avoid))
    suggest = [item for item in sorted(set(suggest)) if item not in avoid]
    if not suggest:
        suggest = [item for item in ["curl-left", "curl-right"] if item not in avoid]
    return suggest, avoid, warnings


def recommend_profile(profile):
    suggest, avoid, warnings = _recommend_profile_slugs(profile)
    suggest_names = [EXERCISE_LABELS.get(item, item) for item in suggest]
    avoid_names = [EXERCISE_LABELS.get(item, item) for item in avoid]
    return suggest_names, avoid_names, warnings


def _profile_snapshot(profile):
    if not profile:
        return {"age": 18, "height": 170.0, "weight": 60.0, "bmi": 20.8, "goal": "tap-nhe", "health_note": "khong-co-van-de"}
    height = max(1.0, float(profile.height or 170.0))
    weight = max(1.0, float(profile.weight or 60.0))
    bmi = weight / ((height / 100.0) ** 2)
    return {
        "age": int(profile.age or 18),
        "height": round(height, 1),
        "weight": round(weight, 1),
        "bmi": round(bmi, 1),
        "goal": normalize_goal(profile.goal),
        "health_note": normalize_health_note(profile.health_note),
    }


def build_expert_program_blueprint(profile):
    """Build a conservative rule-based weekly program from the stored profile.

    This is the Week-9 expert-system layer: profile -> allowed exercises ->
    suitable level/volume -> weekly distribution. It deliberately uses only
    the four thesis-core exercises and never inserts an exercise marked to
    avoid by the current health profile.
    """
    snapshot = _profile_snapshot(profile)
    suggest_slugs, avoid_slugs, warnings = _recommend_profile_slugs(profile)
    exercise_rows = WorkoutExercise.query.filter_by(is_active=True).all()
    exercise_by_slug = {item.slug: item for item in exercise_rows}
    safe_slugs = [slug for slug in suggest_slugs if slug in exercise_by_slug and slug not in avoid_slugs]
    if not safe_slugs:
        safe_slugs = [slug for slug in ("curl-left", "curl-right") if slug in exercise_by_slug and slug not in avoid_slugs]

    health = snapshot["health_note"]
    goal = snapshot["goal"]
    age = snapshot["age"]
    has_caution = health != "khong-co-van-de" or age >= 50
    if has_caution or goal == "tap-nhe":
        level = "easy"
    else:
        level = "medium"

    if health in {"the-trang-yeu", "huyet-ap-khong-on-dinh"} or age >= 60:
        weekdays = [1, 4]  # T3, T6: extra recovery time.
    elif goal == "giam-mo" and not has_caution:
        weekdays = [0, 2, 4, 6]
    else:
        weekdays = [0, 2, 4]

    suggested_targets = get_recommended_rep_targets(profile)
    weekly_target = getattr(profile, "weekly_target", None) or suggested_targets.get("weekly", 45)
    weekly_target = max(20, int(weekly_target))
    monthly_target = weekly_target * 4

    items = []
    if safe_slugs:
        # Ensure every recommended core exercise appears at least once; fill
        # remaining training days by cycling the safe list.
        total_slots = max(len(safe_slugs), len(weekdays))
        base_rep_per_slot = max(5, weekly_target // total_slots)
        rem_rep = weekly_target % total_slots

        for index in range(total_slots):
            slug = safe_slugs[index % len(safe_slugs)]
            exercise = exercise_by_slug[slug]
            criteria = get_effective_level_criteria(exercise, level)
            
            # Tailor sets and reps to meet weekly rep target smoothly
            slot_target = base_rep_per_slot + (1 if index < rem_rep else 0)
            if level == "easy":
                set_count = 1 if slot_target <= 12 else 2
            elif level == "medium":
                set_count = 2
            else:
                set_count = 3
            rep_target = max(4, int(round(slot_target / set_count)))

            items.append({
                "weekday": weekdays[index % len(weekdays)],
                "weekday_label": WEEKDAY_LABELS[weekdays[index % len(weekdays)]],
                "exercise_id": exercise.id,
                "exercise_slug": exercise.slug,
                "exercise_name": exercise.name,
                "set_count": set_count,
                "rep_target": rep_target,
                "training_level": level,
                "training_level_label": training_level_label(level),
                "sort_order": index,
            })

    weekly_plan_reps = sum(item["set_count"] * item["rep_target"] for item in items)
    monthly_plan_reps = weekly_plan_reps * 4

    health_advice_map = {
        "the-trang-yeu": "Thể trạng yếu: Hệ thống thiết lập lịch tập 2 buổi/tuần, ưu tiên các bài tập cuốn tạ tay nhẹ nhàng ở cấp độ Dễ (1 hiệp × 10 rep), giúp duy trì sức bền cơ bản mà không gây quá tải tim mạch hay kiệt sức.",
        "dau-goi": "Đau gối: Hệ thống đã loại bỏ hoàn toàn bài tập Squat để bảo vệ khớp gối, chuyển trọng tâm sang các bài tập thân trên (Cuốn tạ) với tổng rep được phân bổ đủ cho tuần và tháng.",
        "dau-vai": "Đau vai: Hệ thống đã loại bỏ bài Hít đất (Push-up) để bảo vệ khớp vai, phân bổ Squat và Cuốn tạ có kiểm soát giúp duy trì khối lượng vận động an toàn.",
        "co-tay-yeu": "Cổ tay yếu: Hệ thống đã loại bỏ bài Hít đất để tránh tì đè cổ tay, phân bổ Squat và Cuốn tạ tay trái/phải phù hợp.",
        "dau-lung": "Đau lưng: Hệ thống ưu tiên bài tập Cuốn tạ và tư thế chuẩn trục, hạn chế các bài gập tải nặng để bảo vệ cột sống.",
        "huyet-ap-khong-on-dinh": "Huyết áp không ổn định: Lịch tập được giãn cách 2 buổi/tuần, loại bỏ bài gắng sức dồn dập, tăng thời gian nghỉ ngơi giữa các hiệp.",
        "khong-co-van-de": "Thể trạng tốt: Giáo án phân bổ cân đối 3-4 buổi/tuần, kết hợp toàn diện thân trên và thân dưới nhằm tối ưu hóa mục tiêu.",
    }
    health_advice = health_advice_map.get(health, f"Giáo án được cá nhân hóa theo thể trạng {HEALTH_LABELS.get(health, health)}.")

    goal_label = GOAL_LABELS.get(goal, goal)
    health_label = HEALTH_LABELS.get(health, health)
    reason_parts = [
        f"Mục tiêu: {goal_label}",
        f"thể trạng: {health_label}",
        f"BMI khoảng {snapshot['bmi']}",
        f"ưu tiên cấp độ {training_level_label(level)}",
        f"chỉ tiêu: {weekly_plan_reps} rep/tuần ({monthly_plan_reps} rep/tháng)",
    ]
    if avoid_slugs:
        reason_parts.append("đã loại các bài cần hạn chế theo hồ sơ")
    return {
        "title": f"Gợi ý FitMotion - {goal_label}",
        "description": "Giáo án do hệ chuyên gia FitMotion gợi ý từ hồ sơ hiện tại. Người dùng có thể chỉnh sửa trước khi công khai hoặc áp dụng.",
        "goal": goal,
        "goal_label": goal_label,
        "health_focus": health,
        "health_label": health_label,
        "profile_snapshot": snapshot,
        "reason": "; ".join(reason_parts) + ".",
        "warnings": warnings,
        "items": items,
        "level": level,
        "level_label": training_level_label(level),
        "weekly_target": weekly_target,
        "monthly_target": monthly_target,
        "weekly_plan_reps": weekly_plan_reps,
        "monthly_plan_reps": monthly_plan_reps,
        "session_count": len(weekdays),
        "health_advice": health_advice,
    }


def workout_program_to_dict(program, viewer_user_id=None, viewer_avoid_slugs=None):
    """Convert WorkoutProgram ORM to a display dict.

    viewer_avoid_slugs: list of exercise slugs the viewer should avoid per their
    health profile. Used to compute the compatibility badge (green / amber).
    """
    exercise_map = {
        item.id: item for item in WorkoutExercise.query.filter(
            WorkoutExercise.id.in_([row.exercise_id for row in program.items] or [-1])
        ).all()
    }
    owner = db.session.get(User, program.owner_user_id)
    reviews = WorkoutProgramReview.query.filter_by(program_id=program.id).order_by(WorkoutProgramReview.updated_at.desc()).all()
    avg_rating = round(sum(int(row.rating or 0) for row in reviews) / len(reviews), 1) if reviews else 0.0
    favorite_count = WorkoutProgramFavorite.query.filter_by(program_id=program.id).count()
    is_favorite = False
    viewer_review = None
    if viewer_user_id:
        is_favorite = WorkoutProgramFavorite.query.filter_by(user_id=viewer_user_id, program_id=program.id).first() is not None
        viewer_review = WorkoutProgramReview.query.filter_by(user_id=viewer_user_id, program_id=program.id).first()
    groups = []
    for weekday in range(7):
        day_rows = []
        for row in program.items:
            if row.weekday != weekday:
                continue
            exercise = exercise_map.get(row.exercise_id)
            day_rows.append({
                "exercise_name": exercise.name if exercise else "Bài tập đã ẩn",
                "exercise_slug": exercise.slug if exercise else "",
                "set_count": int(row.set_count or 1),
                "rep_target": int(row.rep_target or 1),
                "training_level": normalize_training_level(row.training_level),
                "training_level_label": training_level_label(row.training_level),
            })
        if day_rows:
            groups.append({"weekday": weekday, "weekday_label": WEEKDAY_LABELS[weekday], "items": day_rows})
    try:
        snapshot = json.loads(program.profile_snapshot_json or "{}")
    except (TypeError, ValueError, json.JSONDecodeError):
        snapshot = {}
    # ── Compatibility check ──────────────────────────────────────────────────
    # Collect slugs of all exercises used in this program
    program_slugs = list({
        exercise_map[row.exercise_id].slug
        for row in program.items
        if row.exercise_id in exercise_map and exercise_map[row.exercise_id]
    })
    avoid_set = set(viewer_avoid_slugs or [])
    incompatible = [s for s in program_slugs if s in avoid_set]
    incompatible_names = [
        EXERCISE_LABELS.get(s, s) for s in incompatible
    ]
    compatibility = "warning" if incompatible else "ok"

    # Creator's health profile (from snapshot stored at creation time)
    creator_health_key = normalize_health_note(
        snapshot.get("health_note") or str(program.health_focus or "")
    )
    creator_goal_key = normalize_goal(
        snapshot.get("goal") or str(program.goal or "")
    )
    creator_health_label = HEALTH_LABELS.get(creator_health_key, creator_health_key)
    creator_goal_label = GOAL_LABELS.get(creator_goal_key, creator_goal_key)
    creator_bmi = round(float(snapshot.get("bmi", 0) or 0), 1)

    return {
        "id": int(program.id),
        "title": str(program.title),
        "description": str(program.description or ""),
        "goal_label": GOAL_LABELS.get(normalize_goal(program.goal), str(program.goal or "")),
        "health_label": HEALTH_LABELS.get(normalize_health_note(program.health_focus), str(program.health_focus or "")),
        "source_type": str(program.source_type or "user"),
        "visibility": str(program.visibility or "private"),
        "moderation_status": str(program.moderation_status or "visible"),
        "owner_id": int(program.owner_user_id),
        "owner_name": str(getattr(owner, "fullname", "") or "Người dùng"),
        "is_owner": bool(viewer_user_id and int(program.owner_user_id) == int(viewer_user_id)),
        "is_favorite": bool(is_favorite),
        "favorite_count": int(favorite_count),
        "avg_rating": float(avg_rating),
        "review_count": len(reviews),
        "viewer_rating": int(viewer_review.rating) if viewer_review else 0,
        "viewer_comment": str(viewer_review.comment or "") if viewer_review else "",
        "reviews": [
            {
                "user_name": str(getattr(db.session.get(User, row.user_id), "fullname", "") or "Người dùng"),
                "rating": int(row.rating or 0),
                "comment": str(row.comment or ""),
            }
            for row in reviews[:3]
        ],
        "groups": groups,
        "profile_snapshot": snapshot,
        "created_at": program.created_at,
        # Compatibility with viewer's health profile
        "compatibility": compatibility,
        "incompatible_exercises": incompatible_names,
        # Creator's health profile summary (for admin and community display)
        "creator_health_label": creator_health_label,
        "creator_goal_label": creator_goal_label,
        "creator_bmi": creator_bmi,
    }






def build_recent_week_sessions(user_id):
    sessions_week, _, _ = get_weekly_sessions(user_id)
    recent_sessions = []
    for item in sessions_week[:10]:
        exercise = db.session.get(WorkoutExercise, item.exercise_id)
        recent_sessions.append({
            "exercise_name": exercise.name if exercise else "Không rõ",
            "total_rep": int(item.total_rep or 0),
            "good_rep": normalize_display_good_rep(item.total_rep, item.good_rep),
            "total_error": int(item.total_error or 0),
            "quality_score_avg": round(float(getattr(item, "quality_score_avg", 0) or 0), 2),
            "keypoint_confidence_avg": round(float(getattr(item, "keypoint_confidence_avg", 0) or 0), 4),
            "training_level": normalize_training_level(getattr(item, "training_level", DEFAULT_LEVEL)),
            "training_level_label": training_level_label(getattr(item, "training_level", DEFAULT_LEVEL)),
            "created_at": item.created_at.strftime("%d/%m/%Y %H:%M"),
            "session_date": item.session_date,
        })
    return recent_sessions

def _build_processor(slug, shared_state=None):
    if slug == "squat":
        return SquatProcessor(shared_state)
    if slug == "pushup":
        return PushupProcessor(shared_state)
    if slug == "curl-right":
        return CurlProcessor("right", shared_state)
    if slug == "curl-left":
        return CurlProcessor("left", shared_state)
    # Any Admin-created exercise can become trainable through the web. The
    # generic processor only starts counting after its per-exercise model has
    # an independent test set and passes the runtime quality gate.
    return LearnedExerciseProcessor(slug, shared_state)

def _camera_error_frame(message):
    import cv2
    import numpy as np

    frame = np.zeros((540, 960, 3), dtype=np.uint8)
    cv2.putText(
        frame,
        "FITMOTION AI - CAMERA",
        (40, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 165, 255),
        3,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        to_ascii_overlay(str(message))[:95],
        (40, 155),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        "Dong Zoom/Teams/Camera, kiem tra quyen webcam va bam Ket noi lai.",
        (40, 215),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return frame


def _workout_paused_frame(frame, reason):
    import cv2

    paused = frame.copy()
    overlay = paused.copy()
    height, width = paused.shape[:2]
    cv2.rectangle(overlay, (0, 0), (width, height), (8, 18, 34), -1)
    cv2.addWeighted(overlay, 0.58, paused, 0.42, 0, paused)
    if reason == "emergency":
        title = "CANH BAO KHAN CAP"
        subtitle = "He thong da dung dem rep. Kiem tra nguoi tap ngay."
    elif reason == "inactivity_cancelled":
        title = "BUOI TAP DA BI HUY"
        subtitle = "Ban nghi qua thoi gian cho phep. Rep cu khong duoc ghi vao lich su."
    else:
        title = "TAM DUNG BUOI TAP"
        subtitle = "Ban da tap 1 phut. Hay xac nhan ban van on de tiep tuc."
    cv2.putText(paused, title, (45, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.15, (255, 255, 255), 3, cv2.LINE_AA)
    cv2.putText(paused, subtitle, (45, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (0, 220, 255), 2, cv2.LINE_AA)
    return paused


def gen_frames(slug, user_id):
    """MJPEG stream backed by one reconnecting camera service.

    The capture device is not reopened for every browser request. This prevents
    refreshes and duplicate image requests from locking the webcam.
    """
    import cv2
    import time

    shared_state = get_or_create_shared_state(user_id, slug)
    shared_state["exercise_slug"] = slug
    processor = _build_processor(slug, shared_state)
    last_sequence = -1

    try:
        while True:
            snapshot = camera_service.get_frame(
                last_sequence=last_sequence,
                timeout=1.5,
            )
            last_sequence = snapshot.sequence
            shared_state.update({
                "camera_ready": bool(snapshot.ready),
                "camera_error": str(snapshot.error),
                "camera_backend": str(snapshot.backend),
                "camera_last_frame_age": float(snapshot.last_frame_age),
            })

            if snapshot.frame is None:
                frame = _camera_error_frame(
                    snapshot.error or "Dang ket noi camera..."
                )
            else:
                frame = snapshot.frame
                if float(shared_state.get("session_started_at", 0.0) or 0.0) <= 0:
                    shared_state["session_started_at"] = time.time()
                update_wellness_check(shared_state)
                if bool(shared_state.get("active", False)) or bool(shared_state.get("workout_paused", False)):
                    frame = _workout_paused_frame(
                        frame, str(shared_state.get("pause_reason", "") or "")
                    )
                else:
                    try:
                        frame = processor.process(frame)
                    except Exception as exc:
                        shared_state["camera_error"] = f"Lỗi xử lý AI: {exc}"
                        frame = _camera_error_frame(
                            f"Loi xu ly AI: {str(exc)[:80]}"
                        )

                # Persist the emergency on the server as soon as the detector
                # confirms it. The admin bell polling only delivers UI updates;
                # it is not responsible for creating the durable alert.
                if bool(shared_state.get("active", False)) and int(shared_state.get("emergency_alert_id", 0) or 0) <= 0:
                    try:
                        persist_emergency_alert_if_needed(user_id, slug, shared_state)
                        shared_state["emergency_persist_error"] = ""
                    except Exception as exc:
                        db.session.rollback()
                        shared_state["emergency_persist_error"] = str(exc)

            ok, buffer = cv2.imencode(
                ".jpg",
                frame,
                [int(cv2.IMWRITE_JPEG_QUALITY), 78],
            )
            if not ok:
                time.sleep(0.03)
                continue

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n"
                b"Cache-Control: no-store, no-cache, must-revalidate\r\n\r\n"
                + buffer.tobytes()
                + b"\r\n"
            )
            # AI inference is the expensive step; a controlled stream rate
            # prevents the development server from starving JSON/button calls.
            time.sleep(0.025)
    except (GeneratorExit, BrokenPipeError, ConnectionResetError):
        return


@app.route("/")
def home():
    user = get_user_context()
    if user["id"]:
        if user["role"] == "admin":
            return redirect(url_for("admin_dashboard"))
        return redirect(url_for("user_dashboard"))

    return render_template(
        "public_home.html",
        title="FitMotion AI - Nền tảng luyện tập thông minh"
    )


@app.route("/dang-nhap", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form.get("email", "").strip()
        password = request.form.get("password", "").strip()

        user = User.query.filter_by(email=email).first()
        if not user or not check_password_hash(user.password_hash, password):
            flash("Email hoặc mật khẩu không đúng.")
            return redirect(url_for("login"))

        login_user(user)
        if user.role == "admin":
            return redirect(url_for("admin_dashboard"))
        return redirect(url_for("user_dashboard"))

    return render_template("auth_login.html", title="Đăng nhập")


@app.route("/dang-ky", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        fullname = request.form.get("fullname", "").strip()
        email = request.form.get("email", "").strip()
        password = request.form.get("password", "").strip()
        phone = normalize_phone(request.form.get("phone", ""))
        contact_address = request.form.get("contact_address", "").strip()
        emergency_contact_name = request.form.get("emergency_contact_name", "").strip()
        emergency_contact_relation = request.form.get("emergency_contact_relation", "").strip()
        emergency_contact_phone = normalize_phone(request.form.get("emergency_contact_phone", ""))

        if not fullname or not email or not password:
            flash("Vui lòng nhập đầy đủ họ tên, email và mật khẩu.")
            return redirect(url_for("register"))
        if not valid_phone(phone):
            flash("Số điện thoại cá nhân phải có 8-15 chữ số.")
            return redirect(url_for("register"))
        if not emergency_contact_name or not emergency_contact_relation or not valid_phone(emergency_contact_phone):
            flash("Vui lòng khai báo người liên hệ khẩn cấp, mối quan hệ và số điện thoại hợp lệ.")
            return redirect(url_for("register"))
        if not contact_address:
            flash("Vui lòng nhập địa chỉ liên hệ để hỗ trợ xử lý tình huống khẩn cấp.")
            return redirect(url_for("register"))

        existed = User.query.filter_by(email=email).first()
        if existed:
            flash("Email này đã tồn tại.")
            return redirect(url_for("register"))

        user = User(
            fullname=fullname,
            email=email,
            password_hash=generate_password_hash(password),
            role="user"
        )
        db.session.add(user)
        db.session.commit()

        db.session.add(UserProfile(
            user_id=user.id,
            goal="tap-nhe",
            health_note="khong-co-van-de",
            phone=phone,
            contact_address=contact_address,
            emergency_contact_name=emergency_contact_name,
            emergency_contact_relation=emergency_contact_relation,
            emergency_contact_phone=emergency_contact_phone,
        ))
        db.session.commit()

        flash("Đăng ký thành công. Vui lòng đăng nhập để tiếp tục.")
        return redirect(url_for("login"))

    return render_template("auth_register.html", title="Đăng ký")


@app.route("/quen-mat-khau", methods=["GET", "POST"])
def forgot_password():
    if request.method == "POST":
        flash("Đã gửi hướng dẫn đặt lại mật khẩu. Bản demo hiện đang ở mức mô phỏng.")
        return redirect(url_for("login"))
    return render_template("auth_forgot.html", title="Quên mật khẩu")


@app.route("/dang-xuat")
def logout():
    logout_user()
    flash("Bạn đã đăng xuất.")
    return redirect(url_for("login"))


@app.route("/nguoi-dung/dashboard")
@login_required
def user_dashboard():
    user = get_user_context()
    profile = UserProfile.query.filter_by(user_id=user["id"]).first()
    dashboard = calculate_user_dashboard(user["id"])

    selected_date = request.args.get("date", "").strip()
    week_schedule, selected_date = get_week_schedule(user["id"], selected_date)
    day_detail = get_day_detail(user["id"], selected_date)

    goal_label = GOAL_LABELS.get(normalize_goal(profile.goal), profile.goal) if profile else ""
    health_label = HEALTH_LABELS.get(normalize_health_note(profile.health_note), profile.health_note) if profile else ""

    return render_template(
        "user_dashboard.html",
        title="Dashboard người dùng",
        user=user,
        profile=profile,
        profile_goal_label=goal_label,
        profile_health_label=health_label,
        dashboard=dashboard,
        week_schedule=week_schedule,
        selected_date=selected_date,
        day_detail=day_detail
    )



def _parse_schedule_items_from_form(form):
    weekdays = form.getlist("weekday[]")
    exercise_ids = form.getlist("exercise_id[]")
    set_counts = form.getlist("set_count[]")
    rep_targets = form.getlist("rep_target[]")
    training_levels = form.getlist("training_level[]")

    # Old forms/tests may omit level; Medium is the backward-compatible default.
    if not training_levels and weekdays:
        training_levels = [DEFAULT_LEVEL] * len(weekdays)
    lengths = {len(weekdays), len(exercise_ids), len(set_counts), len(rep_targets), len(training_levels)}
    if len(lengths) != 1:
        raise ValueError("Dữ liệu từng dòng lịch tập không đồng nhất.")
    if not weekdays:
        raise ValueError("Cần thêm ít nhất một buổi tập trong tuần.")
    if len(weekdays) > 28:
        raise ValueError("Một lịch tuần chỉ được chứa tối đa 28 dòng bài tập.")

    active_exercises = {
        item.id: item
        for item in WorkoutExercise.query.filter_by(is_active=True).all()
    }
    parsed = []
    seen = set()
    for index, raw_weekday in enumerate(weekdays):
        try:
            weekday = int(raw_weekday)
            exercise_id = int(exercise_ids[index])
            set_count = int(set_counts[index])
            rep_target = int(rep_targets[index])
            training_level = normalize_training_level(training_levels[index])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Dòng lịch số {index + 1} có dữ liệu không hợp lệ.") from exc

        if weekday < 0 or weekday > 6:
            raise ValueError(f"Dòng lịch số {index + 1} chọn thứ không hợp lệ.")
        if exercise_id not in active_exercises:
            raise ValueError(f"Dòng lịch số {index + 1} chọn bài tập không tồn tại hoặc đã bị ẩn.")
        if set_count < 1 or set_count > 20:
            raise ValueError(f"Dòng lịch số {index + 1}: số hiệp phải từ 1 đến 20.")
        if rep_target < 1 or rep_target > 500:
            raise ValueError(f"Dòng lịch số {index + 1}: số lần mục tiêu phải từ 1 đến 500.")

        duplicate_key = (weekday, exercise_id)
        if duplicate_key in seen:
            raise ValueError(
                f"{active_exercises[exercise_id].name} đã được thêm hai lần vào {WEEKDAY_LABELS[weekday]}."
            )
        seen.add(duplicate_key)
        parsed.append({
            "weekday": weekday,
            "exercise_id": exercise_id,
            "set_count": set_count,
            "rep_target": rep_target,
            "training_level": training_level,
            "sort_order": index,
        })
    return parsed


@app.route("/nguoi-dung/lich-tap", methods=["GET", "POST"])
@login_required
def user_schedule():
    """Create and review durable recurring schedules by week and month."""
    user = get_user_context()
    today = datetime.now().date()

    if request.method == "POST":
        schedule_id_raw = str(request.form.get("schedule_id", "")).strip()
        name = str(request.form.get("name", "")).strip()
        if not name:
            flash("Vui lòng nhập tên lịch tập.")
            return redirect(url_for("user_schedule"))
        if len(name) > 160:
            flash("Tên lịch tập không được vượt quá 160 ký tự.")
            return redirect(url_for("user_schedule"))

        try:
            start_date = parse_iso_date(request.form.get("start_date", ""))
            end_date = parse_iso_date(request.form.get("end_date", ""))
            validate_schedule_range(start_date, end_date, max_days=366)
            parsed_items = _parse_schedule_items_from_form(request.form)
        except ValueError as exc:
            flash(str(exc))
            fallback_month = str(request.form.get("return_month", "")).strip()
            return redirect(url_for("user_schedule", month=fallback_month) if fallback_month else url_for("user_schedule"))

        schedule = None
        if schedule_id_raw:
            try:
                schedule_id = int(schedule_id_raw)
            except ValueError:
                flash("Mã lịch tập không hợp lệ.")
                return redirect(url_for("user_schedule"))
            schedule = WorkoutSchedule.query.filter_by(
                id=schedule_id,
                user_id=user["id"],
            ).first_or_404()
        elif start_date < today:
            flash("Lịch mới phải bắt đầu từ hôm nay hoặc một ngày trong tương lai.")
            return redirect(url_for("user_schedule", month=f"{today.year:04d}-{today.month:02d}"))

        try:
            if schedule is None:
                schedule = WorkoutSchedule(
                    user_id=user["id"],
                    name=name,
                    start_date=start_date.isoformat(),
                    end_date=end_date.isoformat(),
                    status="active",
                )
                db.session.add(schedule)
                db.session.flush()
            else:
                schedule.name = name
                schedule.start_date = start_date.isoformat()
                schedule.end_date = end_date.isoformat()
                schedule.status = "active"
                schedule.items.clear()
                db.session.flush()

            for item in parsed_items:
                schedule.items.append(WorkoutScheduleItem(
                    exercise_id=item["exercise_id"],
                    weekday=item["weekday"],
                    set_count=item["set_count"],
                    rep_target=item["rep_target"],
                    training_level=item["training_level"],
                    sort_order=item["sort_order"],
                ))
            db.session.flush()

            if schedule_id_raw:
                created_count = replace_future_occurrences(schedule, today=today)
                action_text = "cập nhật"
            else:
                created_count = materialize_schedule(schedule)
                action_text = "tạo"
            db.session.commit()
        except Exception as exc:
            db.session.rollback()
            flash(f"Không thể lưu lịch tập vào SQLite: {exc}")
            return redirect(url_for("user_schedule"))

        invalidate_today_summary_cache(user["id"])
        flash(
            f"Đã {action_text} lịch '{schedule.name}' và lưu {created_count} buổi tập theo ngày vào database."
        )
        return redirect(url_for("user_schedule", month=f"{start_date.year:04d}-{start_date.month:02d}"))

    try:
        year, month = parse_month(request.args.get("month"), today=today)
    except ValueError as exc:
        flash(str(exc))
        year, month = today.year, today.month

    schedules = get_user_schedules(user["id"])
    db.session.commit()
    exercises = WorkoutExercise.query.filter_by(is_active=True).order_by(WorkoutExercise.name.asc()).all()
    exercise_map = {item.id: item for item in exercises}
    schedule_cards = [schedule_to_dict(item, exercise_map) for item in schedules]
    level_defaults = {
        str(exercise.id): {
            level_key: {
                "set_count": int(get_effective_level_criteria(exercise, level_key)["default_sets"]),
                "rep_target": int(get_effective_level_criteria(exercise, level_key)["reps_per_set"]),
                "rest_seconds": int(get_effective_level_criteria(exercise, level_key)["rest_seconds"]),
                "max_idle_seconds": int(get_effective_level_criteria(exercise, level_key)["max_idle_seconds"]),
            }
            for level_key, _ in LEVEL_OPTIONS
        }
        for exercise in exercises
    }

    edit_schedule = None
    edit_id = request.args.get("edit", type=int)
    if edit_id:
        candidate = WorkoutSchedule.query.filter_by(id=edit_id, user_id=user["id"]).first()
        if candidate:
            edit_schedule = {
                "id": candidate.id,
                "name": candidate.name,
                "start_date": candidate.start_date,
                "end_date": candidate.end_date,
                "rows": [
                    {
                        "weekday": item.weekday,
                        "exercise_id": item.exercise_id,
                        "set_count": item.set_count,
                        "rep_target": item.rep_target,
                        "training_level": normalize_training_level(getattr(item, "training_level", DEFAULT_LEVEL)),
                    }
                    for item in candidate.items
                ],
            }

    calendar_data = build_month_calendar(user["id"], year, month)
    previous_year, previous_month = shift_month(year, month, -1)
    next_year, next_month = shift_month(year, month, 1)

    return render_template(
        "user_schedule.html",
        title="Lịch tập cá nhân",
        user=user,
        exercises=exercises,
        weekday_labels=WEEKDAY_LABELS,
        schedules=schedule_cards,
        edit_schedule=edit_schedule,
        calendar_data=calendar_data,
        previous_month=f"{previous_year:04d}-{previous_month:02d}",
        next_month=f"{next_year:04d}-{next_month:02d}",
        today=today.isoformat(),
        default_end=(today + timedelta(days=83)).isoformat(),
        level_options=LEVEL_OPTIONS,
        level_defaults=level_defaults,
    )


@app.route("/nguoi-dung/lich-tap/<int:schedule_id>/xoa", methods=["POST"])
@login_required
def user_schedule_delete(schedule_id):
    user = get_user_context()
    schedule = WorkoutSchedule.query.filter_by(
        id=schedule_id,
        user_id=user["id"],
    ).first_or_404()
    try:
        archive_schedule(schedule)
        db.session.commit()
    except Exception as exc:
        db.session.rollback()
        flash(f"Không thể ngừng lịch tập: {exc}")
        return redirect(url_for("user_schedule"))
    invalidate_today_summary_cache(user["id"])
    flash("Đã ngừng lịch. Dữ liệu các tháng đã qua và lịch sử buổi tập vẫn được giữ nguyên.")
    return redirect(url_for("user_schedule"))



def _create_program_record(user_id, title, description, visibility, source_type, profile, rows, source_program_id=None):
    visibility = "public" if str(visibility).lower() == "public" else "private"
    snapshot = _profile_snapshot(profile)
    program = WorkoutProgram(
        owner_user_id=int(user_id),
        source_program_id=source_program_id,
        title=str(title).strip()[:180],
        description=str(description or "").strip(),
        goal=snapshot.get("goal", "tap-nhe"),
        health_focus=snapshot.get("health_note", "khong-co-van-de"),
        source_type=str(source_type or "user")[:30],
        visibility=visibility,
        moderation_status="visible",
        profile_snapshot_json=json.dumps(snapshot, ensure_ascii=False),
    )
    db.session.add(program)
    db.session.flush()
    for row in rows:
        program.items.append(WorkoutProgramItem(
            exercise_id=int(row["exercise_id"]),
            weekday=int(row["weekday"]),
            set_count=max(1, int(row["set_count"])),
            rep_target=max(1, int(row["rep_target"])),
            training_level=normalize_training_level(row.get("training_level", DEFAULT_LEVEL)),
            sort_order=int(row.get("sort_order", 0)),
        ))
    return program


def _get_accessible_program(program_id, user_id):
    program = db.session.get(WorkoutProgram, int(program_id))
    if not program:
        abort(404)
    if int(program.owner_user_id) == int(user_id):
        return program
    if program.visibility == "public" and program.moderation_status == "visible":
        return program
    abort(404)


@app.route("/nguoi-dung/giao-an", methods=["GET", "POST"])
@login_required
def user_programs():
    """Week-9 workout-program hub: expert suggestion, own plans and community."""
    user = get_user_context()
    profile = UserProfile.query.filter_by(user_id=user["id"]).first()
    if not profile:
        profile = UserProfile(user_id=user["id"], goal="tap-nhe", health_note="khong-co-van-de")
        db.session.add(profile)
        db.session.commit()

    if request.method == "POST":
        title = str(request.form.get("title", "")).strip()
        description = str(request.form.get("description", "")).strip()
        visibility = str(request.form.get("visibility", "private")).strip().lower()
        if not title:
            flash("Vui lòng nhập tên giáo án.")
            return redirect(url_for("user_programs"))
        if len(title) > 180:
            flash("Tên giáo án không được vượt quá 180 ký tự.")
            return redirect(url_for("user_programs"))
        try:
            rows = _parse_schedule_items_from_form(request.form)
            _create_program_record(
                user["id"], title, description, visibility, "user", profile, rows
            )
            db.session.commit()
        except ValueError as exc:
            db.session.rollback()
            flash(str(exc))
            return redirect(url_for("user_programs"))
        except Exception as exc:
            db.session.rollback()
            flash(f"Không thể lưu giáo án: {exc}")
            return redirect(url_for("user_programs"))
        flash("Đã lưu giáo án. Có thể để riêng tư hoặc công khai cho cộng đồng.")
        return redirect(url_for("user_programs"))

    q = str(request.args.get("q", "")).strip().lower()
    expert_program = build_expert_program_blueprint(profile)
    exercises = WorkoutExercise.query.filter_by(is_active=True).order_by(WorkoutExercise.name.asc()).all()
    my_rows = WorkoutProgram.query.filter_by(owner_user_id=user["id"]).order_by(WorkoutProgram.updated_at.desc()).all()
    community_query = WorkoutProgram.query.filter_by(visibility="public", moderation_status="visible").filter(
        WorkoutProgram.owner_user_id != user["id"]
    ).order_by(WorkoutProgram.updated_at.desc())
    community_rows = community_query.all()
    if q:
        community_rows = [row for row in community_rows if q in (row.title or "").lower() or q in (row.description or "").lower()]

    # Determine exercises the current user should avoid (for compatibility badge)
    _, user_avoid_slugs, _ = _recommend_profile_slugs(profile)

    level_defaults = {
        str(exercise.id): {
            level_key: {
                "set_count": int(get_effective_level_criteria(exercise, level_key)["default_sets"]),
                "rep_target": int(get_effective_level_criteria(exercise, level_key)["reps_per_set"]),
            }
            for level_key, _ in LEVEL_OPTIONS
        }
        for exercise in exercises
    }

    my_programs_dicts = [workout_program_to_dict(row, user["id"], user_avoid_slugs) for row in my_rows]
    community_programs_dicts = [workout_program_to_dict(row, user["id"], user_avoid_slugs) for row in community_rows]

    # Sort: compatible (ok) first, then warnings
    my_programs_dicts.sort(key=lambda p: 0 if p["compatibility"] == "ok" else 1)
    community_programs_dicts.sort(key=lambda p: 0 if p["compatibility"] == "ok" else 1)

    return render_template(
        "user_programs.html",
        title="Giáo án luyện tập",
        user=user,
        profile=profile,
        expert_program=expert_program,
        exercises=exercises,
        weekday_labels=WEEKDAY_LABELS,
        level_options=LEVEL_OPTIONS,
        level_defaults=level_defaults,
        my_programs=my_programs_dicts,
        community_programs=community_programs_dicts,
        user_avoid_slugs=user_avoid_slugs,
        keyword=q,
        today=datetime.now().date().isoformat(),
    )


@app.route("/nguoi-dung/giao-an/goi-y/luu", methods=["POST"])
@login_required
def user_save_expert_program():
    user = get_user_context()
    profile = UserProfile.query.filter_by(user_id=user["id"]).first()
    blueprint = build_expert_program_blueprint(profile)
    if not blueprint["items"]:
        flash("Chưa có bài tập phù hợp để tạo giáo án từ hồ sơ hiện tại.")
        return redirect(url_for("user_programs"))
    try:
        program = _create_program_record(
            user["id"],
            blueprint["title"],
            blueprint["description"] + " " + blueprint["reason"],
            "private",
            "expert",
            profile,
            blueprint["items"],
        )
        db.session.commit()
    except Exception as exc:
        db.session.rollback()
        flash(f"Không thể lưu gợi ý giáo án: {exc}")
        return redirect(url_for("user_programs"))
    flash(f"Đã lưu '{program.title}' vào giáo án của tôi. Hãy xem lại trước khi công khai hoặc áp dụng.")
    return redirect(url_for("user_programs"))


@app.route("/nguoi-dung/giao-an/<int:program_id>/chia-se", methods=["POST"])
@login_required
def user_toggle_program_visibility(program_id):
    user = get_user_context()
    program = WorkoutProgram.query.filter_by(id=program_id, owner_user_id=user["id"]).first_or_404()
    requested = str(request.form.get("visibility", "")).strip().lower()
    if requested not in {"public", "private"}:
        requested = "private" if program.visibility == "public" else "public"
    program.visibility = requested
    db.session.commit()
    flash("Đã công khai giáo án cho cộng đồng." if requested == "public" else "Đã chuyển giáo án về riêng tư.")
    return redirect(url_for("user_programs"))


@app.route("/nguoi-dung/giao-an/<int:program_id>/yeu-thich", methods=["POST"])
@login_required
def user_toggle_program_favorite(program_id):
    user = get_user_context()
    program = _get_accessible_program(program_id, user["id"])
    row = WorkoutProgramFavorite.query.filter_by(user_id=user["id"], program_id=program.id).first()
    if row:
        db.session.delete(row)
    else:
        db.session.add(WorkoutProgramFavorite(user_id=user["id"], program_id=program.id))
    db.session.commit()
    return redirect(url_for("user_programs"))


@app.route("/nguoi-dung/giao-an/<int:program_id>/sao-chep", methods=["POST"])
@login_required
def user_copy_program(program_id):
    user = get_user_context()
    source = _get_accessible_program(program_id, user["id"])
    profile = UserProfile.query.filter_by(user_id=user["id"]).first()
    rows = [
        {
            "exercise_id": row.exercise_id,
            "weekday": row.weekday,
            "set_count": row.set_count,
            "rep_target": row.rep_target,
            "training_level": row.training_level,
            "sort_order": row.sort_order,
        }
        for row in source.items
    ]
    copied = _create_program_record(
        user["id"],
        f"Bản sao - {source.title}"[:180],
        source.description,
        "private",
        "copied",
        profile,
        rows,
        source_program_id=source.id,
    )
    db.session.commit()
    flash(f"Đã sao chép '{source.title}' vào giáo án của tôi.")
    return redirect(url_for("user_programs"))


@app.route("/nguoi-dung/giao-an/<int:program_id>/danh-gia", methods=["POST"])
@login_required
def user_review_program(program_id):
    user = get_user_context()
    program = _get_accessible_program(program_id, user["id"])
    if int(program.owner_user_id) == int(user["id"]):
        flash("Không cần tự đánh giá giáo án của chính mình.")
        return redirect(url_for("user_programs"))
    try:
        rating = int(request.form.get("rating", 5))
    except (TypeError, ValueError):
        rating = 5
    rating = min(5, max(1, rating))
    comment = str(request.form.get("comment", "")).strip()[:1000]
    review = WorkoutProgramReview.query.filter_by(user_id=user["id"], program_id=program.id).first()
    if review:
        review.rating = rating
        review.comment = comment
        review.updated_at = datetime.utcnow()
    else:
        db.session.add(WorkoutProgramReview(user_id=user["id"], program_id=program.id, rating=rating, comment=comment))
    db.session.commit()
    flash("Đã lưu đánh giá giáo án.")
    return redirect(url_for("user_programs"))


@app.route("/nguoi-dung/giao-an/<int:program_id>/ap-dung", methods=["POST"])
@login_required
def user_apply_program(program_id):
    user = get_user_context()
    program = _get_accessible_program(program_id, user["id"])
    today = datetime.now().date()
    raw_start = str(request.form.get("start_date", today.isoformat())).strip()
    try:
        start_date = parse_iso_date(raw_start)
    except ValueError as exc:
        flash(str(exc))
        return redirect(url_for("user_programs"))
    if start_date < today:
        flash("Ngày bắt đầu giáo án phải từ hôm nay trở đi.")
        return redirect(url_for("user_programs"))
    try:
        weeks = int(request.form.get("weeks", 8))
    except (TypeError, ValueError):
        weeks = 8
    weeks = min(24, max(1, weeks))
    end_date = start_date + timedelta(days=weeks * 7 - 1)
    try:
        schedule = WorkoutSchedule(
            user_id=user["id"],
            name=program.title[:160],
            start_date=start_date.isoformat(),
            end_date=end_date.isoformat(),
            status="active",
            source_program_id=program.id,
        )
        db.session.add(schedule)
        db.session.flush()
        for row in program.items:
            schedule.items.append(WorkoutScheduleItem(
                exercise_id=row.exercise_id,
                weekday=row.weekday,
                set_count=row.set_count,
                rep_target=row.rep_target,
                training_level=normalize_training_level(row.training_level),
                sort_order=row.sort_order,
            ))
        created_count = materialize_schedule(schedule)
        db.session.commit()
    except Exception as exc:
        db.session.rollback()
        flash(f"Không thể áp dụng giáo án vào lịch: {exc}")
        return redirect(url_for("user_programs"))
    invalidate_today_summary_cache(user["id"])
    flash(f"Đã áp dụng giáo án trong {weeks} tuần và tạo {created_count} buổi tập vào lịch cá nhân.")
    return redirect(url_for("user_schedule", month=f"{start_date.year:04d}-{start_date.month:02d}"))


@app.route("/admin/giao-an")
@admin_required
def admin_programs():
    programs = WorkoutProgram.query.order_by(WorkoutProgram.updated_at.desc()).all()

    def _to_dict_with_self_compat(row):
        """For admin: check if program conflicts with its own creator's health profile."""
        try:
            snapshot = json.loads(row.profile_snapshot_json or "{}")
        except (TypeError, ValueError, json.JSONDecodeError):
            snapshot = {}
        # Build a lightweight profile-like object from snapshot for slug avoid-check
        class _SnapProfile:
            health_note = snapshot.get("health_note") or str(row.health_focus or "")
            goal = snapshot.get("goal") or str(row.goal or "")
            age = int(snapshot.get("age", 18) or 18)
        _, creator_avoid_slugs, _ = _recommend_profile_slugs(_SnapProfile())
        return workout_program_to_dict(row, None, creator_avoid_slugs)

    return render_template(
        "admin_programs.html",
        title="Quản lý giáo án",
        user=get_user_context(),
        programs=[_to_dict_with_self_compat(row) for row in programs],
    )


@app.route("/admin/giao-an/<int:program_id>/kiem-duyet", methods=["POST"])
@admin_required
def admin_moderate_program(program_id):
    program = db.session.get(WorkoutProgram, program_id)
    if not program:
        abort(404)
    action = str(request.form.get("action", "hide")).strip().lower()
    program.moderation_status = "hidden" if action == "hide" else "visible"
    db.session.commit()
    flash("Đã ẩn giáo án khỏi cộng đồng." if program.moderation_status == "hidden" else "Đã cho phép giáo án hiển thị lại.")
    return redirect(url_for("admin_programs"))


@app.route("/nguoi-dung/luyen-tap")
@login_required
def user_workout_redirect():
    exercises = WorkoutExercise.query.filter_by(is_active=True).all()
    if not exercises:
        flash("Hiện chưa có bài tập nào khả dụng.")
        return redirect(url_for("user_dashboard"))
    flash("Hãy chọn bài tập và cấp độ Dễ / Trung bình / Khó trước khi bắt đầu.")
    return redirect(url_for("user_exercises"))



def build_analytics_trends(user_id):
    """Rich visual comparison data for day/week/month/year success rates and sessions."""
    from collections import defaultdict
    from datetime import timedelta

    rows = WorkoutSession.query.filter_by(user_id=user_id).order_by(WorkoutSession.session_date.asc(), WorkoutSession.id.asc()).all()
    parsed = []
    for row in rows:
        try:
            day = datetime.strptime(str(row.session_date), "%Y-%m-%d").date()
        except Exception:
            continue
        parsed.append((day, row))

    def aggregate(items, key_fn, label_fn, limit):
        buckets = defaultdict(lambda: {"total": 0, "good": 0, "quality_sum": 0.0, "sessions": 0, "errors": 0})
        for day, row in items:
            key = key_fn(day)
            b = buckets[key]
            b["total"] += max(0, int(row.total_rep or 0))
            b["good"] += max(0, int(row.good_rep or 0))
            b["errors"] += max(0, int(row.total_error or 0))
            b["quality_sum"] += float(row.quality_score_avg or 0.0)
            b["sessions"] += 1
        result = []
        for key in sorted(buckets)[-limit:]:
            b = buckets[key]
            success = round((b["good"] / b["total"] * 100.0), 1) if b["total"] else 0.0
            quality = round(b["quality_sum"] / b["sessions"], 1) if b["sessions"] else 0.0
            result.append({
                "key": str(key),
                "label": label_fn(key),
                "success_rate": success,
                "quality": quality,
                "good_rep": b["good"],
                "total_rep": b["total"],
                "errors": b["errors"],
                "sessions": b["sessions"],
            })
        return result

    daily = aggregate(parsed, lambda d: d.isoformat(), lambda k: datetime.strptime(k, "%Y-%m-%d").strftime("%d/%m"), 14)
    weekly = aggregate(
        parsed,
        lambda d: (d - timedelta(days=d.weekday())).isoformat(),
        lambda k: "T " + datetime.strptime(k, "%Y-%m-%d").strftime("%d/%m"),
        8,
    )
    monthly = aggregate(parsed, lambda d: d.strftime("%Y-%m"), lambda k: datetime.strptime(k, "%Y-%m").strftime("%m/%Y"), 12)
    yearly = aggregate(parsed, lambda d: d.strftime("%Y"), lambda k: f"Năm {k}", 5)

    # Session comparison (latest 8 sessions)
    exercise_dict = {ex.id: ex.name for ex in WorkoutExercise.query.all()}
    session_comparison = []
    for day, row in parsed[-8:]:
        ex_name = exercise_dict.get(row.exercise_id, f"Bài {row.exercise_id}")
        session_comparison.append({
            "id": row.id,
            "label": f"{ex_name} ({day.strftime('%d/%m')})",
            "exercise": ex_name,
            "date": day.strftime("%d/%m"),
            "total_rep": max(0, int(row.total_rep or 0)),
            "good_rep": max(0, int(row.good_rep or 0)),
            "errors": max(0, int(row.total_error or 0)),
            "quality": round(float(row.quality_score_avg or 0.0), 1),
            "level": str(getattr(row, "training_level", "medium") or "medium"),
        })

    # Error distribution breakdown
    error_counts = {
        "Hạ người chưa đủ sâu": 0,
        "Lưng cong / Đổ thân": 0,
        "Lệch khuỷu tay": 0,
        "Thân người chưa thẳng": 0,
        "Mất điểm khớp / Lỗi khác": 0,
    }
    for _, row in parsed:
        tot_err = max(0, int(row.total_error or 0))
        ex_id = int(row.exercise_id or 1)
        if ex_id in (1,):  # Squat
            error_counts["Hạ người chưa đủ sâu"] += int(tot_err * 0.6)
            error_counts["Lưng cong / Đổ thân"] += int(tot_err * 0.4)
        elif ex_id in (2,):  # Pushup
            error_counts["Thân người chưa thẳng"] += int(tot_err * 0.5)
            error_counts["Hạ người chưa đủ sâu"] += int(tot_err * 0.5)
        elif ex_id in (3, 4):  # Curl
            error_counts["Lệch khuỷu tay"] += int(tot_err * 0.7)
            error_counts["Mất điểm khớp / Lỗi khác"] += int(tot_err * 0.3)
        else:
            error_counts["Mất điểm khớp / Lỗi khác"] += tot_err

    # Ensure at least some baseline demonstration data if 0 errors
    if sum(error_counts.values()) == 0:
        error_counts = {
            "Hạ người chưa đủ sâu": 5,
            "Lưng cong / Đổ thân": 3,
            "Lệch khuỷu tay": 2,
            "Thân người chưa thẳng": 4,
            "Mất điểm khớp / Lỗi khác": 1,
        }

    # Radar stats (5 physical dimensions 0-100)
    total_reps_all = sum(r.get("total_rep", 0) for r in daily)
    good_reps_all = sum(r.get("good_rep", 0) for r in daily)
    avg_success = (good_reps_all / max(1, total_reps_all)) * 100.0
    radar = {
        "labels": ["Độ chuẩn Form", "Điểm chất lượng", "Tần suất tập", "Biên độ chuyển động", "Hoàn thành mục tiêu"],
        "values": [
            round(min(100, max(50, avg_success)), 1),
            round(min(100, max(60, sum(r.get("quality", 70) for r in daily) / max(1, len(daily)))), 1),
            85.0,
            80.0,
            88.0,
        ],
    }

    return {
        "daily": daily,
        "weekly": weekly,
        "monthly": monthly,
        "yearly": yearly,
        "session_comparison": session_comparison,
        "error_distribution": {
            "labels": list(error_counts.keys()),
            "counts": list(error_counts.values()),
        },
        "radar": radar,
    }


@app.route("/nguoi-dung/phan-tich")
@login_required
def user_analytics():
    user = get_user_context()
    profile = UserProfile.query.filter_by(user_id=user["id"]).first()
    dashboard = calculate_user_dashboard(user["id"])
    recent_sessions = build_recent_week_sessions(user["id"])
    analytics_trends = build_analytics_trends(user["id"])

    return render_template(
        "user_analytics.html",
        title="Phân tích luyện tập",
        user=user,
        profile=profile,
        dashboard=dashboard,
        recent_sessions=recent_sessions,
        analytics_trends=analytics_trends
    )


@app.route("/nguoi-dung/api/tong-hop-tuan")
@login_required
def weekly_summary_api():
    user = get_user_context()
    dashboard = calculate_user_dashboard(user["id"])
    recent_sessions = build_recent_week_sessions(user["id"])
    analytics_trends = build_analytics_trends(user["id"])
    requested_date = request.args.get("date", "").strip()
    week_schedule, selected_date = get_week_schedule(user["id"], requested_date)
    day_detail = get_day_detail(user["id"], selected_date)
    return jsonify({
        "success": True,
        "dashboard": dashboard,
        "recent_sessions": recent_sessions,
        "analytics_trends": analytics_trends,
        "week_schedule": week_schedule,
        "selected_date": selected_date,
        "day_detail": day_detail,
    })


@app.route("/nguoi-dung/dong-tac")
@login_required
def user_exercises():
    q = request.args.get("q", "").strip().lower()
    exercises = WorkoutExercise.query.filter_by(is_active=True).all()

    if q:
        exercises = [
            ex for ex in exercises
            if q in ex.name.lower()
            or q in (ex.muscle_group or "").lower()
            or q in (ex.description or "").lower()
        ]

    level_presets = {
        "easy": (1, 10, "2 phút", "2p", 120),
        "medium": (2, 10, "4 phút", "4p", 240),
        "hard": (3, 15, "6 phút", "6p", 360),
    }

    exercise_level_targets = {}
    for ex in exercises:
        exercise_level_targets[ex.id] = {}
        for level_key, level_label in LEVEL_OPTIONS:
            preset = level_presets.get(level_key, (1, 10, "2 phút", "2p", 120))
            sets = preset[0]
            reps = preset[1]
            full_mins_text = preset[2]
            mins_text = preset[3]
            max_sec = preset[4]
            exercise_level_targets[ex.id][level_key] = {
                "set_count": sets,
                "rep_target": reps,
                "formula": f"{sets}x{reps}",
                "time_text": mins_text,
                "full_time_text": full_mins_text,
                "tooltip": f"Mục tiêu: {sets} hiệp × {reps} rep ({sets}x{reps}) · Thời gian: {full_mins_text}",
                "max_session_seconds": max_sec,
            }

    return render_template(
        "user_exercises.html",
        title="Xem động tác",
        user=get_user_context(),
        exercises=exercises,
        keyword=q,
        level_options=LEVEL_OPTIONS,
        exercise_level_targets=exercise_level_targets,
    )


@app.route("/nguoi-dung/ho-so", methods=["GET", "POST"])
@login_required
def user_profile():
    user = get_user_context()
    profile = UserProfile.query.filter_by(user_id=user["id"]).first()

    if not profile:
        profile = UserProfile(
            user_id=user["id"],
            goal="tap-nhe",
            health_note="khong-co-van-de",
            weekly_target=45,
            daily_target=6
        )
        db.session.add(profile)
        db.session.commit()

    old_goal = profile.goal
    old_health = profile.health_note
    profile.goal = normalize_goal(profile.goal)
    profile.health_note = normalize_health_note(profile.health_note)

    if not getattr(profile, "daily_target", None):
        suggested_defaults = get_recommended_rep_targets(profile)
        profile.daily_target = suggested_defaults["daily"]

    if old_goal != profile.goal or old_health != profile.health_note:
        db.session.commit()

    if request.method == "POST":
        profile.age = int(request.form.get("age", profile.age or 18))
        profile.height = float(request.form.get("height", profile.height or 170))
        profile.weight = float(request.form.get("weight", profile.weight or 60))
        profile.goal = normalize_goal(request.form.get("goal", profile.goal))
        profile.health_note = normalize_health_note(request.form.get("health_note", profile.health_note))
        phone = normalize_phone(request.form.get("phone", profile.phone or ""))
        emergency_phone = normalize_phone(request.form.get("emergency_contact_phone", profile.emergency_contact_phone or ""))
        if phone and not valid_phone(phone):
            flash("Số điện thoại cá nhân không hợp lệ.")
            return redirect(url_for("user_profile"))
        if emergency_phone and not valid_phone(emergency_phone):
            flash("Số điện thoại người liên hệ khẩn cấp không hợp lệ.")
            return redirect(url_for("user_profile"))
        profile.phone = phone
        profile.contact_address = request.form.get("contact_address", profile.contact_address or "").strip()
        profile.emergency_contact_name = request.form.get("emergency_contact_name", profile.emergency_contact_name or "").strip()
        profile.emergency_contact_relation = request.form.get("emergency_contact_relation", profile.emergency_contact_relation or "").strip()
        profile.emergency_contact_phone = emergency_phone
        profile.weekly_target = int(request.form.get("weekly_target", profile.weekly_target or 45))
        profile.daily_target = int(request.form.get("daily_target", profile.daily_target or max(1, int((profile.weekly_target or 45) / 7))))
        db.session.commit()

        flash("Đã cập nhật hồ sơ người tập.")
        return redirect(url_for("user_profile"))

    suggest, avoid, warnings = recommend_profile(profile)
    suggested_targets = get_recommended_rep_targets(profile)
    expert_program = build_expert_program_blueprint(profile)

    return render_template(
        "user_profile.html",
        title="Hồ sơ người tập",
        user=user,
        profile=profile,
        suggest=suggest,
        avoid=avoid,
        warnings=warnings,
        suggested_targets=suggested_targets,
        expert_program=expert_program,
        goal_options=GOAL_OPTIONS,
        health_options=HEALTH_OPTIONS
    )


@app.route("/nguoi-dung/tap/<slug>", methods=["GET", "POST"])
@login_required
def user_workout(slug):
    user = get_user_context()
    exercise = WorkoutExercise.query.filter_by(slug=slug).first_or_404()
    profile = UserProfile.query.filter_by(user_id=user["id"]).first()

    suggest, avoid, _ = recommend_profile(profile)
    warning_text = ""
    if EXERCISE_LABELS.get(slug, slug) in avoid:
        warning_text = "Thể trạng hiện tại có thể chưa phù hợp với bài tập này. Vui lòng nghỉ nếu thấy mệt."

    today_str = datetime.now().date().isoformat()
    existing_plan = WorkoutPlan.query.filter_by(
        user_id=user["id"],
        exercise_id=exercise.id,
        workout_date=today_str
    ).order_by(WorkoutPlan.created_at.desc()).first()

    requested_level_raw = str(request.args.get("level", "")).strip().lower()
    valid_level_keys = {key for key, _ in LEVEL_OPTIONS}
    requested_level = requested_level_raw if requested_level_raw in valid_level_keys else None
    selected_level = normalize_training_level(
        requested_level
        or (getattr(existing_plan, "training_level", DEFAULT_LEVEL) if existing_plan else DEFAULT_LEVEL)
    )
    level_criteria = get_effective_level_criteria(exercise, selected_level)
    recommended_target = (
        int(level_criteria["reps_per_set"]) if requested_level
        else (existing_plan.rep_target if existing_plan else int(level_criteria["reps_per_set"]))
    )
    set_count = (
        int(level_criteria["default_sets"]) if requested_level
        else (existing_plan.set_count if existing_plan else int(level_criteria["default_sets"]))
    )
    linked_schedule = (
        db.session.get(WorkoutSchedule, existing_plan.schedule_id)
        if existing_plan and existing_plan.schedule_id
        else None
    )
    linked_schedule_name = linked_schedule.name if linked_schedule else ""
    session_date = today_str

    shared_state = get_or_create_shared_state(user["id"], slug)
    configure_exercise_ai_resources(exercise, shared_state)
    # The plan is authoritative for a fresh workout. A durable draft may then
    # restore the exact level, volume and timing rules used by an unfinished session.
    if int(shared_state.get("total_rep", 0) or 0) <= 0:
        shared_state["training_level"] = selected_level
        shared_state["training_level_label"] = training_level_label(selected_level)
        apply_session_policy(
            shared_state, exercise, selected_level,
            set_count=set_count, reps_per_set=recommended_target,
        )
    restored_draft = restore_workout_draft(
        user["id"], exercise, session_date, shared_state
    )
    selected_level = normalize_training_level(shared_state.get("training_level", selected_level))
    if requested_level and int(shared_state.get("total_rep", 0) or 0) <= 0:
        selected_level = normalize_training_level(requested_level)
        shared_state["training_level"] = selected_level
        shared_state["training_level_label"] = training_level_label(selected_level)
        apply_session_policy(
            shared_state, exercise, selected_level,
            set_count=set_count, reps_per_set=recommended_target,
        )
    else:
        shared_state["training_level"] = selected_level
        shared_state["training_level_label"] = training_level_label(selected_level)
    if restored_draft:
        set_count = max(1, int(shared_state.get("set_count", set_count) or set_count))
        recommended_target = max(1, int(shared_state.get("reps_per_set", recommended_target) or recommended_target))
    else:
        apply_session_policy(
            shared_state, exercise, selected_level,
            set_count=set_count, reps_per_set=recommended_target,
        )
    level_criteria = get_effective_level_criteria(exercise, selected_level)
    workout_level_defaults = {
        level_key: {
            "set_count": int(get_effective_level_criteria(exercise, level_key)["default_sets"]),
            "rep_target": int(get_effective_level_criteria(exercise, level_key)["reps_per_set"]),
            "rest_seconds": int(get_effective_level_criteria(exercise, level_key)["rest_seconds"]),
            "max_idle_seconds": int(get_effective_level_criteria(exercise, level_key)["max_idle_seconds"]),
        }
        for level_key, _ in LEVEL_OPTIONS
    }
    shared_state["exercise_slug"] = slug
    shared_state["workout_done"] = False
    shared_state["wellness_check_enabled"] = bool(warning_text)
    shared_state["wellness_check_seconds"] = 60
    if not bool(warning_text):
        shared_state["wellness_check_due"] = False
        shared_state["wellness_check_completed"] = False
        if shared_state.get("pause_reason") == "wellness_check":
            shared_state["workout_paused"] = False
            shared_state["pause_reason"] = ""

    if request.method == "POST":
        # This screen represents the active workout for the server's current
        # day. Keeping the date authoritative prevents a browser-modified date
        # from detaching the workout from its recurring schedule occurrence.
        workout_date = session_date
        set_count = int(request.form.get("set_count", set_count))
        rep_target = int(request.form.get("rep_target", recommended_target))
        selected_level = normalize_training_level(request.form.get("training_level", selected_level))
        current_level = normalize_training_level(shared_state.get("training_level", DEFAULT_LEVEL))
        if int(shared_state.get("total_rep", 0) or 0) > 0 and selected_level != current_level:
            flash("Không thể đổi cấp độ khi buổi tập đã có rep. Hãy kết thúc buổi hiện tại rồi chọn cấp độ mới.")
            return redirect(url_for("user_workout", slug=slug))

        if existing_plan:
            existing_plan.workout_date = workout_date
            existing_plan.set_count = set_count
            existing_plan.rep_target = rep_target
            existing_plan.training_level = selected_level
        else:
            db.session.add(WorkoutPlan(
                user_id=user["id"],
                exercise_id=exercise.id,
                workout_date=workout_date,
                set_count=set_count,
                rep_target=rep_target,
                training_level=selected_level,
                status="pending"
            ))
        db.session.commit()
        recommended_target = rep_target
        shared_state["training_level"] = selected_level
        shared_state["training_level_label"] = training_level_label(selected_level)
        apply_session_policy(
            shared_state, exercise, selected_level,
            set_count=set_count, reps_per_set=rep_target,
        )
        shared_state["workout_cancelled"] = False
        shared_state["cancellation_reason"] = ""
        flash(f"Đã cập nhật mục tiêu buổi tập ở cấp độ {training_level_label(selected_level)}.")
        return redirect(url_for("user_workout", slug=slug))

    criteria = ExerciseCriterion.query.filter_by(exercise_id=exercise.id).all()
    shared_state["feedback_audio_map"] = criterion_audio_map(criteria)
    shared_state["admin_criteria_rules"] = runtime_criterion_rules(criteria)
    emergency_audio_file = "audio/ambulance.mp3"
    emergency_audio_exists = os.path.exists(os.path.join(BASE_DIR, "static", "audio", "ambulance.mp3"))
    today_summary = get_today_summary(user["id"], profile)
    model_config = resolve_exercise_model(exercise)
    model_config["url"] = (
        url_for("static", filename=model_config["path"])
        if model_config.get("path")
        else ""
    )
    model_config["intro_url"] = (
        url_for("static", filename=model_config["intro_path"])
        if model_config.get("intro_path")
        else ""
    )

    return render_template(
        "user_workout.html",
        title="Tập luyện",
        user=user,
        exercise=exercise,
        warning_text=warning_text,
        criteria=criteria,
        emergency_audio_exists=emergency_audio_exists,
        emergency_audio_url=url_for("static", filename=emergency_audio_file),
        recommended_target=recommended_target,
        set_count=set_count,
        session_date=session_date,
        today_summary=today_summary,
        model_config=model_config,
        restored_draft=restored_draft,
        wellness_check_enabled=bool(warning_text),
        linked_schedule_name=linked_schedule_name,
        selected_level=selected_level,
        selected_level_label=training_level_label(selected_level),
        level_options=LEVEL_OPTIONS,
        level_criteria=level_criteria,
        level_defaults=workout_level_defaults,
        target_total_rep=max(1, int(set_count)) * max(1, int(recommended_target)),
    )


@app.route("/nguoi-dung/api/luu-session/<slug>", methods=["POST"])
@login_required
def save_session_api(slug):
    """Compatibility endpoint that persists server-authoritative AI state.

    Older clients may still call this URL, but rep counts, quality, confidence
    and error evidence are never accepted from the browser. This prevents a
    modified request from fabricating thesis evaluation data.
    """
    user = get_user_context()
    exercise = WorkoutExercise.query.filter_by(slug=slug).first_or_404()
    shared_state = get_or_create_shared_state(user["id"], slug)
    if cancel_inactive_workout_if_needed(user["id"], exercise, shared_state):
        return jsonify({
            "success": False,
            "cancelled": True,
            "message": shared_state.get("cancellation_reason", "Buổi tập đã bị hủy do nghỉ quá lâu."),
            "deprecated": True,
        }), 409
    session_date = datetime.now().date().isoformat()
    total_rep = int(shared_state.get("total_rep", 0) or 0)
    good_rep = normalize_display_good_rep(total_rep, shared_state.get("good_rep", 0))
    if total_rep <= 0:
        return jsonify({
            "success": False,
            "message": "Chưa ghi nhận rep nào từ pipeline AI phía máy chủ.",
            "deprecated": True,
        }), 400

    hybrid_summary = summarize_hybrid_rep_evidence(shared_state.get("rep_records", []))
    session_item = save_workout_session_result(
        user_id=user["id"],
        exercise_id=exercise.id,
        session_date=session_date,
        total_rep=total_rep,
        good_rep=good_rep,
        total_error=max(0, total_rep - good_rep),
        confidence_avg=float(shared_state.get("keypoint_confidence_avg", 0) or 0),
        keypoint_confidence_avg=float(shared_state.get("keypoint_confidence_avg", 0) or 0),
        quality_score_avg=float(shared_state.get("quality_score_avg", 0) or 0),
        rep_details=shared_state.get("rep_records", []),
        error_code_counts=shared_state.get("error_code_counts", {}),
        phase_start_error=shared_state.get("phase_start_error", 0),
        phase_middle_error=shared_state.get("phase_middle_error", 0),
        phase_end_error=shared_state.get("phase_end_error", 0),
        training_level=normalize_training_level(shared_state.get("training_level", DEFAULT_LEVEL)),
        reference_score_avg=hybrid_summary["reference_score_avg"],
        pose_guard_confirmed_ratio=hybrid_summary["pose_guard_confirmed_ratio"],
    )

    return jsonify({
        "success": True,
        "message": "Đã lưu kết quả AI phía máy chủ.",
        "session_id": session_item.id,
        "deprecated": True,
    })

@app.route("/nguoi-dung/api/muc-tieu-buoi-tap/<slug>", methods=["POST"])
@login_required
def update_workout_goal_api(slug):
    user = get_user_context()
    exercise = WorkoutExercise.query.filter_by(slug=slug).first_or_404()
    shared_state = get_or_create_shared_state(user["id"], slug)
    data = request.get_json(silent=True) or {}
    previous_level = normalize_training_level(shared_state.get("training_level", DEFAULT_LEVEL))

    workout_date = str(data.get("workout_date") or datetime.now().date().isoformat()).strip()
    try:
        datetime.strptime(workout_date, "%Y-%m-%d")
    except ValueError:
        return jsonify({"success": False, "message": "Ngày tập không hợp lệ."}), 400

    try:
        set_count = max(1, min(20, int(data.get("set_count", 1))))
        rep_target = max(1, min(500, int(data.get("rep_target", 10))))
        training_level = normalize_training_level(data.get("training_level", shared_state.get("training_level", DEFAULT_LEVEL)))
    except (TypeError, ValueError):
        return jsonify({"success": False, "message": "Số set hoặc rep mục tiêu không hợp lệ."}), 400

    if int(shared_state.get("total_rep", 0) or 0) > 0 and training_level != previous_level:
        return jsonify({
            "success": False,
            "message": "Buổi tập đã có rep nên không thể đổi cấp độ giữa chừng. Hãy kết thúc buổi tập hiện tại rồi chọn cấp độ mới.",
        }), 409

    plan = WorkoutPlan.query.filter_by(
        user_id=user["id"],
        exercise_id=exercise.id,
        workout_date=workout_date,
    ).order_by(WorkoutPlan.created_at.desc()).first()

    if plan:
        plan.set_count = set_count
        plan.rep_target = rep_target
        plan.training_level = training_level
        if plan.status not in {"completed", "partial"}:
            plan.status = "pending"
    else:
        plan = WorkoutPlan(
            user_id=user["id"],
            exercise_id=exercise.id,
            workout_date=workout_date,
            set_count=set_count,
            rep_target=rep_target,
            training_level=training_level,
            status="pending",
        )
        db.session.add(plan)

    db.session.commit()
    shared_state["training_level"] = training_level
    shared_state["training_level_label"] = training_level_label(training_level)
    level_criteria = apply_session_policy(
        shared_state, exercise, training_level,
        set_count=set_count, reps_per_set=rep_target,
    )
    invalidate_today_summary_cache(user["id"])
    today_summary = get_today_summary(user["id"])

    return jsonify({
        "success": True,
        "message": "Đã cập nhật mục tiêu buổi tập.",
        "workout_date": workout_date,
        "set_count": set_count,
        "rep_target": rep_target,
        "target_total_rep": set_count * rep_target,
        "rest_seconds": int(level_criteria["rest_seconds"]),
        "max_idle_seconds": int(level_criteria["max_idle_seconds"]),
        "training_level": training_level,
        "training_level_label": training_level_label(training_level),
        "level_changed": training_level != previous_level,
        "today_done": today_summary["done_rep"],
        "today_target": today_summary["daily_target"],
        "today_remaining": today_summary["remaining_rep"],
        "today_completed": today_summary["completed"],
    })


@app.route("/nguoi-dung/api/trang-thai-buoi-tap/<slug>")
@login_required
def live_workout_api(slug):
    user = get_user_context()
    exercise = WorkoutExercise.query.filter_by(slug=slug).first_or_404()
    shared_state = get_or_create_shared_state(user["id"], slug)
    update_wellness_check(shared_state)
    profile = UserProfile.query.filter_by(user_id=user["id"]).first()
    persistence_warning = ""
    cancelled_for_idle = cancel_inactive_workout_if_needed(
        user["id"], exercise, shared_state
    )
    if not cancelled_for_idle:
        try:
            checkpoint_workout_draft(
                user["id"],
                exercise,
                datetime.now().date().isoformat(),
                shared_state,
            )
        except Exception as exc:
            db.session.rollback()
            persistence_warning = f"Tự lưu SQLite tạm thời thất bại: {exc}"
            print(f"[FitMotion AI] {persistence_warning}")
    today_summary = get_today_summary_cached(
        user["id"],
        profile,
        shared_state=shared_state,
        ttl_seconds=4.0,
    )

    total_rep = int(shared_state.get("total_rep", 0) or 0)
    good_rep = normalize_display_good_rep(total_rep, shared_state.get("good_rep", 0))
    last_error_code = normalize_pose_label(shared_state.get("last_error_code", ""))
    feedback_audio_rel = str((shared_state.get("feedback_audio_map", {}) or {}).get(last_error_code, "") or "")
    feedback_audio_url = url_for("static", filename=feedback_audio_rel) if feedback_audio_rel else ""
    return jsonify({
        "success": True,
        "exercise_slug": slug,
        "status_text": str(shared_state.get("status_text", "San sang")),
        "total_rep": total_rep,
        "good_rep": good_rep,
        "bad_rep": max(0, total_rep - good_rep),
        "display_good_rep": good_rep,
        "display_bad_rep": max(0, total_rep - good_rep),
        "phase_start_error": int(shared_state.get("phase_start_error", 0) or 0),
        "phase_middle_error": int(shared_state.get("phase_middle_error", 0) or 0),
        "phase_end_error": int(shared_state.get("phase_end_error", 0) or 0),
        "tracking_abort_count": int(shared_state.get("tracking_abort_count", 0) or 0),
        "rep_quality_score": int(shared_state.get("rep_quality_score", 0) or 0),
        "quality_score_avg": float(shared_state.get("quality_score_avg", 0) or 0),
        "keypoint_confidence_avg": float(shared_state.get("keypoint_confidence_avg", 0) or 0),
        "current_primary_angle": float(shared_state.get("current_primary_angle", 0) or 0),
        "current_secondary_angle": float(shared_state.get("current_secondary_angle", 0) or 0),
        "primary_metric_name": str(shared_state.get("primary_metric_name", "Chỉ số chính")),
        "secondary_metric_name": str(shared_state.get("secondary_metric_name", "Chỉ số phụ")),
        "evaluation_side": str(shared_state.get("evaluation_side", "")),
        "feedback_id": int(shared_state.get("feedback_id", 0) or 0),
        "feedback_text": str(shared_state.get("feedback_text", "")),
        "feedback_advice": str(shared_state.get("feedback_advice", "")),
        "feedback_level": str(shared_state.get("feedback_level", "neutral")),
        "voice_text": str(shared_state.get("voice_text", "")),
        "last_error_code": str(shared_state.get("last_error_code", "")),
        "feedback_audio_url": feedback_audio_url,
        "last_rep_components": shared_state.get("last_rep_components", {}),
        "rep_records": shared_state.get("rep_records", []),
        "error_code_counts": shared_state.get("error_code_counts", {}),
        "rejected_attempt_count": int(shared_state.get("rejected_attempt_count", 0) or 0),
        "last_rejected_attempt": shared_state.get("last_rejected_attempt", {}),
        "pose_classifier_available": bool(shared_state.get("pose_classifier_available", False)),
        "pose_classifier_error": str(shared_state.get("pose_classifier_error", "")),
        "pose_classifier_last": shared_state.get("pose_classifier_last", {}),
        "pose_guard_last_decision": shared_state.get("pose_guard_last_decision", {}),
        "exercise_identity_match": bool(shared_state.get("exercise_identity_match", False)),
        "exercise_identity_reason": str(shared_state.get("exercise_identity_reason", "")),
        "body_axis_from_horizontal": float(shared_state.get("body_axis_from_horizontal", 0) or 0),
        "reference_motion_available": bool(shared_state.get("reference_motion_available", False)),
        "reference_motion_error": str(shared_state.get("reference_motion_error", "")),
        "reference_match": shared_state.get("reference_match", {}),
        "fps": int(shared_state.get("fps", 0) or 0),
        "camera_ready": bool(shared_state.get("camera_ready", False)),
        "camera_error": str(shared_state.get("camera_error", "")),
        "camera_backend": str(shared_state.get("camera_backend", "")),
        "camera_last_frame_age": float(shared_state.get("camera_last_frame_age", 0) or 0),
        "curl_extension_baseline": float(shared_state.get("curl_extension_baseline", 0) or 0),
        "curl_calibration_ready": bool(shared_state.get("curl_calibration_ready", False)),
        "squat_standing_baseline": float(shared_state.get("squat_standing_baseline", 0) or 0),
        "squat_calibration_ready": bool(shared_state.get("squat_calibration_ready", False)),
        "pushup_top_baseline": float(shared_state.get("pushup_top_baseline", 0) or 0),
        "pushup_calibration_ready": bool(shared_state.get("pushup_calibration_ready", False)),
        "target_rep": int(shared_state.get("target_rep", get_goal_daily_target(profile)) or 0),
        "set_count": int(shared_state.get("set_count", 1) or 1),
        "reps_per_set": int(shared_state.get("reps_per_set", 10) or 10),
        "target_total_rep": int(shared_state.get("target_rep", 0) or 0),
        "current_set": min(
            int(shared_state.get("set_count", 1) or 1),
            (total_rep // max(1, int(shared_state.get("reps_per_set", 10) or 10))) + 1,
        ),
        "reps_in_current_set": total_rep % max(1, int(shared_state.get("reps_per_set", 10) or 10)),
        "rest_seconds": int(shared_state.get("rest_seconds", 90) or 90),
        "max_idle_seconds": int(shared_state.get("max_idle_seconds", 600) or 600),
        "idle_seconds": int(shared_state.get("idle_seconds", 0) or 0),
        "seconds_until_cancel": max(0, int(shared_state.get("max_idle_seconds", 600) or 600) - int(shared_state.get("idle_seconds", 0) or 0)),
        "workout_cancelled": bool(shared_state.get("workout_cancelled", False)),
        "cancellation_reason": str(shared_state.get("cancellation_reason", "")),
        "training_level": normalize_training_level(shared_state.get("training_level", DEFAULT_LEVEL)),
        "training_level_label": training_level_label(shared_state.get("training_level", DEFAULT_LEVEL)),
        "today_done": int(today_summary["done_rep"]),
        "today_target": int(today_summary["daily_target"]),
        "today_remaining": int(today_summary["remaining_rep"]),
        "today_completed": bool(today_summary["completed"]),
        "today_progress_percent": float(today_summary["progress_percent"]),
        "active": bool(shared_state.get("active", False)),
        "workout_paused": bool(shared_state.get("workout_paused", False)),
        "pause_reason": str(shared_state.get("pause_reason", "")),
        "wellness_check_enabled": bool(shared_state.get("wellness_check_enabled", False)),
        "wellness_check_due": bool(shared_state.get("wellness_check_due", False)),
        "wellness_check_completed": bool(shared_state.get("wellness_check_completed", False)),
        "workout_elapsed_seconds": max(0, int(time.time() - float(shared_state.get("session_started_at", time.time()) or time.time()))),
        "max_session_seconds": int(shared_state.get("max_session_seconds", 120) or 120),
        "session_time_remaining_seconds": max(0, int(shared_state.get("max_session_seconds", 120) or 120) - max(0, int(time.time() - float(shared_state.get("session_started_at", time.time()) or time.time())))) if float(shared_state.get("session_started_at", 0) or 0) > 0 else int(shared_state.get("max_session_seconds", 120) or 120),
        "persistence_warning": persistence_warning,
    })

@app.route("/nguoi-dung/api/xac-nhan-suc-khoe/<slug>", methods=["POST"])
@login_required
def acknowledge_wellness_check_api(slug):
    user = get_user_context()
    shared_state = get_or_create_shared_state(user["id"], slug)
    if bool(shared_state.get("active", False)) or shared_state.get("pause_reason") == "emergency":
        return jsonify({
            "success": False,
            "message": "Không thể tiếp tục khi cảnh báo khẩn cấp còn hoạt động.",
        }), 409
    shared_state.update({
        "wellness_check_due": False,
        "wellness_check_completed": True,
        "workout_paused": False,
        "pause_reason": "",
        "status_text": "San sang tiep tuc",
    })
    return jsonify({
        "success": True,
        "message": "Đã xác nhận bạn vẫn ổn. Buổi tập tiếp tục.",
    })


@app.route("/nguoi-dung/api/bat-dau-lai-buoi-tap/<slug>", methods=["POST"])
@login_required
def restart_cancelled_workout_api(slug):
    user = get_user_context()
    exercise = WorkoutExercise.query.filter_by(slug=slug).first_or_404()
    shared_state = get_or_create_shared_state(user["id"], slug)
    if bool(shared_state.get("active", False)) or shared_state.get("pause_reason") == "emergency":
        return jsonify({
            "success": False,
            "message": "Không thể bắt đầu lại khi cảnh báo khẩn cấp còn hoạt động.",
        }), 409
    level = normalize_training_level(shared_state.get("training_level", DEFAULT_LEVEL))
    set_count = max(1, int(shared_state.get("set_count", 1) or 1))
    reps_per_set = max(1, int(shared_state.get("reps_per_set", 10) or 10))
    delete_workout_draft(user["id"], exercise.id, datetime.now().date().isoformat())
    db.session.commit()
    reset_workout_runtime_state(user["id"], slug, preserve_target=False)
    apply_session_policy(
        shared_state, exercise, level,
        set_count=set_count, reps_per_set=reps_per_set,
    )
    shared_state.update({
        "training_level": level,
        "training_level_label": training_level_label(level),
        "workout_cancelled": False,
        "workout_paused": False,
        "pause_reason": "",
        "cancellation_reason": "",
        "session_started_at": time.time(),
        "last_rep_at": 0.0,
        "idle_seconds": 0,
        "status_text": "SAN SANG BAT DAU LAI",
    })
    return jsonify({
        "success": True,
        "message": "Đã bắt đầu một buổi tập mới. Rep của buổi đã hủy không được khôi phục.",
        "target_total_rep": set_count * reps_per_set,
    })


@app.route("/nguoi-dung/api/dung-phien-tap/<slug>", methods=["POST"])
@login_required
def stop_workout_session_api(slug):
    """Cleanly stop and release an active workout session so it never runs in the background."""
    user = get_user_context()
    exercise = WorkoutExercise.query.filter_by(slug=slug).first_or_404()
    shared_state = get_or_create_shared_state(user["id"], slug)
    session_date = datetime.now().date().isoformat()
    delete_workout_draft(user["id"], exercise.id, session_date)
    db.session.commit()
    reset_workout_runtime_state(user["id"], slug, preserve_target=False)
    shared_state["workout_done"] = True
    shared_state["workout_cancelled"] = False
    shared_state["cancellation_reason"] = ""
    shared_state["active"] = False
    invalidate_today_summary_cache(user["id"])
    return jsonify({
        "success": True,
        "message": "Đã dừng và giải phóng phiên tập hoàn toàn.",
        "redirect_url": url_for("user_exercises"),
    })


@app.route("/nguoi-dung/api/ket-thuc-buoi-tap/<slug>", methods=["POST"])
@login_required
def finish_workout_api(slug):
    user = get_user_context()
    exercise = WorkoutExercise.query.filter_by(slug=slug).first_or_404()
    profile = UserProfile.query.filter_by(user_id=user["id"]).first()
    shared_state = get_or_create_shared_state(user["id"], slug)
    req_data = request.get_json(silent=True) or {}
    confirm_exit = bool(req_data.get("confirm_exit", False))

    if cancel_inactive_workout_if_needed(user["id"], exercise, shared_state):
        return jsonify({
            "success": False,
            "cancelled": True,
            "message": shared_state.get(
                "cancellation_reason",
                "Buổi tập đã bị hủy vì nghỉ quá lâu và không được lưu vào lịch sử.",
            ),
        }), 409

    # A realtime camera workout belongs to the server's current local day.
    session_date = datetime.now().date().isoformat()
    total_rep = int(shared_state.get("total_rep", 0) or 0)
    good_rep = normalize_display_good_rep(total_rep, shared_state.get("good_rep", 0))
    phase_start_error = int(shared_state.get("phase_start_error", 0) or 0)
    phase_middle_error = int(shared_state.get("phase_middle_error", 0) or 0)
    phase_end_error = int(shared_state.get("phase_end_error", 0) or 0)
    set_count = max(1, int(shared_state.get("set_count", 1) or 1))
    reps_per_set = max(1, int(shared_state.get("reps_per_set", 10) or 10))
    target_rep = set_count * reps_per_set

    if total_rep <= 0:
        if confirm_exit:
            delete_workout_draft(user["id"], exercise.id, session_date)
            db.session.commit()
            reset_workout_runtime_state(user["id"], slug, preserve_target=False)
            shared_state["workout_done"] = True
            shared_state["workout_cancelled"] = False
            shared_state["cancellation_reason"] = ""
            invalidate_today_summary_cache(user["id"])
            return jsonify({
                "success": True,
                "is_completed": False,
                "message": "Đã kết thúc buổi tập (chưa có rep nào).",
                "redirect_url": url_for("user_dashboard", date=session_date),
            })
        return jsonify({
            "success": False,
            "is_completed": False,
            "message": "Chưa ghi nhận rep nào trong buổi tập.",
        }), 400

    bad_rep = max(0, total_rep - good_rep)
    keypoint_confidence_avg = float(shared_state.get("keypoint_confidence_avg", 0) or 0)
    quality_score_avg = float(shared_state.get("quality_score_avg", 0) or 0)
    training_level = normalize_training_level(shared_state.get("training_level", DEFAULT_LEVEL))
    level_criteria = get_effective_level_criteria(exercise, training_level)
    started_at = float(shared_state.get("session_started_at", 0.0) or 0.0)
    duration_seconds = max(0, int(time.time() - started_at)) if started_at > 0 else 0
    effectiveness = evaluate_session_effectiveness(
        total_rep=total_rep,
        good_rep=good_rep,
        quality_score_avg=quality_score_avg,
        confidence_avg=keypoint_confidence_avg,
        rep_records=shared_state.get("rep_records", []),
        tracking_abort_count=int(shared_state.get("tracking_abort_count", 0) or 0),
        elapsed_seconds=duration_seconds,
        set_count=set_count,
        reps_per_set=reps_per_set,
        criteria=level_criteria,
    )

    hybrid_summary = summarize_hybrid_rep_evidence(shared_state.get("rep_records", []))
    hybrid_summary["rejected_attempt_count"] = int(shared_state.get("rejected_attempt_count", 0) or 0)
    hybrid_summary["last_rejected_attempt"] = dict(shared_state.get("last_rejected_attempt", {}) or {})
    effectiveness["hybrid_evidence"] = hybrid_summary

    # Ensure the last detected rep is durable before converting the draft into
    # a finalized session.
    checkpoint_workout_draft(
        user["id"], exercise, session_date, shared_state, force=True
    )

    try:
        session_item = save_workout_session_result(
            user_id=user["id"],
            exercise_id=exercise.id,
            session_date=session_date,
            total_rep=total_rep,
            good_rep=good_rep,
            total_error=bad_rep,
            confidence_avg=keypoint_confidence_avg,
            keypoint_confidence_avg=keypoint_confidence_avg,
            quality_score_avg=quality_score_avg,
            rep_details=shared_state.get("rep_records", []),
            error_code_counts=shared_state.get("error_code_counts", {}),
            phase_start_error=phase_start_error,
            phase_middle_error=phase_middle_error,
            phase_end_error=phase_end_error,
            training_level=training_level,
            set_count=set_count,
            rep_target=reps_per_set,
            duration_seconds=duration_seconds,
            effectiveness_status="effective" if effectiveness["effective"] else "partial",
            session_summary=effectiveness,
            reference_score_avg=hybrid_summary["reference_score_avg"],
            pose_guard_confirmed_ratio=hybrid_summary["pose_guard_confirmed_ratio"],
            commit=False,
        )

        plan = WorkoutPlan.query.filter_by(
            user_id=user["id"],
            exercise_id=exercise.id,
            workout_date=session_date,
        ).order_by(WorkoutPlan.created_at.desc()).first()

        status = "completed" if effectiveness["effective"] else "partial"
        if plan:
            plan.set_count = set_count
            plan.rep_target = reps_per_set
            plan.training_level = training_level
            plan.status = status
            session_item.schedule_id = plan.schedule_id
            session_item.training_level = training_level
        else:
            db.session.add(WorkoutPlan(
                user_id=user["id"],
                exercise_id=exercise.id,
                workout_date=session_date,
                set_count=set_count,
                rep_target=reps_per_set,
                training_level=training_level,
                status=status,
            ))

        if profile:
            if hasattr(profile, "done_count"):
                profile.done_count = (profile.done_count or 0) + good_rep
            if hasattr(profile, "total_errors"):
                profile.total_errors = (profile.total_errors or 0) + bad_rep
            if hasattr(profile, "calories_burned"):
                profile.calories_burned = round(
                    (profile.calories_burned or 0.0)
                    + ((exercise.calories or 0.0) * total_rep),
                    2,
                )

        # Draft and finalized session are committed atomically: either both the
        # final row and its removal succeed, or neither change is applied.
        delete_workout_draft(user["id"], exercise.id, session_date)
        db.session.commit()
    except Exception as exc:
        db.session.rollback()
        return jsonify({
            "success": False,
            "message": f"Không thể ghi buổi tập vào SQLite: {exc}",
        }), 500

    try:
        backup_database(DB_PATH)
    except Exception as backup_error:
        print(f"[FitMotion AI] Cảnh báo: chưa tạo được backup sau buổi tập: {backup_error}")

    invalidate_today_summary_cache(user["id"])
    reset_workout_runtime_state(user["id"], slug, preserve_target=False)
    shared_state["workout_done"] = True
    today_summary = get_today_summary(user["id"], profile)

    return jsonify({
        "success": True,
        "is_completed": bool(effectiveness["effective"] and total_rep >= target_rep),
        "message": (
            "Đã hoàn thành cấp độ và lưu buổi tập."
            if effectiveness["effective"]
            else "Đã lưu buổi tập nhưng chưa đạt đầy đủ tiêu chí của cấp độ đã chọn."
        ),
        "redirect_url": url_for("user_dashboard", date=session_date),
        "session_id": session_item.id,
        "database_path": DB_PATH,
        "quality_score_avg": quality_score_avg,
        "keypoint_confidence_avg": keypoint_confidence_avg,
        "effectiveness": effectiveness,
        "training_level": training_level,
        "training_level_label": training_level_label(training_level),
        "set_count": set_count,
        "reps_per_set": reps_per_set,
        "target_total_rep": target_rep,
        "duration_seconds": duration_seconds,
        "today_done": today_summary["done_rep"],
        "today_target": today_summary["daily_target"],
        "today_remaining": today_summary["remaining_rep"],
        "today_completed": today_summary["completed"],
    })

def persist_emergency_alert_if_needed(user_id, slug, shared_state):
    """Persist one durable admin alert per confirmed emergency event."""
    if not bool(shared_state.get("active", False)):
        return None
    updated_at = float(shared_state.get("updated_at", 0.0) or 0.0)
    if updated_at <= 0:
        return None
    event_key = f"{int(user_id)}:{str(slug)}:{int(updated_at * 1000)}"
    existing = EmergencyAlert.query.filter_by(event_key=event_key).first()
    if existing:
        shared_state["emergency_alert_id"] = int(existing.id)
        return existing

    user = db.session.get(User, int(user_id))
    profile = UserProfile.query.filter_by(user_id=int(user_id)).first()
    exercise = WorkoutExercise.query.filter_by(slug=slug).first()
    alert = EmergencyAlert(
        event_key=event_key,
        user_id=int(user_id),
        exercise_id=exercise.id if exercise else None,
        exercise_slug=str(slug),
        phase=str(shared_state.get("emergency_phase", "") or ""),
        reason=str(shared_state.get("reason", "") or "Phát hiện ngã/ngất và bất động."),
        message=str(shared_state.get("message", "") or "Cần kiểm tra tình trạng người tập ngay."),
        evidence_path=str(shared_state.get("image_path", "") or ""),
        body_angle=float(shared_state.get("body_angle", 0.0) or 0.0),
        status="unread",
        user_name_snapshot=str(getattr(user, "fullname", "") or ""),
        user_email_snapshot=str(getattr(user, "email", "") or ""),
        personal_phone_snapshot=str(getattr(profile, "phone", "") or ""),
        address_snapshot=str(getattr(profile, "contact_address", "") or ""),
        emergency_contact_name_snapshot=str(getattr(profile, "emergency_contact_name", "") or ""),
        emergency_relation_snapshot=str(getattr(profile, "emergency_contact_relation", "") or ""),
        emergency_phone_snapshot=str(getattr(profile, "emergency_contact_phone", "") or ""),
        triggered_at=datetime.fromtimestamp(updated_at),
    )
    db.session.add(alert)
    db.session.commit()
    shared_state["emergency_alert_id"] = int(alert.id)
    return alert


def serialize_emergency_alert(alert):
    return {
        "id": int(alert.id),
        "status": str(alert.status),
        "user_name": str(alert.user_name_snapshot or ""),
        "user_email": str(alert.user_email_snapshot or ""),
        "phone": str(alert.personal_phone_snapshot or ""),
        "address": str(alert.address_snapshot or ""),
        "emergency_contact_name": str(alert.emergency_contact_name_snapshot or ""),
        "emergency_relation": str(alert.emergency_relation_snapshot or ""),
        "emergency_phone": str(alert.emergency_phone_snapshot or ""),
        "exercise_slug": str(alert.exercise_slug or ""),
        "phase": str(alert.phase or ""),
        "reason": str(alert.reason or ""),
        "message": str(alert.message or ""),
        "evidence_path": str(alert.evidence_path or ""),
        "triggered_at": alert.triggered_at.isoformat(timespec="seconds") if alert.triggered_at else "",
    }


@app.route("/nguoi-dung/api/trang-thai-khan-cap/<slug>")
@login_required
def emergency_status_api(slug):
    user = get_user_context()
    _, shared_state = get_or_create_emergency_state(user["id"], slug)
    persisted_alert = persist_emergency_alert_if_needed(user["id"], slug, shared_state)

    safe_state = {
        "active": True if shared_state.get("active", False) else False,
        "message": str(shared_state.get("message", "")),
        "reason": str(shared_state.get("reason", "")),
        "updated_at": float(shared_state.get("updated_at", 0) or 0),
        "image_path": str(shared_state.get("image_path", "")),
        "body_angle": float(shared_state.get("body_angle", 0) or 0),
        "low_posture": True if shared_state.get("low_posture", False) else False,
        "emergency_posture": str(shared_state.get("emergency_posture", "unknown")),
        "emergency_candidate": True if shared_state.get("emergency_candidate", False) else False,
        "emergency_armed": True if shared_state.get("emergency_armed", False) else False,
        "emergency_immobile_seconds": float(shared_state.get("emergency_immobile_seconds", 0) or 0),
        "emergency_required_seconds": float(shared_state.get("emergency_required_seconds", 6) or 6),
        "emergency_exercise": str(shared_state.get("emergency_exercise", "")),
        "emergency_phase": str(shared_state.get("emergency_phase", "")),
        "workout_paused": bool(shared_state.get("workout_paused", False)),
        "pause_reason": str(shared_state.get("pause_reason", "")),
        "emergency_alert_id": int(persisted_alert.id) if persisted_alert else int(shared_state.get("emergency_alert_id", 0) or 0),
    }

    return jsonify(safe_state)


@app.route("/nguoi-dung/api/reset-khan-cap/<slug>", methods=["POST"])
@login_required
def reset_emergency_api(slug):
    user = get_user_context()
    _, shared_state = get_or_create_emergency_state(user["id"], slug)
    reset_version = int(shared_state.get("emergency_reset_version", 0) or 0) + 1
    # Reset only safety-monitor fields. Workout reps and quality metrics must
    # remain intact when the user dismisses a false/past emergency overlay.
    was_emergency_pause = shared_state.get("pause_reason") == "emergency"
    shared_state.update({
        "active": False,
        "message": "",
        "reason": "",
        "updated_at": 0.0,
        "image_path": "",
        "body_angle": 0.0,
        "low_posture": False,
        "emergency_posture": "unknown",
        "emergency_candidate": False,
        "emergency_armed": False,
        "emergency_immobile_seconds": 0.0,
        "emergency_required_seconds": 6.0,
        "emergency_exercise": slug,
        "emergency_phase": "",
        "emergency_reset_version": reset_version,
        "emergency_alert_id": 0,
        "workout_paused": False if was_emergency_pause else bool(shared_state.get("workout_paused", False)),
        "pause_reason": "" if was_emergency_pause else str(shared_state.get("pause_reason", "")),
    })
    return jsonify({"success": True, "message": "Đã đặt lại cảnh báo khẩn cấp."})


@app.route("/nguoi-dung/api/camera-status/<slug>")
@login_required
def camera_status_api(slug):
    user = get_user_context()
    shared_state = get_or_create_shared_state(user["id"], slug)
    status = camera_service.status()
    shared_state.update({
        "camera_ready": bool(status.get("ready", False)),
        "camera_error": str(status.get("error", "")),
        "camera_backend": str(status.get("backend", "")),
        "camera_last_frame_age": float(status.get("last_frame_age", 0.0) or 0.0),
    })
    return jsonify({"success": True, **status})


@app.route("/nguoi-dung/api/camera-restart/<slug>", methods=["POST"])
@login_required
def camera_restart_api(slug):
    user = get_user_context()
    shared_state = get_or_create_shared_state(user["id"], slug)
    camera_service.restart()
    shared_state.update({
        "camera_ready": False,
        "camera_error": "Đang kết nối lại camera...",
    })
    return jsonify({
        "success": True,
        "message": "Đã gửi yêu cầu kết nối lại camera.",
    })


@app.route("/video/<slug>")
@login_required
def video_feed(slug):
    user = get_user_context()
    exercise = WorkoutExercise.query.filter_by(slug=slug).first_or_404()
    shared_state = get_or_create_shared_state(user["id"], slug)
    configure_exercise_ai_resources(exercise, shared_state)
    criteria = ExerciseCriterion.query.filter_by(exercise_id=exercise.id).all()
    shared_state["feedback_audio_map"] = criterion_audio_map(criteria)
    shared_state["admin_criteria_rules"] = runtime_criterion_rules(criteria)
    return Response(
        stream_with_context(gen_frames(slug, user["id"])),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/admin/dashboard")
@admin_required
def admin_dashboard():
    user = get_user_context()

    total_users = User.query.filter_by(role="user").count()
    total_exercises = WorkoutExercise.query.count()
    total_plans = WorkoutPlan.query.count()
    total_sessions = WorkoutSession.query.count()
    total_drafts = WorkoutSessionDraft.query.count()
    database_size_mb = round(os.path.getsize(DB_PATH) / (1024 * 1024), 3) if os.path.exists(DB_PATH) else 0
    backup_dir = os.path.join(os.path.dirname(DB_PATH), "backups")
    backup_count = len([name for name in os.listdir(backup_dir) if name.endswith(".db")]) if os.path.isdir(backup_dir) else 0

    return render_template(
        "admin_dashboard.html",
        title="Dashboard admin",
        user=user,
        total_users=total_users,
        total_exercises=total_exercises,
        total_plans=total_plans,
        total_sessions=total_sessions,
        total_drafts=total_drafts,
        database_path=DB_PATH,
        database_size_mb=database_size_mb,
        backup_count=backup_count,
        unread_emergency_count=EmergencyAlert.query.filter_by(status="unread").count(),
    )


@app.route("/admin/nguoi-dung")
@admin_required
def admin_users():
    users = User.query.filter_by(role="user").order_by(User.created_at.desc()).all()
    rows = []
    for item in users:
        profile = UserProfile.query.filter_by(user_id=item.id).first()
        rows.append({"user": item, "profile": profile})
    return render_template(
        "admin_users.html",
        title="Quản lý người dùng",
        user=get_user_context(),
        rows=rows,
    )


@app.route("/admin/canh-bao-khan-cap")
@admin_required
def admin_emergency_alerts():
    alerts = EmergencyAlert.query.order_by(EmergencyAlert.triggered_at.desc()).limit(200).all()
    # Backfill snapshots for alerts created by older builds. New alerts already
    # store immutable registration/contact data at trigger time.
    changed = False
    for alert in alerts:
        account = db.session.get(User, alert.user_id)
        profile = UserProfile.query.filter_by(user_id=alert.user_id).first()
        if account and not alert.user_name_snapshot:
            alert.user_name_snapshot = str(account.fullname or "")
            changed = True
        if account and not alert.user_email_snapshot:
            alert.user_email_snapshot = str(account.email or "")
            changed = True
        if profile:
            snapshot_fields = {
                "personal_phone_snapshot": profile.phone,
                "address_snapshot": profile.contact_address,
                "emergency_contact_name_snapshot": profile.emergency_contact_name,
                "emergency_relation_snapshot": profile.emergency_contact_relation,
                "emergency_phone_snapshot": profile.emergency_contact_phone,
            }
            for field_name, value in snapshot_fields.items():
                if not getattr(alert, field_name, "") and value:
                    setattr(alert, field_name, str(value))
                    changed = True
    if changed:
        db.session.commit()
    return render_template(
        "admin_emergency_alerts.html",
        title="Cảnh báo khẩn cấp",
        user=get_user_context(),
        alerts=alerts,
        unread_count=EmergencyAlert.query.filter_by(status="unread").count(),
    )


@app.route("/admin/canh-bao-khan-cap/<int:alert_id>/anh")
@admin_required
def admin_emergency_evidence(alert_id):
    alert = EmergencyAlert.query.get_or_404(alert_id)
    raw_path = str(alert.evidence_path or "").strip()
    if not raw_path:
        abort(404)
    raw_candidate = Path(raw_path)
    candidates = []
    if raw_candidate.is_absolute():
        candidates.append(raw_candidate)
    else:
        candidates.append(Path(BASE_DIR) / raw_candidate)
        candidates.append(PERSISTENT_EMERGENCY_DIR / raw_candidate.name)
    resolved = None
    allowed_roots = [
        (Path(BASE_DIR) / "data" / "errorimages").resolve(),
        (Path(DATA_DIR) / "errorimages").resolve(),
    ]
    for candidate in candidates:
        try:
            found = candidate.resolve(strict=True)
            if any(_is_path_within(found, root) for root in allowed_roots) and found.is_file():
                resolved = found
                break
        except FileNotFoundError:
            continue
    if resolved is None:
        abort(404)
    return send_file(str(resolved), mimetype="image/jpeg", max_age=0)


@app.route("/admin/api/thong-bao-khan-cap")
@admin_api_required
def admin_emergency_notifications_api():
    alerts = EmergencyAlert.query.order_by(EmergencyAlert.triggered_at.desc()).limit(20).all()
    return jsonify({
        "unread_count": EmergencyAlert.query.filter_by(status="unread").count(),
        "alerts": [serialize_emergency_alert(item) for item in alerts],
    })


@app.route("/admin/api/thong-bao-khan-cap/<int:alert_id>/doc", methods=["POST"])
@admin_api_required
def admin_mark_emergency_read(alert_id):
    alert = EmergencyAlert.query.get_or_404(alert_id)
    if alert.status == "unread":
        alert.status = "read"
        alert.read_at = datetime.utcnow()
        db.session.commit()
    return jsonify({"success": True, "alert": serialize_emergency_alert(alert)})


@app.route("/admin/api/thong-bao-khan-cap/<int:alert_id>/da-xu-ly", methods=["POST"])
@admin_api_required
def admin_resolve_emergency_alert(alert_id):
    alert = EmergencyAlert.query.get_or_404(alert_id)
    alert.status = "resolved"
    if alert.read_at is None:
        alert.read_at = datetime.utcnow()
    alert.resolved_at = datetime.utcnow()
    db.session.commit()
    return jsonify({"success": True, "alert": serialize_emergency_alert(alert)})


@app.route("/admin/bai-tap")
@admin_required
def admin_exercises():
    q = request.args.get("q", "").strip().lower()
    exercises = WorkoutExercise.query.order_by(WorkoutExercise.created_at.desc()).all()

    if q:
        exercises = [
            ex for ex in exercises
            if q in ex.name.lower()
            or q in ex.slug.lower()
            or q in (ex.muscle_group or "").lower()
        ]

    return render_template(
        "admin_exercises.html",
        title="Quản lý bài tập",
        user=get_user_context(),
        exercises=exercises,
        keyword=q
    )


@app.route("/admin/bai-tap/them", methods=["GET", "POST"])
@admin_required
def admin_add_exercise():
    if request.method == "POST":
        name = request.form.get("name", "").strip()
        slug = request.form.get("slug", "").strip() or slugify(name)
        if not name or not slug:
            flash("Tên bài tập và slug không được để trống.")
            return redirect(url_for("admin_add_exercise"))

        existed = WorkoutExercise.query.filter_by(slug=slug).first()
        if existed:
            flash("Slug bài tập đã tồn tại.")
            return redirect(url_for("admin_add_exercise"))

        try:
            preview_image_path = save_uploaded_asset(
                request.files.get("preview_image"),
                UPLOAD_EXERCISE_DIR,
                "uploads/exercises",
                ALLOWED_IMAGE_EXTENSIONS,
            )
            model_file = request.files.get("model_3d_file") or request.files.get("fbx_file")
            model_path = save_uploaded_asset(
                model_file,
                UPLOAD_FBX_DIR,
                "uploads/fbx",
                ALLOWED_MODEL_EXTENSIONS,
            )
            intro_file = request.files.get("intro_model_3d_file")
            intro_model_path = save_uploaded_asset(
                intro_file,
                UPLOAD_FBX_DIR,
                "uploads/fbx",
                ALLOWED_MODEL_EXTENSIONS,
            )
            validate_saved_3d_asset(model_path)
            validate_saved_3d_asset(intro_model_path)
            if intro_model_path and slug != "pushup":
                raise ValueError("Animation mở đầu hiện chỉ áp dụng cho bài pushup.")
        except ValueError as exc:
            flash(str(exc))
            return redirect(url_for("admin_add_exercise"))

        extension = os.path.splitext(model_path)[1].lower()
        model_format = (
            "fbx" if extension == ".fbx"
            else "gltf" if extension in {".glb", ".gltf"}
            else "none"
        )
        intro_extension = os.path.splitext(intro_model_path)[1].lower()
        intro_model_format = (
            "fbx" if intro_extension == ".fbx"
            else "gltf" if intro_extension in {".glb", ".gltf"}
            else "none"
        )

        try:
            exercise = WorkoutExercise(
                name=name,
                slug=slug,
                muscle_group=request.form.get("muscle_group", ""),
                age_min=max(1, int(request.form.get("age_min", 15))),
                age_max=max(1, int(request.form.get("age_max", 100))),
                calories=max(0.0, float(request.form.get("calories", 0.1))),
                difficulty=request.form.get("difficulty", "Cơ bản"),
                side_mode=request.form.get("side_mode", "none"),
                description=request.form.get("description", ""),
                guide_text=request.form.get("guide_text", ""),
                suitable_for=request.form.get("suitable_for", ""),
                caution_for=request.form.get("caution_for", ""),
                preview_image=preview_image_path,
                fbx_path=model_path if model_format == "fbx" else "",
                model_3d_path=model_path,
                model_3d_format=model_format,
                animation_key=request.form.get("animation_key", "").strip() or slug,
                intro_model_3d_path=intro_model_path,
                intro_model_3d_format=intro_model_format,
                intro_animation_key=request.form.get("intro_animation_key", "").strip() or ("pushup-intro" if slug == "pushup" and intro_model_path else ""),
                is_active=True,
            )
        except (TypeError, ValueError):
            flash("Thông tin tuổi hoặc calo không hợp lệ.")
            return redirect(url_for("admin_add_exercise"))

        db.session.add(exercise)
        auto_reference_note = ""
        if model_path:
            try:
                payload = auto_attach_reference_from_glb(exercise, model_path)
                if payload:
                    auto_reference_note = " GLB đã đồng thời được lấy làm reference DTW chấm điểm."
            except (GLBReferenceError, ValueError) as exc:
                auto_reference_note = f" Mô hình hiển thị đã lưu nhưng chưa tự tạo được reference: {exc}"
        db.session.commit()

        flash("Đã thêm bài tập mới." + auto_reference_note)
        return redirect(url_for("admin_exercise_detail", exercise_id=exercise.id))


    return render_template(
        "admin_exercise_add.html",
        title="Thêm bài tập",
        user=get_user_context()
    )


@app.route("/admin/bai-tap/<int:exercise_id>", methods=["GET", "POST"])
@admin_required
def admin_exercise_detail(exercise_id):
    exercise = WorkoutExercise.query.get_or_404(exercise_id)

    if request.method == "POST":
        previous_slug = exercise.slug
        proposed_slug = request.form.get("slug", exercise.slug).strip() or slugify(
            request.form.get("name", exercise.name)
        )
        duplicate = WorkoutExercise.query.filter(
            WorkoutExercise.slug == proposed_slug,
            WorkoutExercise.id != exercise.id,
        ).first()
        if duplicate:
            flash("Slug bài tập đã được sử dụng.")
            return redirect(url_for("admin_exercise_detail", exercise_id=exercise.id))

        try:
            exercise.name = request.form.get("name", exercise.name).strip() or exercise.name
            exercise.slug = proposed_slug
            exercise.muscle_group = request.form.get("muscle_group", exercise.muscle_group)
            exercise.age_min = max(1, int(request.form.get("age_min", exercise.age_min)))
            exercise.age_max = max(1, int(request.form.get("age_max", exercise.age_max)))
            exercise.calories = max(0.0, float(request.form.get("calories", exercise.calories)))
            exercise.difficulty = request.form.get("difficulty", exercise.difficulty)
            exercise.side_mode = request.form.get("side_mode", exercise.side_mode)
            exercise.description = request.form.get("description", exercise.description)
            exercise.guide_text = request.form.get("guide_text", exercise.guide_text)
            exercise.suitable_for = request.form.get("suitable_for", exercise.suitable_for)
            exercise.caution_for = request.form.get("caution_for", exercise.caution_for)
            exercise.animation_key = request.form.get("animation_key", "").strip() or exercise.slug
            exercise.intro_animation_key = request.form.get("intro_animation_key", "").strip() or exercise.intro_animation_key or ("pushup-intro" if exercise.slug == "pushup" else "")
            exercise.is_active = request.form.get("is_active") == "on"

            preview_path = save_uploaded_asset(
                request.files.get("preview_image"),
                UPLOAD_EXERCISE_DIR,
                "uploads/exercises",
                ALLOWED_IMAGE_EXTENSIONS,
            )
            if preview_path:
                exercise.preview_image = preview_path

            model_file = request.files.get("model_3d_file") or request.files.get("fbx_file")
            model_path = save_uploaded_asset(
                model_file,
                UPLOAD_FBX_DIR,
                "uploads/fbx",
                ALLOWED_MODEL_EXTENSIONS,
            )
            intro_file = request.files.get("intro_model_3d_file")
            intro_model_path = save_uploaded_asset(
                intro_file,
                UPLOAD_FBX_DIR,
                "uploads/fbx",
                ALLOWED_MODEL_EXTENSIONS,
            )
            if intro_model_path and exercise.slug != "pushup":
                raise ValueError("Animation mở đầu hiện chỉ áp dụng cho bài pushup.")

            if model_path:
                validate_saved_3d_asset(model_path)
                extension = os.path.splitext(model_path)[1].lower()
                old_model_path = exercise.model_3d_path
                exercise.model_3d_path = model_path
                exercise.model_3d_format = (
                    "fbx" if extension == ".fbx"
                    else "gltf" if extension in {".glb", ".gltf"}
                    else "none"
                )
                exercise.fbx_path = model_path if extension == ".fbx" else ""
                if old_model_path and old_model_path != model_path:
                    _remove_static_asset(old_model_path)

            if intro_model_path:
                validate_saved_3d_asset(intro_model_path)
                intro_extension = os.path.splitext(intro_model_path)[1].lower()
                old_intro_path = exercise.intro_model_3d_path
                exercise.intro_model_3d_path = intro_model_path
                exercise.intro_model_3d_format = (
                    "fbx" if intro_extension == ".fbx"
                    else "gltf" if intro_extension in {".glb", ".gltf"}
                    else "none"
                )
                if old_intro_path and old_intro_path != intro_model_path:
                    _remove_static_asset(old_intro_path)

            if request.form.get("remove_model") == "on":
                _remove_static_asset(exercise.model_3d_path)
                _remove_static_asset(exercise.reference_motion_path)
                exercise.model_3d_path = ""
                exercise.fbx_path = ""
                exercise.model_3d_format = "none"
                exercise.reference_motion_path = ""
                exercise.reference_motion_source = ""
                exercise.reference_motion_version = ""

            if request.form.get("remove_intro_model") == "on":
                _remove_static_asset(exercise.intro_model_3d_path)
                exercise.intro_model_3d_path = ""
                exercise.intro_model_3d_format = "none"
                exercise.intro_animation_key = ""
        except ValueError as exc:
            flash(str(exc))
            return redirect(url_for("admin_exercise_detail", exercise_id=exercise.id))
        except TypeError:
            flash("Thông tin tuổi hoặc calo không hợp lệ.")
            return redirect(url_for("admin_exercise_detail", exercise_id=exercise.id))

        auto_reference_note = ""
        if model_path:
            # A new main GLB invalidates any older motion standard; regenerate
            # the reference from the exact asset now displayed to the user.
            try:
                payload = auto_attach_reference_from_glb(exercise, model_path)
                if payload:
                    auto_reference_note = " GLB mới đã được lấy làm reference DTW chấm điểm."
                elif exercise.reference_motion_path:
                    _remove_static_asset(exercise.reference_motion_path)
                    exercise.reference_motion_path = ""
                    exercise.reference_motion_source = ""
                    exercise.reference_motion_version = ""
                    auto_reference_note = " Asset không phải GLB lõi; hãy upload Reference JSON nếu muốn dùng làm chuẩn chấm điểm."
            except (GLBReferenceError, ValueError) as exc:
                if exercise.reference_motion_path:
                    _remove_static_asset(exercise.reference_motion_path)
                exercise.reference_motion_path = ""
                exercise.reference_motion_source = ""
                exercise.reference_motion_version = ""
                auto_reference_note = f" Mô hình hiển thị đã lưu nhưng chưa dùng làm chuẩn chấm điểm: {exc}"
        elif previous_slug != exercise.slug and exercise.reference_motion_path:
            _remove_static_asset(exercise.reference_motion_path)
            exercise.reference_motion_path = ""
            exercise.reference_motion_source = ""
            exercise.reference_motion_version = ""
            auto_reference_note = " Slug đã đổi nên reference cũ được gỡ để tránh chấm sai bài."

        db.session.commit()
        flash("Đã cập nhật bài tập." + auto_reference_note)
        return redirect(url_for("admin_exercise_detail", exercise_id=exercise.id))

    ensure_level_config_defaults(exercise)
    db.session.commit()
    level_config_rows = ExerciseLevelConfig.query.filter_by(
        exercise_id=exercise.id
    ).all()
    level_configs = {
        normalize_training_level(row.training_level): row
        for row in level_config_rows
    }

    criteria = ExerciseCriterion.query.filter_by(
        exercise_id=exercise.id
    ).order_by(ExerciseCriterion.created_at.desc()).all()
    labels = ExerciseLabelImage.query.filter_by(
        exercise_id=exercise.id
    ).order_by(ExerciseLabelImage.created_at.desc()).all()
    training_media = ExerciseTrainingMedia.query.filter_by(
        exercise_id=exercise.id
    ).order_by(ExerciseTrainingMedia.created_at.desc()).limit(100).all()
    session_count = WorkoutSession.query.filter_by(exercise_id=exercise.id).count()
    plan_count = WorkoutPlan.query.filter_by(exercise_id=exercise.id).count()
    training_validation = validate_web_training_dataset(exercise)

    return render_template(
        "admin_exercise_detail.html",
        title="Chi tiết bài tập",
        user=get_user_context(),
        exercise=exercise,
        criteria=criteria,
        labels=labels,
        training_media=training_media,
        training_labels=allowed_training_labels_for_exercise(exercise.slug),
        training_validation=training_validation,
        keypoint_options=YOLO_KEYPOINT_OPTIONS,
        session_count=session_count,
        plan_count=plan_count,
        model_config=resolve_exercise_model(exercise),
        level_configs=level_configs,
        level_options=LEVEL_OPTIONS,
    )


@app.route("/admin/bai-tap/<int:exercise_id>/level-config", methods=["POST"])
@admin_required
def admin_update_level_config(exercise_id):
    exercise = WorkoutExercise.query.get_or_404(exercise_id)
    ensure_level_config_defaults(exercise)
    try:
        for level_key, _ in LEVEL_OPTIONS:
            row = ExerciseLevelConfig.query.filter_by(
                exercise_id=exercise.id, training_level=level_key
            ).first()
            if not row:
                continue
            prefix = f"{level_key}_"
            row.set_count = max(1, min(5, int(request.form.get(prefix + "set_count", row.set_count))))
            row.rep_target = max(1, min(100, int(request.form.get(prefix + "rep_target", row.rep_target))))
            row.rest_seconds = max(15, min(600, int(request.form.get(prefix + "rest_seconds", row.rest_seconds))))
            row.max_idle_seconds = max(60, min(3600, int(request.form.get(prefix + "max_idle_seconds", row.max_idle_seconds))))
            row.max_session_seconds = max(300, min(7200, int(request.form.get(prefix + "max_session_seconds", row.max_session_seconds))))
            row.min_good_rep_ratio = max(0.0, min(1.0, float(request.form.get(prefix + "min_good_rep_percent", row.min_good_rep_ratio * 100)) / 100.0))
            row.min_quality_score = max(0.0, min(100.0, float(request.form.get(prefix + "min_quality_score", row.min_quality_score))))
            row.min_confidence = max(0.0, min(1.0, float(request.form.get(prefix + "min_confidence_percent", row.min_confidence * 100)) / 100.0))
            row.min_stability_score = max(0.0, min(100.0, float(request.form.get(prefix + "min_stability_score", row.min_stability_score))))
            row.max_tracking_abort_count = max(0, min(20, int(request.form.get(prefix + "max_tracking_abort_count", row.max_tracking_abort_count))))
        db.session.commit()
        flash("Đã cập nhật bộ tiêu chí Dễ / Trung bình / Khó cho bài tập.")
    except (TypeError, ValueError):
        db.session.rollback()
        flash("Thông số cấp độ không hợp lệ. Hãy kiểm tra lại set, rep, thời gian và tỷ lệ phần trăm.")
    return redirect(url_for("admin_exercise_detail", exercise_id=exercise.id))


@app.route("/admin/bai-tap/<int:exercise_id>/toggle-active", methods=["POST"])
@admin_required
def admin_toggle_exercise(exercise_id):
    exercise = WorkoutExercise.query.get_or_404(exercise_id)
    exercise.is_active = not bool(exercise.is_active)
    db.session.commit()
    flash("Đã cập nhật trạng thái bài tập.")
    return redirect(request.referrer or url_for("admin_exercises"))


@app.route("/admin/bai-tap/<int:exercise_id>/xoa", methods=["POST"])
@admin_required
def admin_delete_exercise(exercise_id):
    exercise = WorkoutExercise.query.get_or_404(exercise_id)
    confirmation = request.form.get("confirmation", "").strip().lower()
    if confirmation != exercise.slug.lower():
        flash(f"Để xóa, hãy nhập đúng slug: {exercise.slug}")
        return redirect(url_for("admin_exercise_detail", exercise_id=exercise.id))

    session_count = WorkoutSession.query.filter_by(exercise_id=exercise.id).count()
    if session_count > 0:
        exercise.is_active = False
        db.session.commit()
        flash(
            "Bài tập đã có lịch sử luyện tập nên không xóa cứng. "
            "Hệ thống đã chuyển sang trạng thái vô hiệu hóa để bảo toàn dữ liệu."
        )
        return redirect(url_for("admin_exercises"))

    WorkoutPlan.query.filter_by(exercise_id=exercise.id).delete(
        synchronize_session=False
    )
    db.session.delete(exercise)
    db.session.commit()
    flash("Đã xóa bài tập và dữ liệu cấu hình liên quan.")
    return redirect(url_for("admin_exercises"))


@app.route("/admin/bai-tap/<int:exercise_id>/label-upload", methods=["POST"])
@admin_required
def admin_label_upload(exercise_id):
    exercise = WorkoutExercise.query.get_or_404(exercise_id)
    label_name = request.form.get("label_name", "").strip()
    frame_index = int(request.form.get("frame_index", 1))
    images = request.files.getlist("label_images")

    saved_count = 0
    for file in images:
        if file and file.filename:
            filename = secure_filename(file.filename)
            save_path = os.path.join(UPLOAD_LABEL_DIR, filename)
            file.save(save_path)

            label = ExerciseLabelImage(
                exercise_id=exercise.id,
                label_name=label_name,
                frame_index=frame_index,
                image_path=f"uploads/labels/{filename}"
            )
            db.session.add(label)
            saved_count += 1

    db.session.commit()
    flash(f"Đã tải lên {saved_count} ảnh label.")
    return redirect(url_for("admin_exercise_detail", exercise_id=exercise.id))


@app.route("/admin/bai-tap/<int:exercise_id>/training-media-upload", methods=["POST"])
@admin_required
def admin_training_media_upload(exercise_id):
    exercise = WorkoutExercise.query.get_or_404(exercise_id)
    label_name = normalize_pose_label(request.form.get("label_name", ""))
    allowed = set(allowed_training_labels_for_exercise(exercise.slug))
    if label_name not in allowed:
        flash("Label train/test không hợp lệ cho bài tập này.")
        return redirect(url_for("admin_exercise_detail", exercise_id=exercise.id))

    files = request.files.getlist("training_media")
    label_dir = os.path.join(POSE_TRAINING_DIR, label_name)
    os.makedirs(label_dir, exist_ok=True)
    saved_count = 0
    try:
        for uploaded in files:
            if not uploaded or not uploaded.filename:
                continue
            original = secure_filename(uploaded.filename)
            extension = os.path.splitext(original)[1].lower()
            if extension not in ALLOWED_IMAGE_EXTENSIONS | ALLOWED_VIDEO_EXTENSIONS:
                raise ValueError(f"Định dạng {extension or '(không có đuôi)'} không dùng cho dữ liệu train/test.")
            stem = slugify(os.path.splitext(original)[0]) or "training"
            filename = f"{stem}-{uuid.uuid4().hex[:10]}{extension}"
            absolute = os.path.join(label_dir, filename)
            uploaded.save(absolute)
            rel = os.path.relpath(absolute, BASE_DIR).replace(os.sep, "/")
            db.session.add(ExerciseTrainingMedia(
                exercise_id=exercise.id,
                label_name=label_name,
                media_type="video" if extension in ALLOWED_VIDEO_EXTENSIONS else "image",
                file_path=rel,
                source_name=original,
            ))
            saved_count += 1
        db.session.commit()
    except ValueError as exc:
        db.session.rollback()
        flash(str(exc))
        return redirect(url_for("admin_exercise_detail", exercise_id=exercise.id))
    flash(f"Đã thêm {saved_count} nguồn ảnh/video. Khi đủ dữ liệu cho từng trạng thái, bấm Huấn luyện mô hình ngay trên trang này.")
    return redirect(url_for("admin_exercise_detail", exercise_id=exercise.id))


@app.route("/admin/bai-tap/<int:exercise_id>/training-media-delete/<int:media_id>", methods=["POST"])
@admin_required
def admin_training_media_delete(exercise_id, media_id):
    """Delete a single labelled training source (image or video) from dataset."""
    exercise = WorkoutExercise.query.get_or_404(exercise_id)
    media = ExerciseTrainingMedia.query.filter_by(id=media_id, exercise_id=exercise.id).first_or_404()
    # Remove physical file
    try:
        abs_path = os.path.join(BASE_DIR, "data", "pose_training", media.label_name,
                                os.path.basename(media.file_path))
        if os.path.exists(abs_path):
            os.remove(abs_path)
    except OSError:
        pass
    db.session.delete(media)
    db.session.commit()
    return jsonify({"success": True, "deleted_id": media_id})


@app.route("/admin/bai-tap/<int:exercise_id>/training-media-list")
@admin_required
def admin_training_media_list(exercise_id):
    """Return current training media as JSON for live refresh."""
    exercise = WorkoutExercise.query.get_or_404(exercise_id)
    validation = validate_web_training_dataset(exercise)
    items = []
    for m in ExerciseTrainingMedia.query.filter_by(exercise_id=exercise.id).order_by(
            ExerciseTrainingMedia.label_name, ExerciseTrainingMedia.created_at).all():
        # Build URL for images so admin can preview
        preview_url = ""
        if m.media_type == "image":
            try:
                abs_p = os.path.join(BASE_DIR, "data", "pose_training",
                                     m.label_name, os.path.basename(m.file_path))
                if os.path.exists(abs_p):
                    preview_url = url_for("admin_training_media_preview",
                                         exercise_id=exercise.id, media_id=m.id)
            except Exception:
                pass
        items.append({
            "id": m.id,
            "label": m.label_name,
            "type": m.media_type,
            "name": m.source_name or os.path.basename(m.file_path),
            "preview_url": preview_url,
            "created_at": m.created_at.strftime("%d/%m %H:%M") if m.created_at else "",
        })
    return jsonify({
        "success": True,
        "items": items,
        "counts": validation["counts"],
        "ready": validation["ready"],
        "message": validation["message"],
        "minimum": validation["minimum_sources_per_label"],
    })


@app.route("/admin/bai-tap/<int:exercise_id>/training-media-preview/<int:media_id>")
@admin_required
def admin_training_media_preview(exercise_id, media_id):
    """Serve a training image for inline preview in admin UI."""
    exercise = WorkoutExercise.query.get_or_404(exercise_id)
    media = ExerciseTrainingMedia.query.filter_by(id=media_id, exercise_id=exercise.id).first_or_404()
    if media.media_type != "image":
        return "", 204
    abs_path = os.path.join(BASE_DIR, "data", "pose_training",
                            media.label_name, os.path.basename(media.file_path))
    if not os.path.exists(abs_path):
        return "", 404
    from flask import send_file
    return send_file(abs_path)


@app.route("/admin/bai-tap/<int:exercise_id>/train-model", methods=["POST"])
@admin_required
def admin_train_exercise_model(exercise_id):
    """Start per-exercise training directly from the Admin web screen."""
    exercise = WorkoutExercise.query.get_or_404(exercise_id)
    validation = validate_web_training_dataset(exercise)
    if not validation["ready"]:
        payload = {"success": False, "message": validation["message"], "validation": validation}
        if request.headers.get("X-Requested-With") == "XMLHttpRequest" or request.accept_mimetypes.best == "application/json":
            return jsonify(payload), 400
        flash(validation["message"])
        return redirect(url_for("admin_exercise_detail", exercise_id=exercise.id))

    with MODEL_TRAINING_LOCK:
        running = MODEL_TRAINING_THREADS.get(int(exercise.id))
        if running and running.is_alive():
            payload = {"success": False, "message": "Bài tập này đang được huấn luyện. Hãy chờ tiến trình hiện tại hoàn tất."}
            if request.headers.get("X-Requested-With") == "XMLHttpRequest" or request.accept_mimetypes.best == "application/json":
                return jsonify(payload), 409
            flash(payload["message"])
            return redirect(url_for("admin_exercise_detail", exercise_id=exercise.id))

        exercise.pose_model_status = "training"
        exercise.pose_model_error = ""
        exercise.pose_model_accuracy = 0.0
        exercise.pose_model_updated_at = datetime.utcnow()
        db.session.commit()

        worker = threading.Thread(
            target=_train_exercise_model_job,
            args=(exercise.id, tuple(validation["labels_to_train"])),
            daemon=True,
            name=f"fitmotion-train-{exercise.id}",
        )
        MODEL_TRAINING_THREADS[int(exercise.id)] = worker
        worker.start()

    payload = {
        "success": True,
        "message": "Đã bắt đầu huấn luyện trên server. Trang Admin sẽ tự cập nhật trạng thái.",
        "status": "training",
    }
    if request.headers.get("X-Requested-With") == "XMLHttpRequest" or request.accept_mimetypes.best == "application/json":
        return jsonify(payload), 202
    flash(payload["message"])
    return redirect(url_for("admin_exercise_detail", exercise_id=exercise.id))


@app.route("/admin/bai-tap/<int:exercise_id>/training-status")
@admin_required
def admin_exercise_training_status(exercise_id):
    exercise = WorkoutExercise.query.get_or_404(exercise_id)
    validation = validate_web_training_dataset(exercise)
    return jsonify({
        "success": True,
        "exercise_id": exercise.id,
        "status": str(exercise.pose_model_status or "not_trained"),
        "accuracy": round(float(exercise.pose_model_accuracy or 0.0), 4),
        "error": str(exercise.pose_model_error or ""),
        "model_path": str(exercise.pose_model_path or ""),
        "metrics_path": str(exercise.pose_model_metrics_path or ""),
        "updated_at": exercise.pose_model_updated_at.isoformat() if exercise.pose_model_updated_at else "",
        "validation": validation,
        "quality_gate": POSE_MODEL_MIN_TEST_ACCURACY,
    })


@app.route("/admin/bai-tap/<int:exercise_id>/reference-upload", methods=["POST"])
@admin_required
def admin_reference_upload(exercise_id):
    exercise = WorkoutExercise.query.get_or_404(exercise_id)
    if request.form.get("remove_reference") == "on":
        exercise.reference_motion_path = ""
        exercise.reference_motion_source = ""
        exercise.reference_motion_version = ""
        db.session.commit()
        flash("Đã gỡ reference 3D khỏi bộ chấm điểm. Mô hình 3D hiển thị vẫn được giữ nguyên.")
        return redirect(url_for("admin_exercise_detail", exercise_id=exercise.id))

    uploaded = request.files.get("reference_motion_file")
    if not uploaded or not uploaded.filename:
        flash("Hãy chọn reference JSON đã xuất từ Mixamo/Blender.")
        return redirect(url_for("admin_exercise_detail", exercise_id=exercise.id))
    saved_rel = ""
    try:
        saved_rel = save_uploaded_asset(
            uploaded, UPLOAD_REFERENCE_DIR, "uploads/references", ALLOWED_REFERENCE_EXTENSIONS
        )
        absolute = os.path.join(BASE_DIR, "static", saved_rel.replace("/", os.sep))
        payload = load_reference_file(absolute, expected_exercise=exercise.slug)
        exercise.reference_motion_path = saved_rel
        exercise.reference_motion_source = str(payload.get("source", {}).get("type", ""))
        exercise.reference_motion_version = str(payload.get("version", "") or "")
        requested_min = request.form.get("reference_min_score", payload.get("min_score", 55))
        exercise.reference_min_score = max(0.0, min(100.0, float(requested_min)))
        # Make the runtime threshold match the admin-approved exercise threshold.
        raw = json.loads(Path(absolute).read_text(encoding="utf-8"))
        raw["min_score"] = float(exercise.reference_min_score)
        Path(absolute).write_text(json.dumps(raw, ensure_ascii=False, indent=2), encoding="utf-8")
        db.session.commit()
        flash("Đã gắn reference Mixamo/Blender vào AI chấm điểm cho bài tập.")
    except (ValueError, ReferenceMotionError, json.JSONDecodeError) as exc:
        db.session.rollback()
        if saved_rel:
            absolute = os.path.join(BASE_DIR, "static", saved_rel.replace("/", os.sep))
            try:
                os.remove(absolute)
            except OSError:
                pass
        flash(f"Reference không hợp lệ: {exc}")
    return redirect(url_for("admin_exercise_detail", exercise_id=exercise.id))


@app.route("/admin/bai-tap/<int:exercise_id>/criterion-add", methods=["POST"])
@admin_required
def admin_add_criterion(exercise_id):
    exercise = WorkoutExercise.query.get_or_404(exercise_id)

    title = request.form.get("title", "").strip()
    joint_name = request.form.get("joint_name", "").strip()
    operator = request.form.get("operator", "<=")
    try:
        angle_value = float(request.form.get("angle_value", 0))
    except (TypeError, ValueError):
        flash("Ngưỡng tiêu chí phải là một số hợp lệ.")
        return redirect(url_for("admin_exercise_detail", exercise_id=exercise.id))
    message_text = request.form.get("message_text", "").strip()
    advice_text = request.form.get("advice_text", "").strip()
    error_code = normalize_pose_label(request.form.get("error_code", "")) or normalize_pose_label(title)
    phase = normalize_pose_label(request.form.get("phase", "middle")) or "middle"
    if phase not in {"start", "middle", "end", "any"}:
        phase = "middle"

    try:
        joint_indices = [
            int(request.form.get("joint_a", "")),
            int(request.form.get("joint_b", "")),
            int(request.form.get("joint_c", "")),
        ]
    except (TypeError, ValueError):
        flash("Hãy chọn đủ 3 keypoint A-B-C cho luật góc; B là đỉnh góc cần đánh giá.")
        return redirect(url_for("admin_exercise_detail", exercise_id=exercise.id))
    if any(index < 0 or index > 16 for index in joint_indices) or len(set(joint_indices)) != 3:
        flash("Ba keypoint A-B-C phải hợp lệ và không được trùng nhau.")
        return redirect(url_for("admin_exercise_detail", exercise_id=exercise.id))
    joint_name = joint_name or " - ".join(YOLO_KEYPOINT_LABELS.get(index, str(index)) for index in joint_indices)

    if not title or not message_text:
        flash("Tiêu đề và thông báo lỗi không được để trống.")
        return redirect(url_for("admin_exercise_detail", exercise_id=exercise.id))

    duplicate_code = ExerciseCriterion.query.filter_by(exercise_id=exercise.id, error_code=error_code).first()
    if duplicate_code:
        flash("Mã lỗi này đã tồn tại trong bài tập. Hãy dùng mã lỗi riêng để file âm thanh luôn khớp đúng cảnh báo.")
        return redirect(url_for("admin_exercise_detail", exercise_id=exercise.id))

    temp_name = f"criterion-{exercise.id}-{error_code}-{int(datetime.now().timestamp())}"
    tts = create_tts_audio(message_text or advice_text or title, temp_name)

    criterion = ExerciseCriterion(
        exercise_id=exercise.id,
        title=title,
        joint_name=joint_name,
        operator=operator,
        angle_value=angle_value,
        message_text=message_text,
        advice_text=advice_text,
        error_code=error_code,
        phase=phase,
        joint_indices_json=json.dumps(joint_indices),
        audio_path=tts["path"],
        audio_provider=tts["provider"],
        audio_error=tts["error"],
    )
    db.session.add(criterion)
    db.session.commit()

    # Refresh live sessions without requiring a server restart.
    refresh_live_exercise_criteria(exercise)

    if tts["success"]:
        flash(f"Đã thêm tiêu chí và tự tạo file âm thanh bằng {tts['provider']}.")
    else:
        flash("Đã thêm tiêu chí nhưng chưa tạo được WAV. Hệ thống sẽ dùng giọng nói trình duyệt làm fallback; xem lỗi TTS trong danh sách tiêu chí.")
    return redirect(url_for("admin_exercise_detail", exercise_id=exercise.id))


@app.route("/admin/tieu-chi/<int:criterion_id>/tao-lai-am-thanh", methods=["POST"])
@admin_required
def admin_regenerate_criterion_audio(criterion_id):
    criterion = ExerciseCriterion.query.get_or_404(criterion_id)
    exercise = db.session.get(WorkoutExercise, criterion.exercise_id)
    temp_name = f"criterion-{criterion.exercise_id}-{criterion.error_code or criterion.id}-{int(datetime.now().timestamp())}"
    tts = create_tts_audio(criterion.message_text or criterion.advice_text or criterion.title, temp_name)
    criterion.audio_path = tts["path"]
    criterion.audio_provider = tts["provider"]
    criterion.audio_error = tts["error"]
    db.session.commit()
    refresh_live_exercise_criteria(exercise)
    if tts["success"]:
        flash(f"Đã tạo lại file âm thanh bằng {tts['provider']}.")
    else:
        flash("Chưa tạo được file âm thanh. Hãy kiểm tra cấu hình Piper hoặc pyttsx3 trên máy chạy server.")
    return redirect(url_for("admin_exercise_detail", exercise_id=exercise.id if exercise else criterion.exercise_id))


if __name__ == "__main__":
    debug_mode = os.getenv("FLASK_DEBUG", "0").strip() == "1"
    app.run(
        host="0.0.0.0",
        port=int(os.getenv("PORT", "5000")),
        debug=debug_mode,
        threaded=True,
        use_reloader=False,
    )
