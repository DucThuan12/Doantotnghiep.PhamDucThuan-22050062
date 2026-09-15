from datetime import datetime
from database import db


class User(db.Model):
    __tablename__ = "users"

    id = db.Column(db.Integer, primary_key=True)
    fullname = db.Column(db.String(120), nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)
    role = db.Column(db.String(20), default="user", nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    profile = db.relationship("UserProfile", backref="user", uselist=False, cascade="all, delete-orphan")


class UserProfile(db.Model):
    __tablename__ = "user_profiles"

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False)

    age = db.Column(db.Integer, default=18)
    height = db.Column(db.Float, default=170.0)
    weight = db.Column(db.Float, default=60.0)
    goal = db.Column(db.String(100), default="tap nhe")
    health_note = db.Column(db.String(255), default="khong co van de dac biet")

    phone = db.Column(db.String(30), default="")
    contact_address = db.Column(db.String(255), default="")
    emergency_contact_name = db.Column(db.String(120), default="")
    emergency_contact_relation = db.Column(db.String(80), default="")
    emergency_contact_phone = db.Column(db.String(30), default="")

    weekly_target = db.Column(db.Integer, default=45)
    daily_target = db.Column(db.Integer, default=6)
    done_count = db.Column(db.Integer, default=0)
    total_errors = db.Column(db.Integer, default=0)
    calories_burned = db.Column(db.Float, default=0.0)


class WorkoutExercise(db.Model):
    __tablename__ = "workout_exercises"

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(120), nullable=False)
    slug = db.Column(db.String(80), unique=True, nullable=False)
    muscle_group = db.Column(db.String(120), default="")
    age_min = db.Column(db.Integer, default=15)
    age_max = db.Column(db.Integer, default=100)
    calories = db.Column(db.Float, default=0.1)
    difficulty = db.Column(db.String(50), default="co ban")
    side_mode = db.Column(db.String(20), default="none")

    description = db.Column(db.Text, default="")
    guide_text = db.Column(db.Text, default="")
    suitable_for = db.Column(db.Text, default="")
    caution_for = db.Column(db.Text, default="")

    preview_image = db.Column(db.String(255), default="")
    preview_video = db.Column(db.String(255), default="")
    # ``fbx_path`` is retained for backward compatibility with the old source.
    fbx_path = db.Column(db.String(255), default="")
    # Web-ready 3D configuration. The main viewer requires a real FBX/GLB
    # character with rig + animation; the old procedural avatar is no longer
    # presented as a substitute for the thesis 3D simulation.
    model_3d_path = db.Column(db.String(255), default="")
    model_3d_format = db.Column(db.String(20), default="none")
    animation_key = db.Column(db.String(80), default="")
    # Optional one-shot transition before the looping exercise animation.
    # Push-up uses this for standing -> plank, then switches to the main GLB.
    intro_model_3d_path = db.Column(db.String(255), default="")
    intro_model_3d_format = db.Column(db.String(20), default="none")
    intro_animation_key = db.Column(db.String(80), default="")
    reference_motion_path = db.Column(db.String(255), default="")
    reference_motion_source = db.Column(db.String(80), default="")
    reference_motion_version = db.Column(db.String(80), default="")
    reference_min_score = db.Column(db.Float, default=55.0)

    # One trainable pose-state model per exercise, following the benchmark
    # thesis design instead of one global checkpoint for every exercise.
    pose_model_path = db.Column(db.String(320), default="")
    pose_model_metrics_path = db.Column(db.String(320), default="")
    pose_model_status = db.Column(db.String(30), default="not_trained")
    pose_model_accuracy = db.Column(db.Float, default=0.0)
    pose_model_error = db.Column(db.Text, default="")
    pose_model_updated_at = db.Column(db.DateTime, nullable=True)

    is_active = db.Column(db.Boolean, default=True)

    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    criteria = db.relationship("ExerciseCriterion", backref="exercise", cascade="all, delete-orphan")
    labels = db.relationship("ExerciseLabelImage", backref="exercise", cascade="all, delete-orphan")
    training_media = db.relationship("ExerciseTrainingMedia", backref="exercise", cascade="all, delete-orphan")
    level_configs = db.relationship(
        "ExerciseLevelConfig", back_populates="exercise", cascade="all, delete-orphan"
    )


class ExerciseLevelConfig(db.Model):
    """Admin-editable workload/session criteria for one exercise level."""
    __tablename__ = "exercise_level_configs"
    __table_args__ = (
        db.UniqueConstraint("exercise_id", "training_level", name="uq_exercise_level_config"),
    )

    id = db.Column(db.Integer, primary_key=True)
    exercise_id = db.Column(db.Integer, db.ForeignKey("workout_exercises.id"), nullable=False, index=True)
    training_level = db.Column(db.String(20), nullable=False, default="medium")
    set_count = db.Column(db.Integer, nullable=False, default=2)
    rep_target = db.Column(db.Integer, nullable=False, default=10)
    rest_seconds = db.Column(db.Integer, nullable=False, default=90)
    max_idle_seconds = db.Column(db.Integer, nullable=False, default=600)
    max_session_seconds = db.Column(db.Integer, nullable=False, default=2400)
    min_good_rep_ratio = db.Column(db.Float, nullable=False, default=0.70)
    min_quality_score = db.Column(db.Float, nullable=False, default=60.0)
    min_confidence = db.Column(db.Float, nullable=False, default=0.60)
    min_stability_score = db.Column(db.Float, nullable=False, default=55.0)
    max_tracking_abort_count = db.Column(db.Integer, nullable=False, default=4)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    exercise = db.relationship("WorkoutExercise", back_populates="level_configs")


class WorkoutPlan(db.Model):
    __tablename__ = "workout_plans"

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False)
    exercise_id = db.Column(db.Integer, db.ForeignKey("workout_exercises.id"), nullable=False)

    workout_date = db.Column(db.String(20), nullable=False)
    set_count = db.Column(db.Integer, default=1)
    rep_target = db.Column(db.Integer, default=10)
    training_level = db.Column(db.String(20), default="medium", nullable=False)
    status = db.Column(db.String(20), default="pending")

    schedule_id = db.Column(db.Integer, db.ForeignKey("workout_schedules.id"), nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)


class WorkoutProgram(db.Model):
    """Reusable workout program that can be private, shared, copied and applied.

    ``source_type`` distinguishes a plan created manually from one generated by
    FitMotion's profile-based expert rules.  A profile snapshot is stored so a
    shared program keeps the reasoning context that existed when it was made.
    """
    __tablename__ = "workout_programs"

    id = db.Column(db.Integer, primary_key=True)
    owner_user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False, index=True)
    source_program_id = db.Column(db.Integer, db.ForeignKey("workout_programs.id"), nullable=True)
    title = db.Column(db.String(180), nullable=False)
    description = db.Column(db.Text, default="")
    goal = db.Column(db.String(100), default="")
    health_focus = db.Column(db.String(120), default="")
    source_type = db.Column(db.String(30), default="user", nullable=False)
    visibility = db.Column(db.String(20), default="private", nullable=False, index=True)
    moderation_status = db.Column(db.String(20), default="visible", nullable=False, index=True)
    profile_snapshot_json = db.Column(db.Text, default="{}")
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    items = db.relationship(
        "WorkoutProgramItem",
        backref="program",
        cascade="all, delete-orphan",
        order_by="WorkoutProgramItem.weekday, WorkoutProgramItem.sort_order, WorkoutProgramItem.id",
    )


class WorkoutProgramItem(db.Model):
    __tablename__ = "workout_program_items"
    __table_args__ = (
        db.CheckConstraint("weekday >= 0 AND weekday <= 6", name="ck_program_item_weekday"),
    )

    id = db.Column(db.Integer, primary_key=True)
    program_id = db.Column(db.Integer, db.ForeignKey("workout_programs.id"), nullable=False, index=True)
    exercise_id = db.Column(db.Integer, db.ForeignKey("workout_exercises.id"), nullable=False)
    weekday = db.Column(db.Integer, nullable=False)
    set_count = db.Column(db.Integer, default=1, nullable=False)
    rep_target = db.Column(db.Integer, default=10, nullable=False)
    training_level = db.Column(db.String(20), default="medium", nullable=False)
    sort_order = db.Column(db.Integer, default=0, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)


class WorkoutProgramFavorite(db.Model):
    __tablename__ = "workout_program_favorites"
    __table_args__ = (
        db.UniqueConstraint("user_id", "program_id", name="uq_program_favorite_user_program"),
    )

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False, index=True)
    program_id = db.Column(db.Integer, db.ForeignKey("workout_programs.id"), nullable=False, index=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)


class WorkoutProgramReview(db.Model):
    __tablename__ = "workout_program_reviews"
    __table_args__ = (
        db.UniqueConstraint("user_id", "program_id", name="uq_program_review_user_program"),
        db.CheckConstraint("rating >= 1 AND rating <= 5", name="ck_program_review_rating"),
    )

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False, index=True)
    program_id = db.Column(db.Integer, db.ForeignKey("workout_programs.id"), nullable=False, index=True)
    rating = db.Column(db.Integer, nullable=False, default=5)
    comment = db.Column(db.Text, default="")
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class WorkoutSchedule(db.Model):
    """A durable recurring schedule owned by one user.

    The schedule stores the date range and lifecycle. Individual weekday/
    exercise assignments live in ``WorkoutScheduleItem`` and are materialized
    into ``WorkoutPlan`` rows so the existing workout flow can keep using the
    same plan table.
    """
    __tablename__ = "workout_schedules"

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False, index=True)
    name = db.Column(db.String(160), nullable=False)
    start_date = db.Column(db.String(20), nullable=False)
    end_date = db.Column(db.String(20), nullable=False)
    status = db.Column(db.String(20), default="active", nullable=False)
    source_program_id = db.Column(db.Integer, db.ForeignKey("workout_programs.id"), nullable=True, index=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    items = db.relationship(
        "WorkoutScheduleItem",
        backref="schedule",
        cascade="all, delete-orphan",
        order_by="WorkoutScheduleItem.weekday, WorkoutScheduleItem.sort_order, WorkoutScheduleItem.id",
    )


class WorkoutScheduleItem(db.Model):
    """One exercise assignment repeated on a weekday inside a schedule."""
    __tablename__ = "workout_schedule_items"
    __table_args__ = (
        db.CheckConstraint("weekday >= 0 AND weekday <= 6", name="ck_schedule_item_weekday"),
    )

    id = db.Column(db.Integer, primary_key=True)
    schedule_id = db.Column(db.Integer, db.ForeignKey("workout_schedules.id"), nullable=False, index=True)
    exercise_id = db.Column(db.Integer, db.ForeignKey("workout_exercises.id"), nullable=False)
    weekday = db.Column(db.Integer, nullable=False)
    set_count = db.Column(db.Integer, default=1, nullable=False)
    rep_target = db.Column(db.Integer, default=10, nullable=False)
    training_level = db.Column(db.String(20), default="medium", nullable=False)
    sort_order = db.Column(db.Integer, default=0, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)


class WorkoutSession(db.Model):
    __tablename__ = "workout_sessions"

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False)
    exercise_id = db.Column(db.Integer, db.ForeignKey("workout_exercises.id"), nullable=False)

    session_date = db.Column(
        db.String(20),
        nullable=False,
        default=lambda: datetime.now().date().isoformat()
    )

    total_rep = db.Column(db.Integer, default=0)
    good_rep = db.Column(db.Integer, default=0)
    total_error = db.Column(db.Integer, default=0)

    # Backward-compatible aggregate confidence column used by older screens.
    confidence_avg = db.Column(db.Float, default=0.0)
    # Week-6 metrics: keep model confidence separate from rep-quality score.
    keypoint_confidence_avg = db.Column(db.Float, default=0.0)
    quality_score_avg = db.Column(db.Float, default=0.0)
    training_level = db.Column(db.String(20), default="medium", nullable=False)
    set_count = db.Column(db.Integer, default=1, nullable=False)
    rep_target = db.Column(db.Integer, default=10, nullable=False)  # per set
    duration_seconds = db.Column(db.Integer, default=0, nullable=False)
    effectiveness_status = db.Column(db.String(30), default="partial", nullable=False)
    session_summary_json = db.Column(db.Text, default="{}")
    reference_score_avg = db.Column(db.Float, default=0.0)
    pose_guard_confirmed_ratio = db.Column(db.Float, default=0.0)
    # JSON text keeps per-rep evidence without forcing a new relational table
    # during the current 12-week project phase.
    rep_details_json = db.Column(db.Text, default="[]")
    error_codes_json = db.Column(db.Text, default="{}")

    phase_start_error = db.Column(db.Integer, default=0)
    phase_middle_error = db.Column(db.Integer, default=0)
    phase_end_error = db.Column(db.Integer, default=0)

    schedule_id = db.Column(db.Integer, db.ForeignKey("workout_schedules.id"), nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)


class WorkoutSessionDraft(db.Model):
    """Durable checkpoint for a workout that has not been finalized yet."""
    __tablename__ = "workout_session_drafts"
    __table_args__ = (
        db.UniqueConstraint("user_id", "exercise_id", "session_date", name="uq_workout_draft_user_exercise_date"),
    )

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False)
    exercise_id = db.Column(db.Integer, db.ForeignKey("workout_exercises.id"), nullable=False)
    session_date = db.Column(db.String(20), nullable=False)

    total_rep = db.Column(db.Integer, default=0)
    good_rep = db.Column(db.Integer, default=0)
    total_error = db.Column(db.Integer, default=0)
    keypoint_confidence_avg = db.Column(db.Float, default=0.0)
    quality_score_avg = db.Column(db.Float, default=0.0)
    training_level = db.Column(db.String(20), default="medium", nullable=False)
    state_json = db.Column(db.Text, default="{}")

    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class ExerciseCriterion(db.Model):
    __tablename__ = "exercise_criteria"

    id = db.Column(db.Integer, primary_key=True)
    exercise_id = db.Column(db.Integer, db.ForeignKey("workout_exercises.id"), nullable=False)

    title = db.Column(db.String(120), nullable=False)
    joint_name = db.Column(db.String(120), default="")
    operator = db.Column(db.String(20), default="<=")
    angle_value = db.Column(db.Float, default=0.0)

    message_text = db.Column(db.Text, default="")
    advice_text = db.Column(db.Text, default="")

    # Runtime key links an Admin-defined rule to the realtime error code.
    # Audio is generated once when the rule is saved, then replayed on workout.
    error_code = db.Column(db.String(80), default="")
    phase = db.Column(db.String(30), default="middle")
    # Optional YOLO-Pose 17-keypoint angle rule.  The value stores three
    # indices [A, B, C]; the measured angle is A-B-C with B as the vertex.
    # Keeping it as JSON preserves backwards compatibility with old criteria.
    joint_indices_json = db.Column(db.Text, default="[]")
    audio_path = db.Column(db.String(255), default="")
    audio_provider = db.Column(db.String(30), default="")
    audio_error = db.Column(db.Text, default="")

    created_at = db.Column(db.DateTime, default=datetime.utcnow)


class ExerciseLabelImage(db.Model):
    __tablename__ = "exercise_label_images"

    id = db.Column(db.Integer, primary_key=True)
    exercise_id = db.Column(db.Integer, db.ForeignKey("workout_exercises.id"), nullable=False)

    label_name = db.Column(db.String(120), nullable=False)
    frame_index = db.Column(db.Integer, default=1)
    image_path = db.Column(db.String(255), nullable=False)

    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class ExerciseTrainingMedia(db.Model):
    """Labelled image/video source used to train/test the pose-phase classifier."""
    __tablename__ = "exercise_training_media"

    id = db.Column(db.Integer, primary_key=True)
    exercise_id = db.Column(db.Integer, db.ForeignKey("workout_exercises.id"), nullable=False, index=True)
    label_name = db.Column(db.String(120), nullable=False, index=True)
    media_type = db.Column(db.String(20), nullable=False, default="image")
    file_path = db.Column(db.String(320), nullable=False)
    source_name = db.Column(db.String(255), default="")
    created_at = db.Column(db.DateTime, default=datetime.utcnow)


class EmergencyAlert(db.Model):
    """Persistent admin notification created from a confirmed fall/immobility event."""
    __tablename__ = "emergency_alerts"

    id = db.Column(db.Integer, primary_key=True)
    event_key = db.Column(db.String(180), unique=True, nullable=False, index=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False, index=True)
    exercise_id = db.Column(db.Integer, db.ForeignKey("workout_exercises.id"), nullable=True)
    exercise_slug = db.Column(db.String(80), default="")
    phase = db.Column(db.String(80), default="")
    reason = db.Column(db.Text, default="")
    message = db.Column(db.Text, default="")
    evidence_path = db.Column(db.String(320), default="")
    body_angle = db.Column(db.Float, default=0.0)
    status = db.Column(db.String(20), default="unread", nullable=False, index=True)

    user_name_snapshot = db.Column(db.String(120), default="")
    user_email_snapshot = db.Column(db.String(120), default="")
    personal_phone_snapshot = db.Column(db.String(30), default="")
    address_snapshot = db.Column(db.String(255), default="")
    emergency_contact_name_snapshot = db.Column(db.String(120), default="")
    emergency_relation_snapshot = db.Column(db.String(80), default="")
    emergency_phone_snapshot = db.Column(db.String(30), default="")

    triggered_at = db.Column(db.DateTime, default=datetime.utcnow, index=True)
    read_at = db.Column(db.DateTime, nullable=True)
    resolved_at = db.Column(db.DateTime, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
