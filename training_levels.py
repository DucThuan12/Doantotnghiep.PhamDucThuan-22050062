"""Training-level configuration for FitMotion AI core exercises.

The exercise catalog has a static ``difficulty`` label, but a user workout needs
its own level because the same exercise may be performed at Easy, Medium or Hard
on different days.  This module is deliberately pure-Python so level rules can
be tested without Flask, OpenCV or model weights.
"""
from __future__ import annotations

from copy import deepcopy

LEVEL_EASY = "easy"
LEVEL_MEDIUM = "medium"
LEVEL_HARD = "hard"
DEFAULT_LEVEL = LEVEL_MEDIUM
VALID_LEVELS = (LEVEL_EASY, LEVEL_MEDIUM, LEVEL_HARD)

LEVEL_LABELS = {
    LEVEL_EASY: "Dễ",
    LEVEL_MEDIUM: "Trung bình",
    LEVEL_HARD: "Khó",
}

LEVEL_OPTIONS = tuple((key, LEVEL_LABELS[key]) for key in VALID_LEVELS)

# Aliases are accepted so old/demo data or Vietnamese form values never create
# a fourth accidental level in the database.
_LEVEL_ALIASES = {
    "easy": LEVEL_EASY,
    "de": LEVEL_EASY,
    "dễ": LEVEL_EASY,
    "co ban": LEVEL_EASY,
    "cơ bản": LEVEL_EASY,
    "beginner": LEVEL_EASY,
    "medium": LEVEL_MEDIUM,
    "trung binh": LEVEL_MEDIUM,
    "trung bình": LEVEL_MEDIUM,
    "normal": LEVEL_MEDIUM,
    "hard": LEVEL_HARD,
    "kho": LEVEL_HARD,
    "khó": LEVEL_HARD,
    "nang cao": LEVEL_HARD,
    "nâng cao": LEVEL_HARD,
    "advanced": LEVEL_HARD,
}


def normalize_training_level(value: object, default: str = DEFAULT_LEVEL) -> str:
    text = str(value or "").strip().lower()
    if text in VALID_LEVELS:
        return text
    return _LEVEL_ALIASES.get(text, default if default in VALID_LEVELS else DEFAULT_LEVEL)


def training_level_label(value: object) -> str:
    return LEVEL_LABELS[normalize_training_level(value)]


# Thresholds control *recognition and rule evaluation*.  Medium preserves the
# previously validated demo values so the upgrade does not silently change the
# existing behaviour.  Easy relaxes required ROM/form; Hard requires more ROM
# and tighter form before a rep is considered good.
THRESHOLD_PROFILES = {
    "squat": {
        LEVEL_EASY: {
            "calibration_min_angle": 125.0,
            "calibration_frames": 4,
            "start_drop_from_baseline": 8.0,
            "finish_tolerance": 24.0,
            "minimum_cycle_rom": 12.0,
            "bottom_angle": 140.0,
            "shallow_limit": 142.0,
            "torso_limit": 112.0,
            "good_score_threshold": 55,
        },
        LEVEL_MEDIUM: {
            "calibration_min_angle": 130.0,
            "calibration_frames": 5,
            "start_drop_from_baseline": 10.0,
            "finish_tolerance": 20.0,
            "minimum_cycle_rom": 16.0,
            "bottom_angle": 128.0,
            "shallow_limit": 132.0,
            "torso_limit": 122.0,
            "good_score_threshold": 60,
        },
        LEVEL_HARD: {
            "calibration_min_angle": 138.0,
            "calibration_frames": 5,
            "start_drop_from_baseline": 12.0,
            "finish_tolerance": 16.0,
            "minimum_cycle_rom": 24.0,
            "bottom_angle": 115.0,
            "shallow_limit": 120.0,
            "torso_limit": 135.0,
            "good_score_threshold": 72,
        },
    },
    "pushup": {
        LEVEL_EASY: {
            "calibration_min_angle": 105.0,
            "calibration_frames": 4,
            "start_drop_from_baseline": 6.0,
            "finish_tolerance": 26.0,
            "minimum_cycle_rom": 10.0,
            "bottom_angle": 138.0,
            "shallow_limit": 132.0,
            "body_line_limit": 122.0,
            "good_score_threshold": 55,
        },
        LEVEL_MEDIUM: {
            "calibration_min_angle": 112.0,
            "calibration_frames": 4,
            "start_drop_from_baseline": 8.0,
            "finish_tolerance": 22.0,
            "minimum_cycle_rom": 14.0,
            "bottom_angle": 128.0,
            "shallow_limit": 120.0,
            "body_line_limit": 135.0,
            "good_score_threshold": 60,
        },
        LEVEL_HARD: {
            "calibration_min_angle": 125.0,
            "calibration_frames": 5,
            "start_drop_from_baseline": 10.0,
            "finish_tolerance": 16.0,
            "minimum_cycle_rom": 22.0,
            "bottom_angle": 106.0,
            "shallow_limit": 112.0,
            "body_line_limit": 150.0,
            "good_score_threshold": 72,
        },
    },
    "curl": {
        LEVEL_EASY: {
            "calibration_min_angle": 110.0,
            "calibration_max_angle": 160.0,
            "calibration_frames": 4,
            "start_drop_from_baseline": 10.0,
            "minimum_range_of_motion": 24.0,
            "full_flexion_ceiling": 120.0,
            "finish_tolerance": 22.0,
            "incomplete_extension_tolerance": 32.0,
            "elbow_shift_limit": 60.0,
            "good_score_threshold": 55,
        },
        LEVEL_MEDIUM: {
            "calibration_min_angle": 118.0,
            "calibration_max_angle": 165.0,
            "calibration_frames": 5,
            "start_drop_from_baseline": 12.0,
            "minimum_range_of_motion": 30.0,
            "full_flexion_ceiling": 112.0,
            "finish_tolerance": 18.0,
            "incomplete_extension_tolerance": 28.0,
            "elbow_shift_limit": 48.0,
            "good_score_threshold": 64,
        },
        LEVEL_HARD: {
            "calibration_min_angle": 125.0,
            "calibration_max_angle": 165.0,
            "calibration_frames": 5,
            "start_drop_from_baseline": 14.0,
            "minimum_range_of_motion": 42.0,
            "full_flexion_ceiling": 88.0,
            "finish_tolerance": 12.0,
            "incomplete_extension_tolerance": 20.0,
            "elbow_shift_limit": 30.0,
            "good_score_threshold": 72,
        },
    },
}


# Scoring profiles are intentionally separate from counting thresholds.
# A complete motion can still be counted while receiving a low quality score.
SCORING_PROFILES = {
    "squat": {
        LEVEL_EASY: {
            "depth": {"worst": 150.0, "best": 112.0},
            "torso": {"worst": 102.0, "best": 155.0},
            "weights": {"depth": 0.35, "torso": 0.20, "confidence": 0.25, "stability": 0.20},
        },
        LEVEL_MEDIUM: {
            "depth": {"worst": 145.0, "best": 95.0},
            "torso": {"worst": 110.0, "best": 165.0},
            "weights": {"depth": 0.40, "torso": 0.25, "confidence": 0.20, "stability": 0.15},
        },
        LEVEL_HARD: {
            "depth": {"worst": 130.0, "best": 85.0},
            "torso": {"worst": 125.0, "best": 175.0},
            "weights": {"depth": 0.45, "torso": 0.30, "confidence": 0.15, "stability": 0.10},
        },
    },
    "pushup": {
        LEVEL_EASY: {
            "depth": {"worst": 148.0, "best": 108.0},
            "body_line": {"worst": 110.0, "best": 165.0},
            "weights": {"depth": 0.35, "body_line": 0.25, "confidence": 0.23, "stability": 0.17},
        },
        LEVEL_MEDIUM: {
            "depth": {"worst": 140.0, "best": 90.0},
            "body_line": {"worst": 120.0, "best": 175.0},
            "weights": {"depth": 0.40, "body_line": 0.30, "confidence": 0.18, "stability": 0.12},
        },
        LEVEL_HARD: {
            "depth": {"worst": 122.0, "best": 75.0},
            "body_line": {"worst": 140.0, "best": 180.0},
            "weights": {"depth": 0.45, "body_line": 0.35, "confidence": 0.12, "stability": 0.08},
        },
    },
    "curl": {
        LEVEL_EASY: {
            "flexion": {"worst": 128.0, "best": 75.0},
            "extension_gap": {"worst": 35.0, "best": 0.0},
            "elbow_control": {"worst": 80.0, "best": 15.0},
            "weights": {"flexion": 0.25, "extension": 0.20, "elbow_control": 0.15, "confidence": 0.25, "stability": 0.15},
        },
        LEVEL_MEDIUM: {
            "flexion": {"worst": 120.0, "best": 55.0},
            "extension_gap": {"worst": 30.0, "best": 0.0},
            "elbow_control": {"worst": 70.0, "best": 8.0},
            "weights": {"flexion": 0.30, "extension": 0.22, "elbow_control": 0.22, "confidence": 0.16, "stability": 0.10},
        },
        LEVEL_HARD: {
            "flexion": {"worst": 105.0, "best": 45.0},
            "extension_gap": {"worst": 18.0, "best": 0.0},
            "elbow_control": {"worst": 45.0, "best": 5.0},
            "weights": {"flexion": 0.32, "extension": 0.25, "elbow_control": 0.25, "confidence": 0.10, "stability": 0.08},
        },
    },
}


def threshold_profile(exercise_key: str, level: object) -> dict:
    key = "curl" if str(exercise_key).startswith("curl") else str(exercise_key)
    normalized = normalize_training_level(level)
    profiles = THRESHOLD_PROFILES.get(key)
    if profiles is None:
        raise KeyError(f"Unsupported level profile: {exercise_key}")
    return deepcopy(profiles[normalized])


def scoring_profile(exercise_key: str, level: object) -> dict:
    key = "curl" if str(exercise_key).startswith("curl") else str(exercise_key)
    normalized = normalize_training_level(level)
    profiles = SCORING_PROFILES.get(key)
    if profiles is None:
        raise KeyError(f"Unsupported scoring profile: {exercise_key}")
    return deepcopy(profiles[normalized])

# Workload/session criteria complement the technical thresholds above.  The
# defaults are product/demo defaults, not medical prescriptions.  Admin can
# override them per exercise in the database, following the same architectural
# idea as the reference thesis: each level carries its own sets/reps and the
# expert layer decides whether the completed session actually met that level.
WORKLOAD_PROFILES = {
    "squat": {
        LEVEL_EASY: {
            "default_sets": 2,
            "reps_per_set": 8,
            "rest_seconds": 90,
            "max_idle_seconds": 600,
            "max_session_seconds": 2400,
            "min_good_rep_ratio": 0.60,
            "min_quality_avg": 55.0,
            "min_confidence_avg": 0.55,
            "min_stability_avg": 45.0,
            "max_tracking_abort_count": 5,
        },
        LEVEL_MEDIUM: {
            "default_sets": 3,
            "reps_per_set": 12,
            "rest_seconds": 90,
            "max_idle_seconds": 600,
            "max_session_seconds": 2400,
            "min_good_rep_ratio": 0.70,
            "min_quality_avg": 60.0,
            "min_confidence_avg": 0.60,
            "min_stability_avg": 55.0,
            "max_tracking_abort_count": 4,
        },
        LEVEL_HARD: {
            "default_sets": 4,
            "reps_per_set": 15,
            "rest_seconds": 90,
            "max_idle_seconds": 600,
            "max_session_seconds": 2400,
            "min_good_rep_ratio": 0.80,
            "min_quality_avg": 72.0,
            "min_confidence_avg": 0.65,
            "min_stability_avg": 65.0,
            "max_tracking_abort_count": 3,
        },
    },
    "pushup": {
        LEVEL_EASY: {
            "default_sets": 2,
            "reps_per_set": 5,
            "rest_seconds": 90,
            "max_idle_seconds": 600,
            "max_session_seconds": 2400,
            "min_good_rep_ratio": 0.60,
            "min_quality_avg": 55.0,
            "min_confidence_avg": 0.55,
            "min_stability_avg": 45.0,
            "max_tracking_abort_count": 5,
        },
        LEVEL_MEDIUM: {
            "default_sets": 3,
            "reps_per_set": 10,
            "rest_seconds": 90,
            "max_idle_seconds": 600,
            "max_session_seconds": 2400,
            "min_good_rep_ratio": 0.70,
            "min_quality_avg": 60.0,
            "min_confidence_avg": 0.60,
            "min_stability_avg": 55.0,
            "max_tracking_abort_count": 4,
        },
        LEVEL_HARD: {
            "default_sets": 3,
            "reps_per_set": 15,
            "rest_seconds": 90,
            "max_idle_seconds": 600,
            "max_session_seconds": 2400,
            "min_good_rep_ratio": 0.80,
            "min_quality_avg": 72.0,
            "min_confidence_avg": 0.65,
            "min_stability_avg": 65.0,
            "max_tracking_abort_count": 3,
        },
    },
    "curl": {
        LEVEL_EASY: {
            "default_sets": 2,
            "reps_per_set": 8,
            "rest_seconds": 75,
            "max_idle_seconds": 600,
            "max_session_seconds": 2400,
            "min_good_rep_ratio": 0.60,
            "min_quality_avg": 55.0,
            "min_confidence_avg": 0.55,
            "min_stability_avg": 45.0,
            "max_tracking_abort_count": 5,
        },
        LEVEL_MEDIUM: {
            "default_sets": 3,
            "reps_per_set": 12,
            "rest_seconds": 75,
            "max_idle_seconds": 600,
            "max_session_seconds": 2400,
            "min_good_rep_ratio": 0.70,
            "min_quality_avg": 64.0,
            "min_confidence_avg": 0.60,
            "min_stability_avg": 55.0,
            "max_tracking_abort_count": 4,
        },
        LEVEL_HARD: {
            "default_sets": 3,
            "reps_per_set": 15,
            "rest_seconds": 75,
            "max_idle_seconds": 600,
            "max_session_seconds": 2400,
            "min_good_rep_ratio": 0.80,
            "min_quality_avg": 72.0,
            "min_confidence_avg": 0.65,
            "min_stability_avg": 65.0,
            "max_tracking_abort_count": 3,
        },
    },
    "generic": {
        LEVEL_EASY: {
            "default_sets": 1,
            "reps_per_set": 8,
            "rest_seconds": 90,
            "max_idle_seconds": 600,
            "max_session_seconds": 2400,
            "min_good_rep_ratio": 0.60,
            "min_quality_avg": 55.0,
            "min_confidence_avg": 0.55,
            "min_stability_avg": 45.0,
            "max_tracking_abort_count": 5,
        },
        LEVEL_MEDIUM: {
            "default_sets": 2,
            "reps_per_set": 12,
            "rest_seconds": 90,
            "max_idle_seconds": 600,
            "max_session_seconds": 2400,
            "min_good_rep_ratio": 0.70,
            "min_quality_avg": 60.0,
            "min_confidence_avg": 0.60,
            "min_stability_avg": 55.0,
            "max_tracking_abort_count": 4,
        },
        LEVEL_HARD: {
            "default_sets": 3,
            "reps_per_set": 15,
            "rest_seconds": 90,
            "max_idle_seconds": 600,
            "max_session_seconds": 2400,
            "min_good_rep_ratio": 0.80,
            "min_quality_avg": 72.0,
            "min_confidence_avg": 0.65,
            "min_stability_avg": 65.0,
            "max_tracking_abort_count": 3,
        },
    },
}


def workload_profile(exercise_key: str, level: object) -> dict:
    key = str(exercise_key or "")
    if key.startswith("curl"):
        key = "curl"
    profiles = WORKLOAD_PROFILES.get(key, WORKLOAD_PROFILES["generic"])
    return deepcopy(profiles[normalize_training_level(level)])


def average_rep_stability(rep_records: list[dict] | None) -> float:
    values = []
    for item in rep_records or []:
        if not isinstance(item, dict):
            continue
        try:
            value = float(item.get("stability", 0) or 0)
        except (TypeError, ValueError):
            continue
        if value > 0:
            values.append(value)
    return sum(values) / len(values) if values else 0.0


def evaluate_session_effectiveness(
    *,
    total_rep: int,
    good_rep: int,
    quality_score_avg: float,
    confidence_avg: float,
    rep_records: list[dict] | None,
    tracking_abort_count: int,
    elapsed_seconds: float,
    set_count: int,
    reps_per_set: int,
    criteria: dict,
) -> dict:
    """Evaluate whether a finished workout met the selected level as a whole.

    Counting a rep and completing a level are intentionally separate.  A user
    may reach the requested volume but still fail the level because too many
    reps were technically poor, tracking was unreliable or the session was
    stretched beyond the configured validity window.
    """
    total_rep = max(0, int(total_rep or 0))
    good_rep = min(total_rep, max(0, int(good_rep or 0)))
    set_count = max(1, int(set_count or 1))
    reps_per_set = max(1, int(reps_per_set or 1))
    target_total = set_count * reps_per_set
    good_ratio = (good_rep / total_rep) if total_rep else 0.0
    stability_avg = average_rep_stability(rep_records)

    checks = {
        "volume_met": total_rep >= target_total,
        "good_ratio_met": good_ratio >= float(criteria.get("min_good_rep_ratio", 0.0) or 0.0),
        "quality_met": float(quality_score_avg or 0.0) >= float(criteria.get("min_quality_avg", 0.0) or 0.0),
        "confidence_met": float(confidence_avg or 0.0) >= float(criteria.get("min_confidence_avg", 0.0) or 0.0),
        "stability_met": stability_avg >= float(criteria.get("min_stability_avg", 0.0) or 0.0),
        "tracking_met": int(tracking_abort_count or 0) <= int(criteria.get("max_tracking_abort_count", 999999) or 999999),
        "duration_met": float(elapsed_seconds or 0.0) <= float(criteria.get("max_session_seconds", 10**9) or 10**9),
    }
    reasons = [name for name, ok in checks.items() if not ok]
    return {
        **checks,
        "effective": all(checks.values()),
        "target_total_rep": target_total,
        "completion_ratio": min(1.0, total_rep / target_total) if target_total else 0.0,
        "good_rep_ratio": round(good_ratio, 4),
        "stability_avg": round(stability_avg, 2),
        "reasons": reasons,
    }
