"""Explainable 0-100 rep-quality scoring for FitMotion AI.

Each exercise exposes named components and fixed weights. This makes the score
traceable in the report and defendable before the thesis committee instead of
being an unexplained single number.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

from training_levels import DEFAULT_LEVEL, normalize_training_level, scoring_profile


def clamp(value: float, low: float = 0.0, high: float = 100.0) -> float:
    return max(low, min(high, float(value)))


def linear_score(value: float, worst: float, best: float) -> float:
    """Map a metric linearly to 0-100 using explicit worst/best anchors."""
    value = float(value)
    worst = float(worst)
    best = float(best)
    if best == worst:
        return 100.0
    ratio = (value - worst) / (best - worst)
    return clamp(ratio * 100.0)


@dataclass(frozen=True)
class RepScore:
    total: int
    components: Dict[str, int]
    weights: Dict[str, float]


def _weighted(components: Dict[str, float], weights: Dict[str, float]) -> RepScore:
    missing = set(weights) - set(components)
    if missing:
        raise ValueError(f"Missing scoring components: {sorted(missing)}")

    total_weight = sum(float(value) for value in weights.values())
    if total_weight <= 0:
        raise ValueError("Scoring weights must have a positive sum")

    total = sum(clamp(components[name]) * float(weights[name]) for name in weights) / total_weight
    return RepScore(
        total=int(round(clamp(total))),
        components={name: int(round(clamp(value))) for name, value in components.items()},
        weights={name: float(value) for name, value in weights.items()},
    )


def score_squat(
    min_knee: float,
    min_torso: float,
    confidence: float,
    stability: float,
    level: str = DEFAULT_LEVEL,
) -> RepScore:
    """Score squat quality using anchors/weights for the selected level."""
    level = normalize_training_level(level)
    profile = scoring_profile("squat", level)
    components = {
        # A smaller minimum knee angle represents a deeper squat.
        "depth": linear_score(min_knee, **profile["depth"]),
        # A torso angle closer to an upright/controlled posture is better.
        "torso": linear_score(min_torso, **profile["torso"]),
        "confidence": clamp(float(confidence) * 100.0),
        "stability": clamp(stability),
    }
    return _weighted(components, profile["weights"])


def score_pushup(
    min_elbow: float,
    min_body: float,
    confidence: float,
    stability: float,
    level: str = DEFAULT_LEVEL,
) -> RepScore:
    """Score push-up quality using anchors/weights for the selected level."""
    level = normalize_training_level(level)
    profile = scoring_profile("pushup", level)
    components = {
        "depth": linear_score(min_elbow, **profile["depth"]),
        "body_line": linear_score(min_body, **profile["body_line"]),
        "confidence": clamp(float(confidence) * 100.0),
        "stability": clamp(stability),
    }
    return _weighted(components, profile["weights"])


def score_curl(
    min_elbow: float,
    max_elbow: float,
    elbow_shift: float,
    confidence: float,
    stability: float,
    extension_target: float | None = None,
    level: str = DEFAULT_LEVEL,
) -> RepScore:
    """Score curl quality without requiring an anatomical 180-degree lockout.

    ``extension_target`` is the user's calibrated comfortable starting angle.
    When supplied, extension is scored by how closely the rep returns to that
    personal baseline. This is more defensible than forcing every user to reach
    the same absolute elbow angle.
    """
    level = normalize_training_level(level)
    profile = scoring_profile("curl", level)
    if extension_target is None:
        # Without a personal baseline, score the absolute return angle using
        # an equivalent level-specific range.
        absolute_worst = 120.0 if level == DEFAULT_LEVEL else (110.0 if level == "easy" else 135.0)
        extension_score = linear_score(max_elbow, worst=absolute_worst, best=165.0)
    else:
        extension_gap = abs(float(extension_target) - float(max_elbow))
        extension_score = linear_score(extension_gap, **profile["extension_gap"])

    components = {
        "flexion": linear_score(min_elbow, **profile["flexion"]),
        "extension": extension_score,
        # Lower displacement of the elbow relative to upper-arm length is better.
        "elbow_control": linear_score(elbow_shift, **profile["elbow_control"]),
        "confidence": clamp(float(confidence) * 100.0),
        "stability": clamp(stability),
    }
    return _weighted(components, profile["weights"])
