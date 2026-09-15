"""Reference-motion scoring for Mixamo/Blender animations.

A rigged animation is only useful to the thesis if it can become measurable
reference data. Blender exports the animation into a compact JSON sequence of
joint-angle metrics. Realtime processors resample the user's rep to the same
normalized timeline and compute an explainable 0-100 similarity score.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

SCHEMA = "fitmotion.reference.v1"
ALLOWED_SOURCE_TYPES = {"mixamo", "mixamo-blender", "validated-3d"}


class ReferenceMotionError(ValueError):
    pass


def validate_reference_payload(payload: Mapping, expected_exercise: str | None = None) -> dict:
    if not isinstance(payload, Mapping):
        raise ReferenceMotionError("Reference JSON phải là object.")
    if str(payload.get("schema", "")) != SCHEMA:
        raise ReferenceMotionError(f"Reference JSON phải dùng schema {SCHEMA}.")
    exercise = str(payload.get("exercise", "")).strip().lower()
    if not exercise:
        raise ReferenceMotionError("Thiếu exercise trong reference JSON.")
    if expected_exercise:
        def _exercise_key(value):
            value = str(value or "").strip().lower().replace("_", "-")
            aliases = {"push-up": "pushup", "push up": "pushup"}
            return aliases.get(value, value)
        expected = _exercise_key(expected_exercise)
        actual = _exercise_key(exercise)
        # Curl left/right must stay distinct: a right-arm animation cannot be
        # accepted as the scoring reference for curl-left (or vice versa).
        if expected != actual:
            raise ReferenceMotionError("Reference JSON không khớp đúng bài và đúng bên đang cấu hình.")
    source = payload.get("source") or {}
    source_type = str(source.get("type", "")).strip().lower()
    if source_type not in ALLOWED_SOURCE_TYPES:
        raise ReferenceMotionError("Reference phải được xuất từ Mixamo/Blender hoặc nguồn 3D đã xác thực.")
    metrics = [str(item) for item in payload.get("metrics", []) if str(item).strip()]
    if not metrics:
        raise ReferenceMotionError("Reference JSON chưa khai báo metrics.")
    frames = payload.get("frames") or []
    if len(frames) < 8:
        raise ReferenceMotionError("Reference cần tối thiểu 8 frame mẫu.")
    previous_t = -1.0
    clean_frames = []
    for item in frames:
        if not isinstance(item, Mapping):
            raise ReferenceMotionError("Mỗi reference frame phải là object.")
        t = float(item.get("t", -1))
        if not 0.0 <= t <= 1.0 or t < previous_t:
            raise ReferenceMotionError("Giá trị t phải tăng dần trong khoảng 0..1.")
        clean = {"t": t}
        for metric in metrics:
            if metric not in item:
                raise ReferenceMotionError(f"Frame thiếu metric: {metric}")
            clean[metric] = float(item[metric])
        clean_frames.append(clean)
        previous_t = t
    scales = {str(k): max(1.0, float(v)) for k, v in (payload.get("metric_scales") or {}).items()}
    for metric in metrics:
        if metric not in scales:
            values = [frame[metric] for frame in clean_frames]
            scales[metric] = max(15.0, max(values) - min(values))
    return {
        "schema": SCHEMA,
        "exercise": exercise,
        "animation_key": str(payload.get("animation_key", "") or ""),
        "version": str(payload.get("version", "1") or "1"),
        "source": dict(source),
        "metrics": metrics,
        "metric_scales": scales,
        "min_score": max(0.0, min(100.0, float(payload.get("min_score", 55.0)))),
        "frames": clean_frames,
    }


def load_reference_file(path: str | Path, expected_exercise: str | None = None) -> dict:
    path = Path(path)
    if not path.is_file():
        raise ReferenceMotionError("Không tìm thấy reference JSON.")
    payload = json.loads(path.read_text(encoding="utf-8"))
    return validate_reference_payload(payload, expected_exercise=expected_exercise)


def _interp_series(values: Sequence[float], target_count: int) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    if len(arr) == target_count:
        return arr
    if len(arr) == 1:
        return np.repeat(arr, target_count)
    old_x = np.linspace(0.0, 1.0, len(arr))
    new_x = np.linspace(0.0, 1.0, target_count)
    return np.interp(new_x, old_x, arr).astype(np.float32)


def _dtw_normalized_error(user_values: Sequence[float], reference_values: Sequence[float], scale: float) -> float:
    """Return path-length-normalized DTW error divided by the metric scale.

    A realtime rep and a Mixamo animation rarely move at exactly the same pace.
    DTW aligns equivalent movement phases before scoring, so a technically
    similar rep is not penalized merely because the user lowers more slowly or
    rises more quickly than the 3D trainer.
    """
    user = np.asarray(user_values, dtype=np.float32).reshape(-1)
    reference = np.asarray(reference_values, dtype=np.float32).reshape(-1)
    if user.size == 0 or reference.size == 0:
        return 1.0
    n, m = int(user.size), int(reference.size)
    cost = np.full((n + 1, m + 1), np.inf, dtype=np.float64)
    length = np.zeros((n + 1, m + 1), dtype=np.int32)
    cost[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            candidates = ((cost[i - 1, j], i - 1, j), (cost[i, j - 1], i, j - 1), (cost[i - 1, j - 1], i - 1, j - 1))
            prev_cost, pi, pj = min(candidates, key=lambda item: item[0])
            local = abs(float(user[i - 1]) - float(reference[j - 1]))
            cost[i, j] = prev_cost + local
            length[i, j] = length[pi, pj] + 1
    path_length = max(1, int(length[n, m]))
    return float(cost[n, m] / path_length / max(1.0, float(scale)))


@dataclass(frozen=True)
class ReferenceMatch:
    available: bool
    score: int = 0
    min_score: int = 0
    passed: bool = True
    per_metric: dict | None = None
    source_type: str = ""
    version: str = ""
    animation_key: str = ""
    sample_count: int = 0
    message: str = ""

    def to_dict(self) -> dict:
        return {
            "available": bool(self.available),
            "score": int(self.score),
            "min_score": int(self.min_score),
            "passed": bool(self.passed),
            "per_metric": dict(self.per_metric or {}),
            "source_type": self.source_type,
            "version": self.version,
            "animation_key": self.animation_key,
            "sample_count": int(self.sample_count),
            "message": self.message,
        }


class ReferenceMotionMatcher:
    def __init__(self, reference_path: str | None = None, expected_exercise: str | None = None):
        self.reference_path = str(reference_path or "")
        self.expected_exercise = expected_exercise
        self.reference = None
        self.load_error = ""
        self.user_frames: list[dict] = []
        if self.reference_path:
            try:
                self.reference = load_reference_file(self.reference_path, expected_exercise)
            except Exception as exc:
                self.load_error = str(exc)

    @property
    def available(self) -> bool:
        return self.reference is not None

    def start_rep(self):
        self.user_frames = []

    def observe(self, metrics: Mapping[str, float]):
        if not self.available:
            return
        needed = self.reference["metrics"]
        if all(metric in metrics for metric in needed):
            self.user_frames.append({metric: float(metrics[metric]) for metric in needed})

    def finish_rep(self, min_score_override: float | None = None) -> ReferenceMatch:
        if not self.available:
            return ReferenceMatch(False, message=self.load_error or "Chưa gắn reference Mixamo.")
        ref = self.reference
        minimum = int(round(ref["min_score"] if min_score_override is None else max(0.0, min(100.0, float(min_score_override)))))
        if len(self.user_frames) < 4:
            return ReferenceMatch(
                True, 0, minimum, False,
                source_type=str(ref["source"].get("type", "")),
                version=ref["version"], animation_key=ref["animation_key"],
                sample_count=len(self.user_frames),
                message="Chưa đủ frame người tập để so với animation chuẩn.",
            )
        per_metric = {}
        normalized_errors = []
        for metric in ref["metrics"]:
            user_values = [item[metric] for item in self.user_frames]
            expected_values = [item[metric] for item in ref["frames"]]
            scale = max(1.0, float(ref["metric_scales"][metric]))
            normalized_error = _dtw_normalized_error(user_values, expected_values, scale)
            metric_score = int(round(max(0.0, min(100.0, 100.0 * (1.0 - normalized_error)))))
            per_metric[metric] = metric_score
            normalized_errors.append(normalized_error)
        overall = int(round(max(0.0, min(100.0, 100.0 * (1.0 - float(np.mean(normalized_errors)))))))
        return ReferenceMatch(
            True,
            overall,
            minimum,
            overall >= minimum,
            per_metric,
            str(ref["source"].get("type", "")),
            ref["version"],
            ref["animation_key"],
            len(self.user_frames),
            "Đã căn chỉnh DTW và so khớp chuỗi người tập với animation 3D chuẩn.",
        )
