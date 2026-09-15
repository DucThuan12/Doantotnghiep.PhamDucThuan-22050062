"""Trainable pose-phase classifier and runtime guard for FitMotion AI.

The benchmark thesis converts body landmarks into a feature vector and learns
exercise states from labelled examples. FitMotion keeps its temporal/state-
machine processors as the primary counter, then uses this optional learned
classifier as a second opinion for exercise identity, phase and left/right curl.

The model is deliberately optional: without a trained checkpoint the realtime
pipeline continues with the explicit geometric rules and never invents an AI
prediction.
"""
from __future__ import annotations

from collections import Counter, deque
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

FEATURE_VERSION = "yolo17-normalized-xyc-v1"
DEFAULT_MODEL_PATH = str(Path("models") / "pose_phase_classifier.pth")

# Recommended labels for an explainable training set. Additional labels are
# supported as long as they follow the same exercise prefix convention.
RECOMMENDED_LABELS = (
    "squat_standing",
    "squat_descending",
    "squat_bottom",
    "squat_ascending",
    "pushup_up",
    "pushup_lowering",
    "pushup_bottom",
    "pushup_rising",
    "curl_left_extended",
    "curl_left_lifting",
    "curl_left_flexed",
    "curl_left_lowering",
    "curl_right_extended",
    "curl_right_lifting",
    "curl_right_flexed",
    "curl_right_lowering",
    "other",
)


def _as_array(points: Sequence, size: int = 17) -> np.ndarray:
    arr = np.asarray(points, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[0] < size or arr.shape[1] < 2:
        raise ValueError("Cần tối thiểu 17 keypoint XY để tạo đặc trưng pose.")
    return arr[:size, :2].copy()


def _as_confidence(conf: Sequence, size: int = 17) -> np.ndarray:
    arr = np.asarray(conf, dtype=np.float32).reshape(-1)
    if arr.size < size:
        raise ValueError("Cần confidence cho đủ 17 keypoint YOLO Pose.")
    return np.clip(arr[:size], 0.0, 1.0)


def normalize_yolo_pose(points: Sequence, conf: Sequence) -> np.ndarray:
    """Return a translation/scale-normalized 17×(x,y,confidence) feature.

    Coordinates are centered around the mean of visible shoulders/hips and
    scaled by torso size. This makes the classifier less sensitive to where the
    person stands in the camera or how far they are from it.
    """
    xy = _as_array(points)
    confidence = _as_confidence(conf)
    anchor_indices = [5, 6, 11, 12]
    visible = [idx for idx in anchor_indices if confidence[idx] >= 0.20]
    if len(visible) >= 2:
        center = np.mean(xy[visible], axis=0)
    else:
        center = np.mean(xy, axis=0)

    shoulder_center = np.mean(xy[[5, 6]], axis=0)
    hip_center = np.mean(xy[[11, 12]], axis=0)
    torso = float(np.linalg.norm(shoulder_center - hip_center))
    shoulder_width = float(np.linalg.norm(xy[5] - xy[6]))
    hip_width = float(np.linalg.norm(xy[11] - xy[12]))
    scale = max(torso, shoulder_width, hip_width, 1.0)

    normalized_xy = (xy - center) / scale
    feature = np.concatenate([normalized_xy, confidence[:, None]], axis=1)
    return feature.astype(np.float32).reshape(-1)


def label_family(label: str) -> tuple[str, str | None]:
    value = str(label or "").strip().lower().replace("-", "_")
    if value.startswith("curl_left_"):
        return "curl", "left"
    if value.startswith("curl_right_"):
        return "curl", "right"
    if value.startswith("squat_"):
        return "squat", None
    if value.startswith("pushup_") or value.startswith("push_up_"):
        return "pushup", None
    return "other", None


def expected_label_prefixes(exercise: str, side: str | None = None) -> tuple[str, ...]:
    exercise = str(exercise or "").strip().lower().replace("-", "_")
    side = str(side or "").strip().lower() or None
    if exercise.startswith("curl"):
        if side == "right" or "right" in exercise:
            return ("curl_right_",)
        if side == "left" or "left" in exercise:
            return ("curl_left_",)
        return ("curl_left_", "curl_right_")
    if exercise == "pushup" or exercise == "push_up":
        return ("pushup_", "push_up_")
    if exercise == "squat":
        return ("squat_",)
    return (exercise + "_",) if exercise else tuple()


def build_mlp(input_dim: int, class_count: int):
    import torch.nn as nn

    return nn.Sequential(
        nn.Linear(int(input_dim), 128),
        nn.ReLU(),
        nn.Dropout(0.15),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Dropout(0.10),
        nn.Linear(64, int(class_count)),
    )


@dataclass(frozen=True)
class GuardDecision:
    available: bool
    accepted: bool
    reason_code: str = ""
    message: str = ""
    expected_ratio: float = 0.0
    opposite_ratio: float = 0.0
    prediction_count: int = 0
    dominant_label: str = ""
    model_sha256: str = ""
    test_accuracy: float = 0.0

    def to_dict(self) -> dict:
        return {
            "available": self.available,
            "accepted": self.accepted,
            "reason_code": self.reason_code,
            "message": self.message,
            "expected_ratio": round(float(self.expected_ratio), 4),
            "opposite_ratio": round(float(self.opposite_ratio), 4),
            "prediction_count": int(self.prediction_count),
            "dominant_label": self.dominant_label,
            "model_sha256": self.model_sha256,
            "test_accuracy": round(float(self.test_accuracy), 4),
        }


class HybridPoseRuntimeGuard:
    """Optional learned second-opinion guard for one realtime processor."""

    def __init__(
        self,
        model_path: str | None = None,
        min_confidence: float = 0.50,
        min_predictions: int = 4,
        expected_ratio: float = 0.55,
        min_test_accuracy: float = 0.70,
    ):
        self.model_path = str(model_path or DEFAULT_MODEL_PATH)
        self.min_confidence = float(min_confidence)
        self.min_predictions = int(min_predictions)
        self.expected_ratio = float(expected_ratio)
        self.min_test_accuracy = float(min_test_accuracy)
        self.model = None
        self.labels: list[str] = []
        self.input_dim = 51
        self.load_error = ""
        self.model_sha256 = ""
        self.test_accuracy = 0.0
        self.session_predictions: deque[tuple[str, float]] = deque(maxlen=60)
        self.rep_predictions: list[tuple[str, float]] = []
        self._load()

    @property
    def available(self) -> bool:
        return self.model is not None and bool(self.labels)

    def _load(self) -> None:
        path = Path(self.model_path)
        if not path.is_file():
            self.load_error = "Chưa có checkpoint classifier đã train."
            return
        try:
            import torch

            checkpoint = torch.load(str(path), map_location="cpu", weights_only=False)
            labels = list(checkpoint.get("labels", []))
            input_dim = int(checkpoint.get("input_dim", 51))
            feature_version = str(checkpoint.get("feature_version", FEATURE_VERSION) or FEATURE_VERSION)
            if feature_version != FEATURE_VERSION:
                raise ValueError(f"Checkpoint dùng feature version {feature_version}, cần {FEATURE_VERSION}.")
            test_metrics = (checkpoint.get("metrics") or {}).get("test") or {}
            test_accuracy = test_metrics.get("accuracy")
            test_samples = int(test_metrics.get("samples", 0) or 0)
            if test_accuracy is None or test_samples <= 0:
                raise ValueError("Checkpoint chưa có test set độc lập; classifier không được dùng realtime.")
            if float(test_accuracy) < self.min_test_accuracy:
                raise ValueError(
                    f"Test accuracy {float(test_accuracy):.3f} thấp hơn cổng {self.min_test_accuracy:.3f}; classifier bị vô hiệu hóa."
                )
            if not labels:
                raise ValueError("Checkpoint không có danh sách label.")
            model = build_mlp(input_dim, len(labels))
            model.load_state_dict(checkpoint["model_state"])
            model.eval()
            self.model = model
            self.labels = [str(item) for item in labels]
            self.input_dim = input_dim
            self.model_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
            self.test_accuracy = float(test_accuracy)
            self.load_error = ""
        except Exception as exc:  # checkpoint must never break webcam fallback
            self.model = None
            self.labels = []
            self.model_sha256 = ""
            self.test_accuracy = 0.0
            self.load_error = str(exc)

    def start_rep(self) -> None:
        self.rep_predictions = []

    def observe(self, points: Sequence, conf: Sequence) -> dict | None:
        if not self.available:
            return None
        try:
            import torch

            feature = normalize_yolo_pose(points, conf)
            tensor = torch.from_numpy(feature).float().unsqueeze(0)
            with torch.no_grad():
                logits = self.model(tensor)
                probability = torch.softmax(logits, dim=1)[0]
                index = int(torch.argmax(probability).item())
                confidence = float(probability[index].item())
            label = self.labels[index]
            if confidence >= self.min_confidence:
                item = (label, confidence)
                self.session_predictions.append(item)
                self.rep_predictions.append(item)
            return {"label": label, "confidence": round(confidence, 4)}
        except Exception as exc:
            self.load_error = str(exc)
            return None

    def validate_rep(self, expected_exercise: str, expected_side: str | None = None) -> GuardDecision:
        if not self.available:
            return GuardDecision(False, True, message=self.load_error, model_sha256=self.model_sha256, test_accuracy=self.test_accuracy)
        predictions = [label for label, confidence in self.rep_predictions if confidence >= self.min_confidence]
        if len(predictions) < self.min_predictions:
            return GuardDecision(
                True, True, message="Chưa đủ frame classifier để phủ quyết luật hình học.",
                prediction_count=len(predictions),
                model_sha256=self.model_sha256,
                test_accuracy=self.test_accuracy,
            )

        counter = Counter(predictions)
        dominant_label, _ = counter.most_common(1)[0]
        prefixes = expected_label_prefixes(expected_exercise, expected_side)
        expected_count = sum(count for label, count in counter.items() if label.startswith(prefixes))
        total = max(1, len(predictions))
        expected_ratio = expected_count / total

        opposite_ratio = 0.0
        if str(expected_exercise).startswith("curl"):
            expected_side = str(expected_side or "").lower()
            opposite_prefix = "curl_right_" if expected_side == "left" else "curl_left_"
            opposite_count = sum(count for label, count in counter.items() if label.startswith(opposite_prefix))
            opposite_ratio = opposite_count / total
            if opposite_ratio >= 0.40 and opposite_ratio > expected_ratio:
                return GuardDecision(
                    True,
                    False,
                    "wrong_side_ml",
                    "Mô hình học từ dữ liệu nhận thấy tay đối diện mới là tay đang thực hiện động tác.",
                    expected_ratio,
                    opposite_ratio,
                    total,
                    dominant_label,
                    self.model_sha256,
                    self.test_accuracy,
                )

        if expected_ratio < self.expected_ratio:
            return GuardDecision(
                True,
                False,
                "exercise_mismatch_ml",
                "Mô hình học từ dữ liệu chưa xác nhận đúng bài/pha động tác đã chọn.",
                expected_ratio,
                opposite_ratio,
                total,
                dominant_label,
                self.model_sha256,
                self.test_accuracy,
            )

        return GuardDecision(
            True,
            True,
            expected_ratio=expected_ratio,
            opposite_ratio=opposite_ratio,
            prediction_count=total,
            dominant_label=dominant_label,
            model_sha256=self.model_sha256,
            test_accuracy=self.test_accuracy,
        )

class BilateralCurlGuard:
    """Geometry-only side guard that works even before the ML model is trained.

    It compares elbow range of motion on both arms during a one-arm curl. If
    the opposite arm clearly performs more of the curl than the selected arm,
    the repetition is rejected as the wrong side.
    """

    def __init__(self, selected_side: str, min_opposite_rom: float = 55.0, dominance_margin: float = 20.0):
        self.selected_side = "right" if str(selected_side).lower() == "right" else "left"
        self.min_opposite_rom = float(min_opposite_rom)
        self.dominance_margin = float(dominance_margin)
        self.reset()

    def reset(self):
        self.active_min = 999.0
        self.active_max = 0.0
        self.opposite_min = 999.0
        self.opposite_max = 0.0
        self.samples = 0

    def observe(self, active_angle: float, opposite_angle: float | None = None):
        self.active_min = min(self.active_min, float(active_angle))
        self.active_max = max(self.active_max, float(active_angle))
        if opposite_angle is not None:
            self.opposite_min = min(self.opposite_min, float(opposite_angle))
            self.opposite_max = max(self.opposite_max, float(opposite_angle))
        self.samples += 1

    @property
    def active_rom(self) -> float:
        return max(0.0, self.active_max - self.active_min) if self.samples else 0.0

    @property
    def opposite_rom(self) -> float:
        if self.opposite_min >= 998.0:
            return 0.0
        return max(0.0, self.opposite_max - self.opposite_min)

    def wrong_side(self) -> bool:
        # Only flag as wrong side when the opposite arm clearly performed a full curl
        # (opposite_rom >= min_opposite_rom = 55 deg) AND the selected arm barely moved
        # (active_rom < 25 deg). If the selected arm has meaningful motion, the user
        # is exercising on the correct side and should receive form feedback instead.
        return (
            self.opposite_rom >= self.min_opposite_rom
            and self.active_rom < 25.0
            and self.opposite_rom >= self.active_rom + self.dominance_margin
        )

    def evidence(self) -> dict:
        return {
            "selected_side": self.selected_side,
            "active_rom": round(self.active_rom, 2),
            "opposite_rom": round(self.opposite_rom, 2),
            "samples": int(self.samples),
            "wrong_side": bool(self.wrong_side()),
        }
