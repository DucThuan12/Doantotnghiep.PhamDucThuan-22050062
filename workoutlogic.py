"""Real-time exercise analysis for FitMotion AI.

Week-6 upgrade scope:
- confidence-aware keypoint validation and stable side selection;
- EMA temporal smoothing and frame-based hysteresis;
- multi-phase state machines for squat, push-up and bicep curl;
- explainable 0-100 quality scoring per repetition;
- session metrics for persistence and later experimental evaluation.
"""
from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass
from typing import Iterable, Sequence

import time

import cv2
import numpy as np

from ai_temporal import TemporalBank, movement_stability
from config import (
    ERRORCOOLDOWN,
    KEYPOINTTHRESH,
    MODELPATH,
    SHOWFPS,
    WARNINGCOOLDOWN,
    WARNINGSHOWSEC,
)
from emergency import EmergencyMonitor
from expert_system import (
    build_violation,
    normalize_error_candidates,
    select_primary_violation,
    serialize_violations,
)
from helper import (
    FPSCounter,
    ResultManager,
    WarningManager,
    coNguoi,
    layDiem,
    save_bad_rep,
    thayKhau,
    tinhgoc,
    vietchu,
)
from message import COMMONMSG, CURLMSG, PUSHUPMSG, SQUATMSG
from pose_model import get_pose_model, run_pose_inference
from pose_learning import HybridPoseRuntimeGuard, BilateralCurlGuard
from reference_motion import ReferenceMotionMatcher
from quality_scoring import RepScore, score_curl, score_pushup, score_squat
from training_levels import (
    DEFAULT_LEVEL,
    normalize_training_level,
    threshold_profile,
    training_level_label,
)


REFERENCE_BLEND_BY_LEVEL = {"easy": 0.10, "medium": 0.20, "hard": 0.30}
REFERENCE_MIN_DELTA_BY_LEVEL = {"easy": -15.0, "medium": 0.0, "hard": 10.0}


VOICE_MESSAGES = {
    "squat:good": "Động tác gập chân đạt. Hãy giữ nhịp độ ổn định.",
    "squat:notlow": "Hãy hạ hông thấp hơn để gập chân đủ biên độ.",
    "squat:backlean": "Giữ ngực mở và lưng thẳng hơn.",
    "pushup:good": "Động tác hít đất đạt. Hãy tiếp tục giữ thân người ổn định.",
    "pushup:notlow": "Hãy hạ người thấp hơn trước khi đẩy lên.",
    "pushup:bodyline": "Giữ vai, hông và cổ chân trên một đường thẳng.",
    "curl:good": "Động tác cuốn tạ đạt. Hãy giữ khuỷu tay cố định.",
    "curl:notbend": "Hãy gập tay thêm để đạt đủ biên độ.",
    "curl:notstraight": "Hãy hạ tạ về gần tư thế bắt đầu đã hiệu chuẩn, không cần khóa cứng khuỷu tay.",
    "curl:elbowshift": "Giữ khuỷu tay gần thân, không đưa khuỷu ra trước.",
    "curl-left:good": "Động tác cuốn tạ tay trái đạt. Hãy giữ khuỷu tay trái cố định.",
    "curl-left:notbend": "Tay trái chưa gập đủ. Hãy cuốn tạ tay trái lên thêm.",
    "curl-left:notstraight": "Hãy hạ tay trái về gần tư thế bắt đầu đã hiệu chuẩn.",
    "curl-left:elbowshift": "Giữ khuỷu tay trái gần thân, không đưa khuỷu ra trước.",
    "curl-right:good": "Động tác cuốn tạ tay phải đạt. Hãy giữ khuỷu tay phải cố định.",
    "curl-right:notbend": "Tay phải chưa gập đủ. Hãy cuốn tạ tay phải lên thêm.",
    "curl-right:notstraight": "Hãy hạ tay phải về gần tư thế bắt đầu đã hiệu chuẩn.",
    "curl-right:elbowshift": "Giữ khuỷu tay phải gần thân, không đưa khuỷu ra trước.",
    "qualitylow": "Chuyển động chưa ổn định. Hãy giảm tốc độ và kiểm soát biên độ.",
    "tracking_lost": "Mất điểm khớp trong khi tập. Hãy trở lại khung hình và thực hiện lại.",
    "exercise_mismatch": "Hệ thống chưa xác nhận đúng bài tập đang thực hiện. Hãy thực hiện lại từ đầu.",
    "wrongside": "Phát hiện sai tư thế bài tập. Hãy điều chỉnh và thực hiện lại.",
}


@dataclass(frozen=True)
class SquatThresholds:
    # Demo-safe defaults: a complete down-up cycle is still mandatory, but
    # the quality gate accepts a comfortable parallel-ish squat instead of
    # requiring competition-level depth.
    calibration_min_angle: float = 130.0
    calibration_frames: int = 5
    start_drop_from_baseline: float = 10.0
    finish_tolerance: float = 20.0
    minimum_cycle_rom: float = 16.0
    bottom_angle: float = 128.0
    shallow_limit: float = 132.0
    torso_limit: float = 122.0
    good_score_threshold: int = 60


@dataclass(frozen=True)
class PushupThresholds:
    # Push-up counting is based on the user's calibrated top position and a
    # verified down-up cycle. Body-line quality affects feedback, not whether
    # an otherwise complete repetition is counted.
    calibration_min_angle: float = 112.0
    calibration_frames: int = 4
    start_drop_from_baseline: float = 8.0
    finish_tolerance: float = 22.0
    minimum_cycle_rom: float = 14.0
    bottom_angle: float = 128.0
    shallow_limit: float = 120.0
    body_line_limit: float = 135.0
    good_score_threshold: int = 60


@dataclass(frozen=True)
class CurlThresholds:
    # A curl starts and ends near the user's comfortable elbow angle. NASM
    # guidance recommends lowering under control and stopping just short of a
    # hard lockout, so the effective baseline is capped below 180 degrees.
    calibration_min_angle: float = 118.0
    calibration_max_angle: float = 165.0
    calibration_frames: int = 5
    start_drop_from_baseline: float = 12.0
    minimum_range_of_motion: float = 30.0
    full_flexion_ceiling: float = 112.0
    finish_tolerance: float = 18.0
    incomplete_extension_tolerance: float = 28.0
    elbow_shift_limit: float = 48.0
    good_score_threshold: int = 64


def _required_visible(conf: Sequence[float], joints: Iterable[int], threshold: float) -> bool:
    return all(thayKhau(conf, int(index), threshold) for index in joints)


def _side_confidence(conf: Sequence[float], joints: Iterable[int]) -> float:
    values = [float(conf[int(index)]) for index in joints]
    return float(np.mean(values)) if values else 0.0


def _choose_side(conf, left_joints, right_joints, threshold):
    """Choose the fully visible side with the higher mean keypoint confidence."""
    left_ok = _required_visible(conf, left_joints, threshold)
    right_ok = _required_visible(conf, right_joints, threshold)

    if left_ok and right_ok:
        return "left" if _side_confidence(conf, left_joints) >= _side_confidence(conf, right_joints) else "right"
    if left_ok:
        return "left"
    if right_ok:
        return "right"
    return None


def _midpoint(kp, first: int, second: int) -> np.ndarray:
    return (np.asarray(kp[first], dtype=np.float32) + np.asarray(kp[second], dtype=np.float32)) / 2.0


def _body_axis_from_horizontal(kp, conf, threshold: float = KEYPOINTTHRESH) -> float | None:
    """Return torso-axis angle in image space: 0=horizontal, 90=upright.

    This is intentionally a coarse exercise-identity guard, not a quality score.
    It prevents an upright squat/curl from being counted by the push-up state
    machine (and vice versa) when no trained exercise classifier is available.
    """
    joints = (5, 6, 11, 12)
    if not _required_visible(conf, joints, threshold):
        return None
    shoulder_mid = _midpoint(kp, 5, 6)
    hip_mid = _midpoint(kp, 11, 12)
    dx = float(hip_mid[0] - shoulder_mid[0])
    dy = float(hip_mid[1] - shoulder_mid[1])
    if abs(dx) + abs(dy) < 1e-6:
        return None
    return float(np.degrees(np.arctan2(abs(dy), abs(dx) + 1e-6)))


def _average_knee_angle(kp, conf, threshold: float = KEYPOINTTHRESH) -> float | None:
    """Average visible knee angle, used to reject squat-like motion in curl."""
    values = []
    for hip_idx, knee_idx, ankle_idx in ((11, 13, 15), (12, 14, 16)):
        if _required_visible(conf, (hip_idx, knee_idx, ankle_idx), threshold):
            values.append(tinhgoc(kp[hip_idx], kp[knee_idx], kp[ankle_idx]))
    return float(np.mean(values)) if values else None


class BaseProcessor:
    """Shared temporal state, session metrics and feedback publishing."""

    GOOD_SCORE_THRESHOLD = 70
    MAX_MISSING_FRAMES = 8
    MAX_REP_FRAMES = 300
    MIN_REP_FRAMES = 5

    def __init__(self, shared_state=None, model=None, load_model: bool = True):
        self.model = model if model is not None else (get_pose_model(MODELPATH) if load_model else None)
        self.fpscounter = FPSCounter()
        self.warning = WarningManager(WARNINGCOOLDOWN, WARNINGSHOWSEC)
        self.resultbox = ResultManager()
        self.lastsave = 0

        self.tongsolan = 0
        self.solandung = 0
        self.trangthai = "READY"

        self.temporal = TemporalBank(alpha=0.35, window_size=10)
        self.rep_scores: list[int] = []
        self.rep_records: list[dict] = []
        self.error_code_counts: Counter[str] = Counter()
        self.session_confidences: list[float] = []
        self.rep_confidences: list[float] = []
        self.rep_primary_values: list[float] = []
        self.rep_frame_count = 0
        self.missing_frame_count = 0
        self.admin_rule_errors: list[tuple] = []

        self.last_rep_score = 0
        self.last_feedback = "Chưa có rep hoàn chỉnh."
        self.last_advice = "Đứng vào khung hình và thực hiện động tác có kiểm soát."
        self.feedback_id = 0

        self.shared_state = shared_state if shared_state is not None else {}
        self.training_level = normalize_training_level(
            self.shared_state.get("training_level", DEFAULT_LEVEL)
        )
        self.shared_state["training_level"] = self.training_level
        self.shared_state["training_level_label"] = training_level_label(self.training_level)
        # Resume completed repetitions from the durable SQLite draft. Only
        # aggregate/rep-level evidence is restored; an unfinished movement
        # phase is intentionally discarded for safety.
        self.tongsolan = max(0, int(self.shared_state.get("total_rep", 0) or 0))
        self.solandung = min(
            self.tongsolan,
            max(0, int(self.shared_state.get("good_rep", 0) or 0)),
        )
        self.rep_scores = [int(value) for value in (self.shared_state.get("rep_scores", []) or [])]
        self.rep_records = list(self.shared_state.get("rep_records", []) or [])[-100:]
        self.error_code_counts = Counter(self.shared_state.get("error_code_counts", {}) or {})
        self.last_rep_score = int(self.shared_state.get("rep_quality_score", 0) or 0)
        self.feedback_id = int(self.shared_state.get("feedback_id", 0) or 0)
        restored_confidences = [
            float(item.get("confidence", 0) or 0)
            for item in self.rep_records
            if isinstance(item, dict) and float(item.get("confidence", 0) or 0) > 0
        ]
        if restored_confidences:
            self.session_confidences = restored_confidences
        self.emergency_monitor = EmergencyMonitor(self.shared_state)
        self.hybrid_guard = HybridPoseRuntimeGuard(
            model_path=self.shared_state.get("pose_classifier_model_path") or None
        )
        self.shared_state["pose_classifier_available"] = bool(self.hybrid_guard.available)
        self.shared_state["pose_classifier_error"] = str(self.hybrid_guard.load_error or "")
        self.shared_state.setdefault("pose_classifier_last", {})
        self.shared_state.setdefault("pose_guard_last_decision", {})
        self.reference_matcher = ReferenceMotionMatcher(
            reference_path=self.shared_state.get("reference_motion_file") or None,
            expected_exercise=self.shared_state.get("exercise_slug") or None,
        )
        self.shared_state["reference_motion_available"] = bool(self.reference_matcher.available)
        self.shared_state["reference_motion_error"] = str(self.reference_matcher.load_error or "")
        self.shared_state.setdefault("reference_match", {})

    def _on_training_level_changed(self) -> None:
        """Hook for exercise processors to reload thresholds safely."""

    def sync_training_level(self) -> bool:
        requested = normalize_training_level(self.shared_state.get("training_level", self.training_level))
        if requested == self.training_level:
            return False
        # The Flask API blocks level changes once a rep exists. This extra guard
        # protects non-Flask callers and stale camera streams from mixing two
        # rule sets in one recorded session.
        if self.tongsolan > 0:
            self.shared_state["training_level"] = self.training_level
            self.shared_state["training_level_label"] = training_level_label(self.training_level)
            return False
        self.training_level = requested
        self.shared_state["training_level"] = requested
        self.shared_state["training_level_label"] = training_level_label(requested)
        self._on_training_level_changed()
        return True

    def infer(self, frame):
        if self.model is None:
            return []
        return run_pose_inference(self.model, frame)

    def update_emergency(self, kp, conf, raw_frame, annotated, exercise_key="", phase=""):
        self.emergency_monitor.update(
            kp,
            conf,
            annotated.shape,
            raw_frame,
            context={"exercise": str(exercise_key), "phase": str(phase)},
        )

    def begin_rep(self) -> None:
        self.rep_confidences = []
        self.rep_primary_values = []
        self.rep_frame_count = 0
        self.missing_frame_count = 0
        self.admin_rule_errors = []
        self.hybrid_guard.start_rep()
        self.reference_matcher.start_rep()

    def observe_reference_metrics(self, metrics: dict) -> None:
        self.reference_matcher.observe(metrics)

    def observe_pose_guard(self, kp, conf) -> dict | None:
        prediction = self.hybrid_guard.observe(kp, conf)
        if prediction is not None:
            self.shared_state["pose_classifier_last"] = dict(prediction)
        self.shared_state["pose_classifier_available"] = bool(self.hybrid_guard.available)
        self.shared_state["pose_classifier_error"] = str(self.hybrid_guard.load_error or "")
        return prediction

    def posture_identity_gate(self, expected: str, kp, conf, *, strict_unknown: bool = False) -> bool:
        """Coarse geometry gate used before a core state machine can advance.

        ``pushup`` requires a horizontal torso; ``squat``/``curl`` require an
        upright torso. The learned classifier remains the stronger second
        opinion at rep completion, but this fallback blocks obvious cross-
        exercise false reps even when no checkpoint has been trained yet.

        The web runtime explicitly enables this gate in ``shared_state``.
        Keeping it opt-in at processor level preserves deterministic unit tests
        and offline utilities that use synthetic keypoints rather than a real
        camera skeleton.
        """
        if not bool(self.shared_state.get("exercise_identity_gate_enabled", False)):
            self.shared_state["exercise_identity_match"] = True
            self.shared_state["exercise_identity_reason"] = "gate_disabled"
            return True

        angle = _body_axis_from_horizontal(kp, conf)
        self.shared_state["body_axis_from_horizontal"] = round(float(angle or 0.0), 1) if angle is not None else 0.0
        if angle is None:
            self.shared_state["exercise_identity_match"] = not strict_unknown
            self.shared_state["exercise_identity_reason"] = "missing_torso_keypoints"
            return not strict_unknown

        expected = str(expected or "").lower()
        if expected == "pushup":
            raw_match = angle <= 45.0
            message = "Hay vao tu the plank nam ngang truoc khi bat dau hit dat."
        elif expected == "squat":
            raw_match = angle >= 25.0
            message = "Dong tac dang chon can tu the than nguoi dung; hay thuc hien dung bai tap."
        else:
            raw_match = angle >= 45.0
            message = "Dong tac dang chon can tu the than nguoi dung; hay thuc hien dung bai tap."

        stable_match = self.temporal.stable(f"identity:{expected}:match", raw_match, 2)
        stable_mismatch = self.temporal.stable(f"identity:{expected}:mismatch", not raw_match, 3)
        self.shared_state["exercise_identity_match"] = bool(stable_match)
        self.shared_state["exercise_identity_reason"] = "" if stable_match else ("posture_mismatch" if stable_mismatch else "checking")
        if stable_mismatch:
            self.resultbox.set("CHUA DUNG BAI TAP", message)
        return bool(stable_match)

    def record_valid_frame(self, conf, joints: Iterable[int]) -> float:
        confidence = _side_confidence(conf, joints)
        self.session_confidences.append(confidence)
        self.missing_frame_count = 0
        return confidence

    def record_rep_frame(self, confidence: float, primary_value: float) -> None:
        self.rep_confidences.append(float(confidence))
        self.rep_primary_values.append(float(primary_value))
        self.rep_frame_count += 1

    def rep_confidence(self) -> float:
        return float(np.mean(self.rep_confidences)) if self.rep_confidences else 0.0

    def rep_stability(self) -> float:
        return movement_stability(self.rep_primary_values)

    def note_tracking_loss(self, is_rep_active: bool, reset_callback, ready_state: str) -> None:
        self.missing_frame_count += 1
        if is_rep_active and self.missing_frame_count >= self.MAX_MISSING_FRAMES:
            self.trangthai = ready_state
            reset_callback()
            # Old smoothed angles must not start a phantom repetition when the
            # person returns to the frame after tracking was lost.
            self.temporal.reset_all()
            self.shared_state["tracking_abort_count"] = int(self.shared_state.get("tracking_abort_count", 0) or 0) + 1
            self.resultbox.set("MAT DIEM KHOP", "TRO LAI KHUNG HINH VA THUC HIEN LAI")
            self.publish_feedback(
                exercise_key="tracking",
                error_code="tracking_lost",
                result="Mất điểm khớp khi đang thực hiện động tác",
                advice="Trở lại khung hình và thực hiện lại từ tư thế bắt đầu.",
                score=0,
                is_good=False,
            )

    @staticmethod
    def _admin_rule_matches(value: float, operator: str, threshold: float) -> bool:
        operator = str(operator or "").strip()
        if operator == "<":
            return value < threshold
        if operator == "<=":
            return value <= threshold
        if operator == ">":
            return value > threshold
        if operator == ">=":
            return value >= threshold
        if operator == "=":
            return abs(value - threshold) <= 1.0
        return False

    def evaluate_admin_criteria(self, kp, conf, phase: str) -> list[tuple]:
        """Evaluate Admin-defined A-B-C angle rules for the current phase.

        The rule format mirrors the benchmark thesis: three pose keypoints, an
        operator, a threshold and a feedback message.  A true expression means
        the configured error is present.  Rules are skipped when any required
        keypoint is below the runtime confidence threshold.
        """
        violations: list[tuple] = []
        current_phase = str(phase or "middle").strip().lower()
        for rule in self.shared_state.get("admin_criteria_rules", []) or []:
            if not isinstance(rule, dict):
                continue
            rule_phase = str(rule.get("phase", "middle") or "middle").strip().lower()
            if rule_phase not in {"any", current_phase}:
                continue
            indices = rule.get("joint_indices", [])
            if not isinstance(indices, (list, tuple)) or len(indices) != 3:
                continue
            try:
                a, b, c = [int(item) for item in indices]
            except (TypeError, ValueError):
                continue
            if any(index < 0 or index >= len(kp) or index >= len(conf) for index in (a, b, c)):
                continue
            if not _required_visible(conf, (a, b, c), KEYPOINTTHRESH):
                continue
            try:
                measured = float(tinhgoc(kp[a], kp[b], kp[c]))
                threshold = float(rule.get("angle_value", 0.0) or 0.0)
            except (TypeError, ValueError, IndexError):
                continue
            if not self._admin_rule_matches(measured, rule.get("operator", ""), threshold):
                continue
            code = str(rule.get("error_code", "") or "admin_rule")
            title = str(rule.get("message_text", "") or "Sai kỹ thuật")
            advice = str(rule.get("advice_text", "") or "Điều chỉnh tư thế và thực hiện lại.")
            violations.append((code, title, advice, current_phase))
        return violations

    def capture_admin_criteria(self, kp, conf, phase: str) -> None:
        """Collect unique Admin rule violations during the current repetition."""
        existing = {str(item[0]) for item in self.admin_rule_errors if item}
        for violation in self.evaluate_admin_criteria(kp, conf, phase):
            code = str(violation[0])
            if code not in existing:
                self.admin_rule_errors.append(violation)
                existing.add(code)

    def publish_feedback(self, exercise_key, error_code, result, advice, score, is_good) -> None:
        self.last_rep_score = int(score)
        self.last_feedback = str(result)
        self.last_advice = str(advice)
        self.feedback_id += 1

        voice_exercise_key = str(exercise_key or "")
        if voice_exercise_key == "curl" and getattr(self, "side", None) in {"left", "right"}:
            voice_exercise_key = f"curl-{self.side}"
        voice_key = f"{voice_exercise_key}:{'good' if is_good else error_code}"
        voice_text = VOICE_MESSAGES.get(voice_key, VOICE_MESSAGES.get(error_code, result or advice))
        # Wrong-side messages for curl are built dynamically and name both arms specifically.
        # Only override the generic voice_text for curl exercises, not for squat/pushup.
        if str(error_code) == "wrongside" and str(advice).strip() and "curl" in str(exercise_key or "").lower():
            voice_text = str(advice)
        self.shared_state.update({
            "feedback_id": int(self.feedback_id),
            "feedback_text": str(result),
            "feedback_advice": str(advice),
            "feedback_level": "good" if is_good else "warning",
            "voice_text": str(voice_text),
            "last_error_code": "" if is_good else str(error_code),
        })

    def register_phase_errors(self, errors: list[tuple]) -> None:
        """Count every distinct rule violation, grouped by movement phase."""
        seen = set()
        for violation in errors:
            error_code = violation.code
            phase = violation.phase
            key = (str(error_code), str(phase))
            if key in seen:
                continue
            seen.add(key)
            phase_key = {
                "start": "phase_start_error",
                "middle": "phase_middle_error",
                "end": "phase_end_error",
            }.get(str(phase), "phase_middle_error")
            self.shared_state[phase_key] = int(self.shared_state.get(phase_key, 0) or 0) + 1
            self.error_code_counts[str(error_code)] += 1

    def reject_attempt(self, exercise_key: str, error_code: str, result: str, advice: str, evidence=None) -> None:
        """Reject a wrong exercise/side attempt without incrementing the rep counter.

        Teacher feedback explicitly requires curl-left not to count a right-arm
        curl. These attempts remain auditable, but they are not written into the
        completed rep sequence and therefore cannot inflate total_rep.
        """
        self.error_code_counts[str(error_code)] += 1
        self.shared_state["rejected_attempt_count"] = int(self.shared_state.get("rejected_attempt_count", 0) or 0) + 1
        self.shared_state["last_rejected_attempt"] = {
            "exercise": str(exercise_key),
            "error_code": str(error_code),
            "result": str(result),
            "advice": str(advice),
            "evidence": dict(evidence or {}),
            "training_level": self.training_level,
        }
        self.resultbox.set(result, advice)
        self.publish_feedback(exercise_key, error_code, result, advice, 0, False)

    def finish_rep(
        self,
        exercise_key: str,
        score: RepScore,
        errors: list[tuple],
        good_message: dict,
        frame,
        metric_lines: list[str],
        good_score_threshold: int | None = None,
    ) -> bool:
        """Finalize one attempted repetition and publish an auditable record."""
        expected_side = getattr(self, "side", None) if str(exercise_key).startswith("curl") else None
        guard_decision = self.hybrid_guard.validate_rep(exercise_key, expected_side=expected_side)
        self.shared_state["pose_guard_last_decision"] = guard_decision.to_dict()
        if guard_decision.available and not guard_decision.accepted:
            if guard_decision.reason_code == "wrong_side_ml":
                code = "wrongside"
                result = "SAI BEN TAY"
                advice = "Hãy tập đúng bên tay đã chọn. Classifier ảnh/video xác nhận tay đối diện đang thực hiện động tác."
            else:
                code = "exercise_mismatch"
                result = "CHUA XAC NHAN DUNG BAI TAP"
                advice = "Classifier ảnh/video chưa đồng thuận với bài/pha đang chọn. Hãy thực hiện lại từ đầu."
            self.reject_attempt(exercise_key, code, result, advice, guard_decision.to_dict())
            return False

        self.tongsolan += 1
        reference_base_min = 55.0
        if self.reference_matcher.available:
            reference_base_min = float(self.reference_matcher.reference.get("min_score", 55.0) or 55.0)
        reference_min = max(0.0, min(100.0, reference_base_min + REFERENCE_MIN_DELTA_BY_LEVEL.get(self.training_level, 0.0)))
        reference_match = self.reference_matcher.finish_rep(min_score_override=reference_min)
        if reference_match.available:
            reference_weight = float(REFERENCE_BLEND_BY_LEVEL.get(self.training_level, 0.20))
            rule_weight = 1.0 - reference_weight
            combined = int(round((float(score.total) * rule_weight) + (float(reference_match.score) * reference_weight)))
            score = RepScore(
                total=combined,
                components={**dict(score.components), "rule_score": int(score.total), "reference_match": int(reference_match.score)},
                weights={"rule_score": round(rule_weight, 2), "reference_match": round(reference_weight, 2)},
            )
            self.shared_state["reference_match"] = reference_match.to_dict()
            if not reference_match.passed:
                errors.append((
                    "reference_mismatch",
                    "CHUA KHOP DONG TAC MAU 3D",
                    "Chuoi goc khop cua rep nay chua khop voi animation Mixamo tham chieu. Hay dieu chinh nhip va bien do.",
                    "middle",
                ))
        else:
            self.shared_state["reference_match"] = reference_match.to_dict()
        self.shared_state["reference_motion_available"] = bool(self.reference_matcher.available)
        self.shared_state["reference_motion_error"] = str(self.reference_matcher.load_error or "")

        score_value = int(score.total)
        threshold = int(self.GOOD_SCORE_THRESHOLD if good_score_threshold is None else good_score_threshold)

        combined_errors = list(errors or []) + list(self.admin_rule_errors or [])
        effective_errors = normalize_error_candidates(exercise_key, combined_errors)
        if not effective_errors and score_value < threshold:
            effective_errors = [build_violation(exercise_key, "qualitylow")]

        is_good = not effective_errors and score_value >= threshold
        if is_good:
            self.solandung += 1
            result = good_message["result"]
            advice = good_message["advice"]
            primary_error_code = "good"
        else:
            self.register_phase_errors(effective_errors)
            primary = select_primary_violation(effective_errors)
            # Admin-entered expert rules should actually drive the spoken
            # feedback, matching the benchmark workflow. Safety/identity errors
            # still take precedence; otherwise speak the first Admin rule hit
            # during the rep so its generated WAV is the one played.
            critical_codes = {"wrongside", "exercise_mismatch", "tracking_lost"}
            admin_codes = [str(item[0]) for item in (self.admin_rule_errors or []) if item]
            if primary is not None and primary.code not in critical_codes and admin_codes:
                admin_primary = next((item for code in admin_codes for item in effective_errors if item.code == code), None)
                if admin_primary is not None:
                    primary = admin_primary
            primary_error_code = primary.code
            result = primary.title
            advice = primary.advice
            if frame is not None:
                details = list(metric_lines)
                details.append(f"DIEM CHAT LUONG: {score_value}/100")
                details.extend(f"{name.upper()}: {value}/100" for name, value in score.components.items())
                self.lastsave = save_bad_rep(
                    frame,
                    exercise_key,
                    primary_error_code,
                    result,
                    advice,
                    details,
                    self.lastsave,
                    ERRORCOOLDOWN,
                )

        self.rep_scores.append(score_value)
        record = {
            "index": int(self.tongsolan),
            "score": score_value,
            "is_good": bool(is_good),
            "confidence": round(self.rep_confidence(), 4),
            "stability": round(self.rep_stability(), 2),
            "components": dict(score.components),
            "primary_error_code": "" if is_good else str(primary_error_code),
            "error_codes": [item.code for item in effective_errors],
            "errors": serialize_violations(effective_errors),
            "training_level": self.training_level,
            "training_level_label": training_level_label(self.training_level),
            "good_score_threshold": threshold,
            "rule_version": "hybrid-v2",
            "pose_guard": guard_decision.to_dict(),
            "reference_match": reference_match.to_dict(),
        }
        self.rep_records.append(record)
        self.rep_records = self.rep_records[-100:]
        self.shared_state["last_rep_at"] = time.time()
        self.shared_state["idle_seconds"] = 0

        self.resultbox.set(result, f"{advice} | DIEM {score_value}/100")
        self.publish_feedback(exercise_key, primary_error_code, result, advice, score_value, is_good)
        self.shared_state["last_rep_components"] = dict(score.components)
        return is_good

    def draw_emergency_overlay(self, frame):
        if not self.shared_state.get("active", False):
            return frame

        overlay = frame.copy()
        _, width = frame.shape[:2]
        cv2.rectangle(overlay, (0, 0), (width, 130), (0, 0, 255), -1)
        cv2.addWeighted(overlay, 0.28, frame, 0.72, 0, frame)
        vietchu(frame, "CANH BAO KHAN CAP", 30, 45, (255, 255, 255), 1.1, 3)
        vietchu(frame, self.shared_state.get("reason", "Phat hien su co bat thuong."), 30, 85, (255, 255, 255), 0.75, 2)
        vietchu(frame, "Bai tap da tam dung. Kiem tra nguoi tap ngay lap tuc.", 30, 120, (255, 255, 0), 0.7, 2)
        return frame

    def finish(self, annotated):
        if self.shared_state.get("active", False):
            self.resultbox.set("BAI TAP DA TAM DUNG", "He thong dang o che do canh bao khan cap")

        good_rep = min(int(self.tongsolan), max(0, int(self.solandung)))
        bad_rep = max(0, int(self.tongsolan) - good_rep)
        quality_average = round(float(np.mean(self.rep_scores)), 2) if self.rep_scores else 0.0
        confidence_average = round(float(np.mean(self.session_confidences)), 4) if self.session_confidences else 0.0

        height, width = annotated.shape[:2]
        panel_x = max(30, width - 350)
        overlay = annotated.copy()
        cv2.rectangle(overlay, (panel_x - 15, 20), (width - 20, min(height - 20, 190)), (10, 18, 32), -1)
        cv2.addWeighted(overlay, 0.70, annotated, 0.30, 0, annotated)
        vietchu(annotated, f"TONG REP: {self.tongsolan}", panel_x, 55, (255, 255, 255), 0.72, 2)
        vietchu(annotated, f"REP DAT: {good_rep}", panel_x, 90, (0, 255, 0), 0.72, 2)
        vietchu(annotated, f"DIEM REP: {self.last_rep_score}/100", panel_x, 125, (0, 255, 255), 0.72, 2)
        vietchu(annotated, f"DIEM TB: {int(round(quality_average))}/100", panel_x, 160, (255, 255, 0), 0.72, 2)
        vietchu(annotated, f"CAP DO: {training_level_label(self.training_level).upper()}", panel_x, 188, (180, 220, 255), 0.58, 2)

        self.resultbox.draw(annotated, 30, 300)
        self.warning.draw(annotated, 30, 390)

        fps = self.fpscounter.get() if SHOWFPS else 0
        if SHOWFPS:
            vietchu(annotated, f"FPS: {fps}", 30, 440, (255, 255, 0), 0.8, 2)

        self.shared_state.update({
            "total_rep": int(self.tongsolan),
            "good_rep": int(good_rep),
            "bad_rep": int(bad_rep),
            "display_good_rep": int(good_rep),
            "display_bad_rep": int(bad_rep),
            "status_text": str(self.trangthai),
            "rep_quality_score": int(self.last_rep_score),
            "quality_score_avg": float(quality_average),
            "keypoint_confidence_avg": float(confidence_average),
            "rep_scores": list(self.rep_scores),
            "rep_records": list(self.rep_records),
            "error_code_counts": dict(self.error_code_counts),
            "training_level": self.training_level,
            "training_level_label": training_level_label(self.training_level),
            "pose_classifier_available": bool(self.hybrid_guard.available),
            "pose_classifier_error": str(self.hybrid_guard.load_error or ""),
            "reference_motion_available": bool(self.reference_matcher.available),
            "reference_motion_error": str(self.reference_matcher.load_error or ""),
            "fps": int(fps),
            "workout_done": False,
        })

        return self.draw_emergency_overlay(annotated)


class SquatProcessor(BaseProcessor):
    LEFT_JOINTS = [5, 11, 13, 15]
    RIGHT_JOINTS = [6, 12, 14, 16]

    def __init__(self, shared_state=None, model=None):
        super().__init__(shared_state, model=model)
        self.thresholds = SquatThresholds(**threshold_profile("squat", self.training_level))
        self.trangthai = "CALIBRATING"
        self.standing_samples = deque(maxlen=30)
        self.baseline_standing_angle: float | None = None
        self.rep_start_angle: float | None = None
        self.descent_confirmed = False
        self.ascent_confirmed = False
        self.rep_min_knee = 999.0
        self.rep_min_torso = 999.0
        self.rep_frame = None
        self.rep_side = ""
        restored_baseline = float(self.shared_state.get("squat_standing_baseline", 0) or 0)
        if bool(self.shared_state.get("squat_calibration_ready", False)) and restored_baseline > 0:
            self.baseline_standing_angle = restored_baseline
            self.standing_samples.extend([restored_baseline] * self.thresholds.calibration_frames)
            self.trangthai = "STANDING"

    def _on_training_level_changed(self) -> None:
        self.thresholds = SquatThresholds(**threshold_profile("squat", self.training_level))
        self.trangthai = "CALIBRATING"
        self.standing_samples.clear()
        self.baseline_standing_angle = None
        self.shared_state["squat_standing_baseline"] = 0.0
        self.shared_state["squat_calibration_ready"] = False
        self.reset_rep()

    def reset_rep(self) -> None:
        self.rep_start_angle = None
        self.descent_confirmed = False
        self.ascent_confirmed = False
        self.rep_min_knee = 999.0
        self.rep_min_torso = 999.0
        self.rep_frame = None
        self.rep_side = ""
        self.begin_rep()
        self.temporal.reset_counters("squat_finish")

    def _baseline_ready(self) -> bool:
        return (
            self.baseline_standing_angle is not None
            and len(self.standing_samples) >= self.thresholds.calibration_frames
        )

    def _update_standing_baseline(self, knee_angle: float, knee_trend: float) -> None:
        if knee_angle >= self.thresholds.calibration_min_angle and abs(knee_trend) <= 2.5:
            self.standing_samples.append(float(knee_angle))
            if len(self.standing_samples) >= self.thresholds.calibration_frames:
                # A high percentile resists a few bent-knee frames without ever
                # requiring a perfectly locked 180-degree knee.
                self.baseline_standing_angle = float(np.percentile(self.standing_samples, 75))

    def _finish_angle(self) -> float:
        baseline = float(self.rep_start_angle or self.baseline_standing_angle or 160.0)
        return baseline - self.thresholds.finish_tolerance

    def _joints_for_side(self, side: str):
        return self.LEFT_JOINTS if side == "left" else self.RIGHT_JOINTS

    def process(self, frame):
        self.sync_training_level()
        results = self.infer(frame)
        annotated = results[0].plot() if results else frame.copy()
        if self.shared_state.get("active", False):
            return self.finish(annotated)

        if not results or not coNguoi(results):
            self.warning.trigger("noperson", COMMONMSG["noperson"])
            self.note_tracking_loss(
                self.trangthai not in {"CALIBRATING", "STANDING"},
                self.reset_rep,
                "STANDING" if self._baseline_ready() else "CALIBRATING",
            )
            return self.finish(annotated)

        kp, conf = layDiem(results)
        self.update_emergency(kp, conf, frame.copy(), annotated, "squat", self.trangthai)
        if self.shared_state.get("active", False):
            return self.finish(annotated)
        self.observe_pose_guard(kp, conf)
        if not self.posture_identity_gate("squat", kp, conf):
            if self.shared_state.get("exercise_identity_reason") == "posture_mismatch" and self.trangthai in {"CALIBRATING", "STANDING"}:
                self.trangthai = "STANDING" if self._baseline_ready() else "CALIBRATING"
                self.reset_rep()
            return self.finish(annotated)

        side = self.rep_side if self.trangthai != "STANDING" and self.rep_side else _choose_side(
            conf, self.LEFT_JOINTS, self.RIGHT_JOINTS, KEYPOINTTHRESH
        )
        if side is None:
            self.warning.trigger("missingleg", COMMONMSG["missingleg"])
            self.note_tracking_loss(
                self.trangthai not in {"CALIBRATING", "STANDING"},
                self.reset_rep,
                "STANDING" if self._baseline_ready() else "CALIBRATING",
            )
            return self.finish(annotated)

        joints = self._joints_for_side(side)
        if not _required_visible(conf, joints, KEYPOINTTHRESH):
            self.warning.trigger("missingleg", COMMONMSG["missingleg"])
            self.note_tracking_loss(
                self.trangthai not in {"CALIBRATING", "STANDING"},
                self.reset_rep,
                "STANDING" if self._baseline_ready() else "CALIBRATING",
            )
            return self.finish(annotated)

        shoulder_idx, hip_idx, knee_idx, ankle_idx = joints
        frame_confidence = self.record_valid_frame(conf, joints)
        knee_raw = tinhgoc(kp[hip_idx], kp[knee_idx], kp[ankle_idx])
        torso_raw = tinhgoc(kp[shoulder_idx], kp[hip_idx], kp[knee_idx])
        knee_angle = self.temporal.update("squat_knee", knee_raw)
        torso_angle = self.temporal.update("squat_torso", torso_raw)
        knee_trend = self.temporal.trend("squat_knee")

        if self.trangthai in {"CALIBRATING", "STANDING"}:
            self._update_standing_baseline(knee_angle, knee_trend)

        self.shared_state.update({
            "current_primary_angle": round(knee_angle, 1),
            "current_secondary_angle": round(torso_angle, 1),
            "primary_metric_name": "Góc gối",
            "secondary_metric_name": "Góc thân",
            "evaluation_side": side,
            "squat_standing_baseline": round(float(self.baseline_standing_angle or 0.0), 1),
            "squat_calibration_ready": bool(self._baseline_ready()),
        })

        if self.trangthai in {"CALIBRATING", "STANDING"}:
            if not self._baseline_ready():
                self.trangthai = "CALIBRATING"
                self.resultbox.set("HIEU CHUAN TU THE DUNG", "DUNG THOAI MAI, KHONG CAN KHOA CUNG DAU GOI")
            else:
                self.trangthai = "STANDING"
                start_angle = float(self.baseline_standing_angle) - self.thresholds.start_drop_from_baseline
                should_start = bool(knee_angle <= start_angle and knee_trend < -0.10)
                if self.temporal.stable("squat_start", should_start, 2):
                    self.trangthai = "DESCENDING"
                    self.begin_rep()
                    self.capture_admin_criteria(kp, conf, "start")
                    self.rep_start_angle = float(self.baseline_standing_angle)
                    self.rep_side = side
                    self.rep_min_knee = knee_angle
                    self.rep_min_torso = torso_angle
                    self.rep_frame = frame.copy()
                    self.descent_confirmed = False
                    self.ascent_confirmed = False
                    self.temporal.reset_counter("squat_finish")
        else:
            self.record_rep_frame(frame_confidence, knee_angle)
            self.capture_admin_criteria(kp, conf, "middle")
            self.observe_reference_metrics({"knee_angle": knee_angle, "torso_angle": torso_angle})
            if knee_angle < self.rep_min_knee:
                self.rep_min_knee = knee_angle
                self.rep_frame = frame.copy()
            self.rep_min_torso = min(self.rep_min_torso, torso_angle)

            if self.rep_frame_count > self.MAX_REP_FRAMES:
                self.missing_frame_count = self.MAX_MISSING_FRAMES - 1
                self.note_tracking_loss(True, self.reset_rep, "STANDING")
                return self.finish(annotated)

            descent_rom = max(0.0, float(self.rep_start_angle or 0.0) - self.rep_min_knee)
            if descent_rom >= self.thresholds.minimum_cycle_rom:
                self.descent_confirmed = True

            if self.trangthai == "DESCENDING":
                if knee_angle <= self.thresholds.bottom_angle or descent_rom >= max(18.0, self.thresholds.minimum_cycle_rom + 2.0):
                    self.trangthai = "BOTTOM"
                elif self.rep_frame_count >= self.MIN_REP_FRAMES and knee_trend > 0.18:
                    self.trangthai = "ASCENDING"
            elif self.trangthai == "BOTTOM" and knee_trend > 0.12:
                self.trangthai = "ASCENDING"

            if self.trangthai == "ASCENDING":
                self.ascent_confirmed = True

            if (
                self.trangthai == "ASCENDING"
                and self.rep_frame_count >= self.MIN_REP_FRAMES
                and self.temporal.stable("squat_finish", knee_angle >= self._finish_angle(), 2)
            ):
                # One rep is one complete DOWN -> UP cycle. A small knee bend
                # that returns to standing is discarded instead of counted.
                if self.descent_confirmed:
                    self.capture_admin_criteria(kp, conf, "end")
                    errors = []
                    # When a validated 3D reference exists, depth/rhythm are
                    # judged by the full reference trajectory (DTW) instead of
                    # a single hard-coded knee cutoff. This prevents a valid
                    # model-like squat from being marked bad just because its
                    # deepest 2D camera angle is a few degrees above a scalar
                    # threshold. The angle rule remains the fallback when no
                    # 3D reference is configured.
                    if (not self.reference_matcher.available) and self.rep_min_knee > self.thresholds.shallow_limit:
                        errors.append(("notlow", SQUATMSG["notlow"]["result"], SQUATMSG["notlow"]["advice"], "middle"))
                    if self.rep_min_torso < self.thresholds.torso_limit:
                        errors.append(("backlean", SQUATMSG["backlean"]["result"], SQUATMSG["backlean"]["advice"], "middle"))

                    score = score_squat(
                        self.rep_min_knee,
                        self.rep_min_torso,
                        self.rep_confidence(),
                        self.rep_stability(),
                        level=self.training_level,
                    )
                    metrics = [
                        f"BEN DANH GIA: {'TRAI' if self.rep_side == 'left' else 'PHAI'}",
                        f"GOC DUNG CA NHAN: {int(round(self.rep_start_angle or 0))}",
                        f"GOC GOI NHO NHAT: {int(self.rep_min_knee)}",
                        f"GOC THAN NHO NHAT: {int(self.rep_min_torso)}",
                    ]
                    self.finish_rep(
                        "squat", score, errors, SQUATMSG["good"], self.rep_frame, metrics,
                        good_score_threshold=self.thresholds.good_score_threshold,
                    )
                self.trangthai = "STANDING"
                self.reset_rep()

        vietchu(annotated, "BAI TAP: SQUAT", 30, 40, (255, 255, 255), 0.9, 2)
        vietchu(annotated, f"BEN DANH GIA: {'TRAI' if side == 'left' else 'PHAI'}", 30, 80)
        vietchu(annotated, f"GOC GOI (LOC): {int(knee_angle)}", 30, 120)
        vietchu(annotated, f"GOC THAN (LOC): {int(torso_angle)}", 30, 160)
        vietchu(annotated, f"GOC DUNG CA NHAN: {int(round(self.baseline_standing_angle or 0))}", 30, 200)
        vietchu(annotated, f"PHA: {self.trangthai}", 30, 240, (255, 255, 0))
        return self.finish(annotated)


class PushupProcessor(BaseProcessor):
    LEFT_JOINTS = [5, 7, 9, 11, 15]
    RIGHT_JOINTS = [6, 8, 10, 12, 16]

    def __init__(self, shared_state=None, model=None):
        super().__init__(shared_state, model=model)
        self.thresholds = PushupThresholds(**threshold_profile("pushup", self.training_level))
        self.trangthai = "CALIBRATING"
        self.top_samples = deque(maxlen=30)
        self.baseline_top_angle: float | None = None
        self.rep_start_angle: float | None = None
        self.descent_confirmed = False
        self.ascent_confirmed = False
        self.rep_min_elbow = 999.0
        self.rep_min_body = 999.0
        self.rep_frame = None
        self.rep_side = ""
        restored_baseline = float(self.shared_state.get("pushup_top_baseline", 0) or 0)
        if bool(self.shared_state.get("pushup_calibration_ready", False)) and restored_baseline > 0:
            self.baseline_top_angle = restored_baseline
            self.top_samples.extend([restored_baseline] * self.thresholds.calibration_frames)
            self.trangthai = "PLANK_UP"

    def _on_training_level_changed(self) -> None:
        self.thresholds = PushupThresholds(**threshold_profile("pushup", self.training_level))
        self.trangthai = "CALIBRATING"
        self.top_samples.clear()
        self.baseline_top_angle = None
        self.shared_state["pushup_top_baseline"] = 0.0
        self.shared_state["pushup_calibration_ready"] = False
        self.reset_rep()

    def reset_rep(self) -> None:
        self.rep_start_angle = None
        self.descent_confirmed = False
        self.ascent_confirmed = False
        self.rep_min_elbow = 999.0
        self.rep_min_body = 999.0
        self.rep_frame = None
        self.rep_side = ""
        self.begin_rep()
        self.temporal.reset_counter("pushup_finish")

    def _baseline_ready(self) -> bool:
        return (
            self.baseline_top_angle is not None
            and len(self.top_samples) >= self.thresholds.calibration_frames
        )

    def _update_top_baseline(self, elbow_angle: float, elbow_trend: float, body_angle: float) -> None:
        if (
            elbow_angle >= self.thresholds.calibration_min_angle
            and body_angle >= 105.0
            and abs(elbow_trend) <= 3.5
        ):
            self.top_samples.append(float(elbow_angle))
            if len(self.top_samples) >= self.thresholds.calibration_frames:
                self.baseline_top_angle = float(np.percentile(self.top_samples, 65))

    def _finish_angle(self) -> float:
        baseline = float(self.rep_start_angle or self.baseline_top_angle or 155.0)
        return baseline - self.thresholds.finish_tolerance

    def _joints_for_side(self, side: str):
        return self.LEFT_JOINTS if side == "left" else self.RIGHT_JOINTS

    def process(self, frame):
        self.sync_training_level()
        results = self.infer(frame)
        annotated = results[0].plot() if results else frame.copy()
        if self.shared_state.get("active", False):
            return self.finish(annotated)

        if not results or not coNguoi(results):
            self.warning.trigger("noperson", COMMONMSG["noperson"])
            self.note_tracking_loss(
                self.trangthai not in {"CALIBRATING", "PLANK_UP"},
                self.reset_rep,
                "PLANK_UP" if self._baseline_ready() else "CALIBRATING",
            )
            return self.finish(annotated)

        kp, conf = layDiem(results)
        self.update_emergency(kp, conf, frame.copy(), annotated, "pushup", self.trangthai)
        if self.shared_state.get("active", False):
            return self.finish(annotated)
        self.observe_pose_guard(kp, conf)
        if not self.posture_identity_gate("pushup", kp, conf, strict_unknown=True):
            if self.shared_state.get("exercise_identity_reason") in {"posture_mismatch", "missing_torso_keypoints"}:
                if self.trangthai in {"CALIBRATING", "PLANK_UP"}:
                    self.trangthai = "CALIBRATING"
                    self.top_samples.clear()
                    self.baseline_top_angle = None
                    self.shared_state["pushup_top_baseline"] = 0.0
                    self.shared_state["pushup_calibration_ready"] = False
                    self.reset_rep()
            return self.finish(annotated)

        side = self.rep_side if self.trangthai != "PLANK_UP" and self.rep_side else _choose_side(
            conf, self.LEFT_JOINTS, self.RIGHT_JOINTS, KEYPOINTTHRESH
        )
        if side is None:
            self.warning.trigger("missingarm", COMMONMSG["missingarm"])
            self.note_tracking_loss(
                self.trangthai not in {"CALIBRATING", "PLANK_UP"},
                self.reset_rep,
                "PLANK_UP" if self._baseline_ready() else "CALIBRATING",
            )
            return self.finish(annotated)

        joints = self._joints_for_side(side)
        arm_joints = joints[:3]  # shoulder, elbow, wrist
        if not _required_visible(conf, arm_joints, KEYPOINTTHRESH):
            self.warning.trigger("missingarm", COMMONMSG["missingarm"])
            self.note_tracking_loss(
                self.trangthai not in {"CALIBRATING", "PLANK_UP"},
                self.reset_rep,
                "PLANK_UP" if self._baseline_ready() else "CALIBRATING",
            )
            return self.finish(annotated)

        shoulder_idx, elbow_idx, wrist_idx, hip_idx, ankle_idx = joints
        frame_confidence = self.record_valid_frame(conf, arm_joints)
        elbow_raw = tinhgoc(kp[shoulder_idx], kp[elbow_idx], kp[wrist_idx])
        ankle_visible = thayKhau(conf, ankle_idx, 0.25)
        knee_idx = 13 if side == "left" else 14
        lower_idx = ankle_idx if ankle_visible else knee_idx
        body_raw = tinhgoc(kp[shoulder_idx], kp[hip_idx], kp[lower_idx])
        elbow_angle = self.temporal.update("pushup_elbow", elbow_raw)
        body_angle = self.temporal.update("pushup_body", body_raw)
        elbow_trend = self.temporal.trend("pushup_elbow")

        if self.trangthai in {"CALIBRATING", "PLANK_UP"}:
            self._update_top_baseline(elbow_angle, elbow_trend, body_angle)

        self.shared_state.update({
            "current_primary_angle": round(elbow_angle, 1),
            "current_secondary_angle": round(body_angle, 1),
            "primary_metric_name": "Góc khuỷu tay",
            "secondary_metric_name": "Độ thẳng thân",
            "evaluation_side": side,
            "pushup_top_baseline": round(float(self.baseline_top_angle or 0.0), 1),
            "pushup_calibration_ready": bool(self._baseline_ready()),
        })

        if self.trangthai in {"CALIBRATING", "PLANK_UP"}:
            if not self._baseline_ready():
                self.trangthai = "CALIBRATING"
                self.resultbox.set("HIEU CHUAN TU THE CHUAN BI", "GIU THAN THANG O VI TRI TREN TRONG VAI GIAY")
            else:
                self.trangthai = "PLANK_UP"
                start_angle = float(self.baseline_top_angle) - self.thresholds.start_drop_from_baseline
                should_start = bool(elbow_angle <= start_angle and elbow_trend < -0.08)
                if self.temporal.stable("pushup_start", should_start, 2):
                    self.trangthai = "LOWERING"
                    self.begin_rep()
                    self.capture_admin_criteria(kp, conf, "start")
                    self.rep_start_angle = float(self.baseline_top_angle)
                    self.rep_side = side
                    self.rep_min_elbow = elbow_angle
                    self.rep_min_body = body_angle
                    self.rep_frame = frame.copy()
                    self.descent_confirmed = False
                    self.ascent_confirmed = False
                    self.temporal.reset_counter("pushup_finish")
        else:
            self.record_rep_frame(frame_confidence, elbow_angle)
            self.capture_admin_criteria(kp, conf, "middle")
            self.observe_reference_metrics({"elbow_angle": elbow_angle, "body_angle": body_angle})
            if elbow_angle < self.rep_min_elbow:
                self.rep_min_elbow = elbow_angle
                self.rep_frame = frame.copy()
            self.rep_min_body = min(self.rep_min_body, body_angle)

            if self.rep_frame_count > self.MAX_REP_FRAMES:
                self.missing_frame_count = self.MAX_MISSING_FRAMES - 1
                self.note_tracking_loss(True, self.reset_rep, "PLANK_UP")
                return self.finish(annotated)

            descent_rom = max(0.0, float(self.rep_start_angle or 0.0) - self.rep_min_elbow)
            if descent_rom >= self.thresholds.minimum_cycle_rom:
                self.descent_confirmed = True

            if self.trangthai == "LOWERING":
                if elbow_angle <= self.thresholds.bottom_angle or descent_rom >= max(14.0, self.thresholds.minimum_cycle_rom + 2.0):
                    self.trangthai = "BOTTOM"
                elif self.rep_frame_count >= self.MIN_REP_FRAMES and elbow_trend > 0.12:
                    self.trangthai = "RISING"
            elif self.trangthai == "BOTTOM" and elbow_trend > 0.08:
                self.trangthai = "RISING"

            if self.trangthai == "RISING":
                self.ascent_confirmed = True

            if (
                self.trangthai == "RISING"
                and self.rep_frame_count >= self.MIN_REP_FRAMES
                and self.temporal.stable("pushup_finish", elbow_angle >= self._finish_angle(), 2)
            ):
                if self.descent_confirmed:
                    self.capture_admin_criteria(kp, conf, "end")
                    errors = []
                    if self.rep_min_elbow > self.thresholds.shallow_limit:
                        errors.append(("notlow", PUSHUPMSG["notlow"]["result"], PUSHUPMSG["notlow"]["advice"], "middle"))
                    if self.rep_min_body < self.thresholds.body_line_limit:
                        errors.append(("bodyline", PUSHUPMSG["bodyline"]["result"], PUSHUPMSG["bodyline"]["advice"], "middle"))

                    score = score_pushup(
                        self.rep_min_elbow,
                        self.rep_min_body,
                        self.rep_confidence(),
                        self.rep_stability(),
                        level=self.training_level,
                    )
                    metrics = [
                        f"BEN DANH GIA: {'TRAI' if self.rep_side == 'left' else 'PHAI'}",
                        f"GOC TREN CA NHAN: {int(round(self.rep_start_angle or 0))}",
                        f"GOC KHUYU TAY NHO NHAT: {int(self.rep_min_elbow)}",
                        f"DO THANG THAN NHO NHAT: {int(self.rep_min_body)}",
                    ]
                    self.finish_rep(
                        "pushup", score, errors, PUSHUPMSG["good"], self.rep_frame, metrics,
                        good_score_threshold=self.thresholds.good_score_threshold,
                    )
                self.trangthai = "PLANK_UP"
                self.reset_rep()

        vietchu(annotated, "BAI TAP: HIT DAT", 30, 40, (255, 255, 255), 0.9, 2)
        vietchu(annotated, f"BEN DANH GIA: {'TRAI' if side == 'left' else 'PHAI'}", 30, 80)
        vietchu(annotated, f"GOC KHUYU (LOC): {int(elbow_angle)}", 30, 120)
        vietchu(annotated, f"DO THANG THAN: {int(body_angle)}", 30, 160)
        vietchu(annotated, f"GOC TREN CA NHAN: {int(round(self.baseline_top_angle or 0))}", 30, 200)
        vietchu(annotated, f"PHA: {self.trangthai}", 30, 240, (255, 255, 0))
        return self.finish(annotated)


class CurlProcessor(BaseProcessor):
    """Adaptive single-arm bicep-curl analysis.

    The processor calibrates the user's comfortable starting elbow angle and
    evaluates return-to-baseline instead of demanding a fixed 180-degree
    lockout. Left and right curls use the same rule set but different keypoints.
    """

    def __init__(self, side="left", shared_state=None, model=None):
        super().__init__(shared_state, model=model)
        self.thresholds = CurlThresholds(**threshold_profile("curl", self.training_level))
        self.side = "right" if side == "right" else "left"
        self.trangthai = "CALIBRATING"
        self.extension_ready = False

        if self.side == "right":
            self.shoulder_idx, self.elbow_idx, self.wrist_idx = 6, 8, 10
            self.opposite_joints = [5, 7, 9]
            self.side_text = "TAY PHAI"
        else:
            self.shoulder_idx, self.elbow_idx, self.wrist_idx = 5, 7, 9
            self.opposite_joints = [6, 8, 10]
            self.side_text = "TAY TRAI"
        self.side_guard = BilateralCurlGuard(self.side)
        # Detect the teacher-reported case where curl-left is selected but the
        # user curls the right arm (or vice versa) before the selected-side
        # state machine even starts.
        self.opposite_extension_samples = deque(maxlen=30)
        self.opposite_extension_baseline: float | None = None
        self.last_wrong_side_feedback_at = 0.0

        self.extension_samples = deque(maxlen=30)
        self.baseline_extension_angle: float | None = None
        self.rep_extension_target: float | None = None

        self.rep_min_elbow = 999.0
        self.rep_max_elbow = 0.0
        self.rep_max_elbow_shift = 0.0
        self.ready_elbow = None
        self.start_elbow = None
        self.upper_arm_length = 1.0
        self.rep_frame = None
        self.ready_knee_angle: float | None = None
        self.rep_knee_start: float | None = None
        self.rep_min_knee: float | None = None
        restored_baseline = min(
            float(self.shared_state.get("curl_extension_baseline", 0) or 0),
            self.thresholds.calibration_max_angle,
        )
        if bool(self.shared_state.get("curl_calibration_ready", False)) and restored_baseline > 0:
            self.baseline_extension_angle = restored_baseline
            self.shared_state["curl_extension_baseline"] = restored_baseline
            self.extension_samples.extend([restored_baseline] * self.thresholds.calibration_frames)
            self.extension_ready = True
            self.trangthai = "EXTENDED"

    def _on_training_level_changed(self) -> None:
        self.thresholds = CurlThresholds(**threshold_profile("curl", self.training_level))
        self.trangthai = "CALIBRATING"
        self.extension_ready = False
        self.extension_samples.clear()
        self.baseline_extension_angle = None
        self.shared_state["curl_extension_baseline"] = 0.0
        self.shared_state["curl_calibration_ready"] = False
        self.reset_rep()

    @property
    def joints(self):
        return [self.shoulder_idx, self.elbow_idx, self.wrist_idx]

    def _baseline_ready(self) -> bool:
        return (
            self.baseline_extension_angle is not None
            and len(self.extension_samples) >= self.thresholds.calibration_frames
        )

    def _update_extension_baseline(self, elbow_angle: float, elbow_trend: float) -> None:
        """Estimate a stable personal extension baseline from visible frames."""
        if (
            elbow_angle >= self.thresholds.calibration_min_angle
            and abs(elbow_trend) <= 2.5
        ):
            ergonomic_angle = min(float(elbow_angle), self.thresholds.calibration_max_angle)
            self.extension_samples.append(ergonomic_angle)
            # Cap the effective target below a hard elbow lockout. This makes a
            # comfortable 140-165 degree start/end posture valid even if the
            # pose estimator reports a near-180 angle for a straight-looking arm.
            self.baseline_extension_angle = min(
                self.thresholds.calibration_max_angle,
                float(np.percentile(self.extension_samples, 65)),
            )

        self.extension_ready = self._baseline_ready()
        if self.extension_ready and self.trangthai == "CALIBRATING":
            self.trangthai = "EXTENDED"

    def _detect_pre_rep_wrong_side(self, selected_angle: float, opposite_angle: float | None) -> bool:
        """Speak immediately when only the non-selected arm starts curling."""
        if opposite_angle is None:
            self.temporal.reset_counter("curl_wrong_side_pre")
            return False
        opposite_trend = self.temporal.trend("curl_opposite_elbow")
        if opposite_angle >= 115.0 and abs(opposite_trend) <= 2.5:
            self.opposite_extension_samples.append(min(float(opposite_angle), 170.0))
            if len(self.opposite_extension_samples) >= 4:
                self.opposite_extension_baseline = float(np.percentile(self.opposite_extension_samples, 65))

        baseline = self.opposite_extension_baseline or self.baseline_extension_angle
        selected_still_extended = selected_angle >= (self._start_flexion_angle() - 5.0)
        # Require a deep curl drop (50 deg) on the opposite arm before flagging wrong side.
        # If the selected arm has already begun flexing (selected_angle has dropped meaningfully),
        # do not override with wrong-side; the user is exercising the correct arm.
        selected_flexing = selected_angle < (self._start_flexion_angle() - 15.0)
        wrong_motion = bool(
            baseline is not None
            and selected_still_extended
            and not selected_flexing
            and (float(baseline) - float(opposite_angle)) >= 50.0
            and opposite_trend < -0.18
        )
        if not self.temporal.stable("curl_wrong_side_pre", wrong_motion, 6):
            return False
        now = time.time()
        if now - self.last_wrong_side_feedback_at < 4.0:
            return True
        self.last_wrong_side_feedback_at = now
        selected_vi = "tay phải" if self.side == "right" else "tay trái"
        opposite_vi = "tay trái" if self.side == "right" else "tay phải"
        self.reject_attempt(
            f"curl-{self.side}",
            "wrongside",
            "SAI BEN TAY",
            f"Bạn đang chọn {selected_vi} nhưng hệ thống phát hiện {opposite_vi} đang cuốn tạ. Hãy giữ {opposite_vi} yên và tập đúng {selected_vi}.",
            {
                "selected_side": self.side,
                "selected_angle": round(float(selected_angle), 1),
                "opposite_angle": round(float(opposite_angle), 1),
                "opposite_baseline": round(float(baseline), 1),
            },
        )
        self.temporal.reset_counter("curl_wrong_side_pre")
        return True

    def _start_flexion_angle(self) -> float:
        baseline = self.baseline_extension_angle or self.thresholds.calibration_min_angle
        return min(140.0, baseline - self.thresholds.start_drop_from_baseline)

    def _full_flexion_angle(self) -> float:
        baseline = self.rep_extension_target or self.baseline_extension_angle or 150.0
        return min(
            self.thresholds.full_flexion_ceiling,
            baseline - self.thresholds.minimum_range_of_motion,
        )

    def _finish_extension_angle(self) -> float:
        baseline = self.rep_extension_target or self.baseline_extension_angle or 150.0
        return max(
            self.thresholds.calibration_min_angle - 5.0,
            baseline - self.thresholds.finish_tolerance,
        )

    def _incomplete_extension_limit(self) -> float:
        baseline = self.rep_extension_target or self.baseline_extension_angle or 150.0
        return max(
            self.thresholds.calibration_min_angle - 10.0,
            baseline - self.thresholds.incomplete_extension_tolerance,
        )

    def reset_rep(self) -> None:
        self.rep_min_elbow = 999.0
        self.rep_max_elbow = 0.0
        self.rep_max_elbow_shift = 0.0
        self.start_elbow = None
        self.upper_arm_length = 1.0
        self.rep_frame = None
        self.rep_extension_target = None
        self.rep_knee_start = None
        self.rep_min_knee = None
        self.begin_rep()
        self.side_guard.reset()
        self.temporal.reset_counter("curl_finish")

    def _complete_rep(self, frame, force_incomplete_extension: bool = False) -> None:
        errors = []
        side_evidence = self.side_guard.evidence()
        self.shared_state["curl_side_evidence"] = side_evidence
        if self.side_guard.wrong_side():
            selected_vi = "tay phải" if self.side == "right" else "tay trái"
            opposite_vi = "tay trái" if self.side == "right" else "tay phải"
            self.reject_attempt(
                f"curl-{self.side}",
                "wrongside",
                "SAI BEN TAY",
                f"Bạn đang chọn {selected_vi} nhưng {opposite_vi} có biên độ lớn hơn. Rep này không được tính; hãy giữ {opposite_vi} yên.",
                side_evidence,
            )
            return
        if self.rep_knee_start is not None and self.rep_min_knee is not None:
            knee_drop = max(0.0, float(self.rep_knee_start) - float(self.rep_min_knee))
            if knee_drop >= 28.0 and float(self.rep_min_knee) <= 135.0:
                self.reject_attempt(
                    f"curl-{self.side}",
                    "exercise_mismatch",
                    "CHUA DUNG BAI TAP",
                    "Hệ thống phát hiện thân dưới đang thực hiện chuyển động giống squat. Rep curl này không được tính.",
                    {"knee_drop": round(knee_drop, 1), "min_knee": round(float(self.rep_min_knee), 1)},
                )
                return
        flexion_limit = self._full_flexion_angle()
        extension_limit = self._incomplete_extension_limit()

        if self.rep_min_elbow > flexion_limit:
            errors.append(("notbend", CURLMSG["notbend"]["result"], CURLMSG["notbend"]["advice"], "middle"))
        if force_incomplete_extension or self.rep_max_elbow < extension_limit:
            errors.append(("notstraight", CURLMSG["notstraight"]["result"], CURLMSG["notstraight"]["advice"], "end"))
        if self.rep_max_elbow_shift > self.thresholds.elbow_shift_limit:
            errors.append(("elbowshift", CURLMSG["elbowshift"]["result"], CURLMSG["elbowshift"]["advice"], "middle"))

        extension_target = self.rep_extension_target or self.baseline_extension_angle
        score = score_curl(
            self.rep_min_elbow,
            self.rep_max_elbow,
            self.rep_max_elbow_shift,
            self.rep_confidence(),
            self.rep_stability(),
            extension_target=extension_target,
            level=self.training_level,
        )
        metrics = [
            f"BEN TAP: {self.side_text}",
            f"GOC BAT DAU CA NHAN: {int(round(extension_target or 0))}",
            f"GOC NHO NHAT: {int(self.rep_min_elbow)}",
            f"GOC LON NHAT: {int(self.rep_max_elbow)}",
            f"BIEN DO: {int(max(0.0, self.rep_max_elbow - self.rep_min_elbow))}",
            f"DO LECH KHUYU: {int(self.rep_max_elbow_shift)}%",
        ]
        self.finish_rep(
            "curl",
            score,
            errors,
            CURLMSG["good"],
            self.rep_frame if self.rep_frame is not None else frame,
            metrics,
            good_score_threshold=self.thresholds.good_score_threshold,
        )

    def process(self, frame):
        self.sync_training_level()
        results = self.infer(frame)
        annotated = results[0].plot() if results else frame.copy()
        if self.shared_state.get("active", False):
            return self.finish(annotated)

        if not results or not coNguoi(results):
            self.warning.trigger("noperson", COMMONMSG["noperson"])
            self.note_tracking_loss(
                self.trangthai not in {"CALIBRATING", "EXTENDED"},
                self.reset_rep,
                "CALIBRATING" if not self._baseline_ready() else "EXTENDED",
            )
            return self.finish(annotated)

        kp, conf = layDiem(results)
        self.update_emergency(kp, conf, frame.copy(), annotated, "curl", self.trangthai)
        if self.shared_state.get("active", False):
            return self.finish(annotated)
        self.observe_pose_guard(kp, conf)
        if not self.posture_identity_gate("curl", kp, conf):
            if self.shared_state.get("exercise_identity_reason") == "posture_mismatch":
                self.trangthai = "EXTENDED" if self._baseline_ready() else "CALIBRATING"
                self.reset_rep()
            return self.finish(annotated)

        if not _required_visible(conf, self.joints, KEYPOINTTHRESH):
            self.warning.trigger("missingarm", COMMONMSG["missingarm"])
            self.note_tracking_loss(
                self.trangthai not in {"CALIBRATING", "EXTENDED"},
                self.reset_rep,
                "CALIBRATING" if not self._baseline_ready() else "EXTENDED",
            )
            return self.finish(annotated)

        shoulder = kp[self.shoulder_idx]
        elbow = kp[self.elbow_idx]
        wrist = kp[self.wrist_idx]
        frame_confidence = self.record_valid_frame(conf, self.joints)
        elbow_raw = tinhgoc(shoulder, elbow, wrist)
        elbow_angle = self.temporal.update("curl_elbow", elbow_raw)
        elbow_trend = self.temporal.trend("curl_elbow")
        opposite_angle = None
        if _required_visible(conf, self.opposite_joints, KEYPOINTTHRESH):
            os_idx, oe_idx, ow_idx = self.opposite_joints
            opposite_raw = tinhgoc(kp[os_idx], kp[oe_idx], kp[ow_idx])
            opposite_angle = self.temporal.update("curl_opposite_elbow", opposite_raw)
        knee_angle = _average_knee_angle(kp, conf)
        if self.trangthai in {"CALIBRATING", "EXTENDED"}:
            self._detect_pre_rep_wrong_side(elbow_angle, opposite_angle)
        else:
            self.side_guard.observe(elbow_angle, opposite_angle)
            if knee_angle is not None:
                if self.rep_knee_start is None:
                    self.rep_knee_start = float(knee_angle)
                self.rep_min_knee = float(knee_angle) if self.rep_min_knee is None else min(float(self.rep_min_knee), float(knee_angle))

        if self.trangthai in {"CALIBRATING", "EXTENDED"}:
            self._update_extension_baseline(elbow_angle, elbow_trend)

        self.shared_state.update({
            "current_primary_angle": round(elbow_angle, 1),
            "current_secondary_angle": round(self.rep_max_elbow_shift, 1),
            "primary_metric_name": "Góc khuỷu tay",
            "secondary_metric_name": "Độ lệch khuỷu (%)",
            "evaluation_side": self.side,
            "curl_extension_baseline": round(float(self.baseline_extension_angle or 0.0), 1),
            "curl_calibration_ready": bool(self._baseline_ready()),
            "curl_side_evidence": self.side_guard.evidence(),
            "curl_knee_angle": round(float(knee_angle or 0.0), 1) if knee_angle is not None else 0.0,
        })

        if self.trangthai in {"CALIBRATING", "EXTENDED"}:
            if self._baseline_ready():
                self.trangthai = "EXTENDED"
                self.ready_elbow = np.asarray(elbow, dtype=np.float32).copy()
                self.upper_arm_length = max(
                    float(np.linalg.norm(np.asarray(shoulder) - np.asarray(elbow))),
                    1.0,
                )

                start_angle = self._start_flexion_angle()
                if (
                    elbow_trend < -0.25
                    and self.temporal.stable("curl_start", elbow_angle <= start_angle, 3)
                ):
                    self.trangthai = "LIFTING"
                    self.begin_rep()
                    self.capture_admin_criteria(kp, conf, "start")
                    self.rep_extension_target = float(self.baseline_extension_angle)
                    self.rep_min_elbow = elbow_angle
                    self.rep_max_elbow = float(self.baseline_extension_angle)
                    self.start_elbow = (
                        self.ready_elbow.copy()
                        if self.ready_elbow is not None
                        else np.asarray(elbow).copy()
                    )
                    self.rep_frame = frame.copy()
                    self.rep_knee_start = float(knee_angle) if knee_angle is not None else None
                    self.rep_min_knee = float(knee_angle) if knee_angle is not None else None
                    self.temporal.reset_counter("curl_finish")
            else:
                self.trangthai = "CALIBRATING"
                self.resultbox.set(
                    "HIEU CHUAN TU THE BAT DAU",
                    "GIU TAY THOAI MAI GAN THANG, KHONG KHOA CUNG KHUYU",
                )
        else:
            self.record_rep_frame(frame_confidence, elbow_angle)
            self.capture_admin_criteria(kp, conf, "middle")
            self.observe_reference_metrics({"elbow_angle": elbow_angle, "elbow_shift": self.rep_max_elbow_shift})
            if elbow_angle < self.rep_min_elbow:
                self.rep_min_elbow = elbow_angle
                self.rep_frame = frame.copy()
            self.rep_max_elbow = max(self.rep_max_elbow, elbow_angle)

            if self.start_elbow is not None:
                displacement = float(np.linalg.norm(np.asarray(elbow) - np.asarray(self.start_elbow)))
                shift_ratio = displacement / self.upper_arm_length * 100.0
                self.rep_max_elbow_shift = max(self.rep_max_elbow_shift, shift_ratio)

            if self.rep_frame_count > self.MAX_REP_FRAMES:
                self.missing_frame_count = self.MAX_MISSING_FRAMES - 1
                self.note_tracking_loss(True, self.reset_rep, "EXTENDED")
                return self.finish(annotated)

            flexion_limit = self._full_flexion_angle()
            finish_limit = self._finish_extension_angle()
            incomplete_limit = self._incomplete_extension_limit()

            if self.trangthai == "LIFTING":
                if elbow_angle <= flexion_limit:
                    self.trangthai = "FLEXED"
                elif self.rep_frame_count >= self.MIN_REP_FRAMES and elbow_trend > 0.7:
                    # Reversal before the target flexion is still an attempted rep.
                    self.trangthai = "LOWERING"
            elif self.trangthai == "FLEXED" and elbow_trend > 0.4:
                self.trangthai = "LOWERING"
            elif self.trangthai == "LOWERING":
                if (
                    self.rep_frame_count >= self.MIN_REP_FRAMES
                    and self.temporal.stable("curl_finish", elbow_angle >= finish_limit, 3)
                ):
                    self.rep_max_elbow = max(self.rep_max_elbow, elbow_angle)
                    completed_extension = self.rep_max_elbow
                    self.capture_admin_criteria(kp, conf, "end")
                    self._complete_rep(frame, force_incomplete_extension=False)
                    self.trangthai = "EXTENDED"
                    # Slowly adapt the baseline between reps, never during a rep.
                    if completed_extension >= self.thresholds.calibration_min_angle:
                        self.extension_samples.append(
                            min(float(completed_extension), self.thresholds.calibration_max_angle)
                        )
                        self.baseline_extension_angle = min(
                            self.thresholds.calibration_max_angle,
                            float(np.percentile(self.extension_samples, 65)),
                        )
                    self.ready_elbow = np.asarray(elbow, dtype=np.float32).copy()
                    self.upper_arm_length = max(
                        float(np.linalg.norm(np.asarray(shoulder) - np.asarray(elbow))),
                        1.0,
                    )
                    self.reset_rep()
                elif (
                    self.rep_frame_count >= self.MIN_REP_FRAMES + 3
                    and self.rep_max_elbow >= incomplete_limit - 10.0
                    and elbow_trend < -0.7
                ):
                    # The user begins another curl before returning close enough
                    # to their own calibrated start posture.
                    self.capture_admin_criteria(kp, conf, "end")
                    self._complete_rep(frame, force_incomplete_extension=True)
                    self.trangthai = "EXTENDED"
                    self.ready_elbow = None
                    self.reset_rep()

        baseline_text = int(round(self.baseline_extension_angle or 0.0))
        vietchu(annotated, "BAI TAP: NANG TA TAY TRUOC", 30, 40, (255, 255, 255), 0.9, 2)
        vietchu(annotated, f"BEN TAP: {self.side_text}", 30, 80)
        vietchu(annotated, f"GOC KHUYU (LOC): {int(elbow_angle)}", 30, 120)
        vietchu(annotated, f"GOC BAT DAU CA NHAN: {baseline_text}", 30, 160)
        vietchu(annotated, f"DO LECH KHUYU: {int(self.rep_max_elbow_shift)}%", 30, 200)
        vietchu(annotated, f"PHA: {self.trangthai}", 30, 240, (255, 255, 0))
        return self.finish(annotated)


class LearnedExerciseProcessor(BaseProcessor):
    """Generic three-state processor for an Admin-trained custom exercise.

    A custom exercise is trained with ``<slug>_start``, ``<slug>_middle`` and
    ``<slug>_end`` labels. YOLO Pose still supplies the skeleton; the learned
    classifier only decides the movement phase. A rep is accepted after a
    stable start -> middle -> end sequence. This mirrors the benchmark thesis
    workflow while keeping FitMotion's confidence filtering and audit trail.
    """

    STABLE_FRAMES = 3

    def __init__(self, slug="unknown", shared_state=None, model=None):
        super().__init__(shared_state, model=model, load_model=(model is None))
        self.slug = str(slug or "unknown")
        self.model_prefix = self.slug.lower().replace("-", "_").replace(" ", "_")
        self.sequence_state = "WAIT_START"
        self.phase_history = deque(maxlen=5)
        self.classifier_confidences: list[float] = []
        self.admin_rule_errors: list[tuple] = []
        self.trangthai = "CHO MODEL" if not self.hybrid_guard.available else "CHO BAT DAU"

    def _phase_from_label(self, label: str) -> str | None:
        value = str(label or "").strip().lower().replace("-", "_")
        prefix = self.model_prefix + "_"
        if not value.startswith(prefix):
            return None
        suffix = value[len(prefix):]
        if suffix.startswith("start"):
            return "start"
        if suffix.startswith("middle"):
            return "middle"
        if suffix.startswith("end"):
            return "end"
        return None

    def _stable_phase(self) -> tuple[str | None, float]:
        if len(self.phase_history) < self.STABLE_FRAMES:
            return None, 0.0
        recent = list(self.phase_history)[-self.STABLE_FRAMES:]
        phases = [item[0] for item in recent]
        if phases[0] and all(item == phases[0] for item in phases):
            return phases[0], float(np.mean([item[1] for item in recent]))
        return None, 0.0

    def _reset_generic_rep(self):
        self.classifier_confidences = []
        self.admin_rule_errors = []
        self.rep_confidences = []
        self.rep_primary_values = []
        self.rep_frame_count = 0
        self.missing_frame_count = 0

    def _complete_generic_rep(self, frame):
        classifier_avg = float(np.mean(self.classifier_confidences)) if self.classifier_confidences else 0.0
        keypoint_avg = self.rep_confidence()
        score_value = int(round(max(0.0, min(1.0, classifier_avg * 0.70 + keypoint_avg * 0.30)) * 100.0))
        score = RepScore(
            total=score_value,
            components={
                "classifier_confidence": int(round(classifier_avg * 100.0)),
                "keypoint_confidence": int(round(keypoint_avg * 100.0)),
            },
            weights={"classifier_confidence": 0.70, "keypoint_confidence": 0.30},
        )
        threshold = int(round(float(self.shared_state.get("min_quality_avg", 60.0) or 60.0)))
        self.finish_rep(
            self.slug,
            score,
            [],
            {
                "result": "DONG TAC DA DUOC XAC NHAN",
                "advice": "GIU Nhip DONG TAC ON DINH VA TIEP TUC",
            },
            frame,
            [
                f"CLASSIFIER: {int(round(classifier_avg * 100))}%",
                f"KEYPOINT: {int(round(keypoint_avg * 100))}%",
            ],
            good_score_threshold=threshold,
        )
        self.sequence_state = "WAIT_START"
        self.trangthai = "CHO BAT DAU"
        self._reset_generic_rep()

    def _capture_admin_rules(self, kp, conf, phase: str) -> None:
        self.capture_admin_criteria(kp, conf, phase)

    def process(self, frame):
        self.sync_training_level()
        annotated = frame.copy()

        if not self.hybrid_guard.available:
            self.trangthai = "CHUA TRAIN MODEL"
            self.shared_state["feedback_text"] = "Bài tập mới chưa có model đã vượt kiểm thử độc lập."
            self.shared_state["feedback_advice"] = "Admin cần tải dữ liệu start/middle/end và bấm Huấn luyện mô hình trên web."
            vietchu(annotated, "BAI TAP MOI - CHUA CO MODEL DAT GATE", 30, 70, (0, 165, 255), 0.78, 2)
            vietchu(annotated, f"SLUG: {self.slug}", 30, 110, (255, 255, 255), 0.68, 2)
            return self.finish(annotated)

        results = self.infer(frame)
        if not results or not coNguoi(results):
            self.phase_history.clear()
            self.note_tracking_loss(
                self.sequence_state != "WAIT_START",
                self._reset_generic_rep,
                "CHO BAT DAU",
            )
            vietchu(annotated, "KHONG THAY NGUOI TAP", 30, 70, (0, 165, 255), 0.8, 2)
            return self.finish(annotated)

        try:
            kp, conf = layDiem(results)
        except Exception:
            return self.finish(annotated)

        if len(kp) < 17 or len(conf) < 17:
            return self.finish(annotated)

        mean_conf = float(np.mean(np.asarray(conf[:17], dtype=np.float32)))
        self.session_confidences.append(mean_conf)
        self.update_emergency(kp, conf, frame, annotated, self.slug, self.sequence_state)

        prediction = self.observe_pose_guard(kp, conf)
        label = str((prediction or {}).get("label", ""))
        pred_conf = float((prediction or {}).get("confidence", 0.0) or 0.0)
        phase = self._phase_from_label(label)
        if phase and pred_conf >= 0.50:
            self.phase_history.append((phase, pred_conf))
        else:
            self.phase_history.append((None, pred_conf))

        if self.sequence_state != "WAIT_START":
            self.record_rep_frame(mean_conf, pred_conf * 100.0)
            self.classifier_confidences.append(pred_conf)

        stable_phase, stable_conf = self._stable_phase()
        if stable_phase == "start" and self.sequence_state == "WAIT_START":
            self.begin_rep()
            self._reset_generic_rep()
            self._capture_admin_rules(kp, conf, "start")
            self.sequence_state = "START"
            self.trangthai = "BAT DAU"
        elif stable_phase == "middle" and self.sequence_state == "START":
            self._capture_admin_rules(kp, conf, "middle")
            self.sequence_state = "MIDDLE"
            self.trangthai = "GIUA DONG TAC"
        elif stable_phase == "end" and self.sequence_state == "MIDDLE":
            # Include the end-state confidence and the Admin expert rules before finalizing.
            self.classifier_confidences.append(stable_conf)
            self._capture_admin_rules(kp, conf, "end")
            self._complete_generic_rep(frame)

        vietchu(annotated, f"BAI TAP HOC: {self.slug.upper()}", 30, 45, (255, 255, 255), 0.75, 2)
        vietchu(annotated, f"PHA: {self.trangthai}", 30, 85, (255, 255, 0), 0.72, 2)
        if label:
            vietchu(annotated, f"MODEL: {label} ({int(round(pred_conf * 100))}%)", 30, 125, (180, 220, 255), 0.62, 2)
        return self.finish(annotated)


class UnsupportedExerciseProcessor(BaseProcessor):
    """Safe fallback: an unknown exercise is never analyzed as squat."""

    def __init__(self, slug="unknown", shared_state=None):
        super().__init__(shared_state, load_model=False)
        self.slug = str(slug)
        self.trangthai = "UNSUPPORTED"

    def process(self, frame):
        annotated = frame.copy()
        vietchu(annotated, "BAI TAP CHUA CO BO PHAN TICH AI", 30, 70, (0, 165, 255), 0.85, 2)
        vietchu(annotated, f"SLUG: {self.slug}", 30, 115, (255, 255, 255), 0.75, 2)
        self.shared_state["feedback_text"] = "Bài tập chưa có bộ phân tích AI."
        self.shared_state["feedback_advice"] = "Vui lòng chọn Squat, Push-up hoặc Bicep Curl ở phiên bản tuần 6."
        return self.finish(annotated)
