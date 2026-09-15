"""Fall-and-immobility monitoring for FitMotion AI.

The monitor is designed for workout use, where Squat and Push-up naturally put
people low or horizontal. It therefore never alarms from posture alone. A real
alert requires an observed transition from normal activity into a collapse-like
lying posture, followed by continuous immobility for a configurable 4-10 seconds.

This is a safety aid for the thesis demo, not a medical device.
"""
from __future__ import annotations

import math
import os
import time
from typing import Mapping

import cv2


class EmergencyMonitor:
    DEFAULT_IMMOBILITY_SECONDS = 6.0
    MAX_OBSERVATION_GAP_SECONDS = 5.0

    # Motion ratios are normalized by frame height.
    ACTIVE_MOTION_RATIO = 0.014
    IMMOBILE_MOTION_RATIO = 0.011
    RECOVERY_MOTION_RATIO = 0.035
    RAPID_SHOULDER_DROP_RATIO = 0.075
    RAPID_HIP_DROP_RATIO = 0.055
    VERY_LARGE_DROP_RATIO = 0.115

    UPRIGHT_ANGLE_MAX = 45.0
    LYING_ANGLE_MIN = 55.0
    LYING_VERTICAL_SPAN_MAX = 0.36
    LYING_SHOULDER_MIN_Y = 0.32

    NORMAL_ACTIVITY_MEMORY_SECONDS = 15.0
    CANDIDATE_SETTLE_SECONDS = 1.5
    PUSHUP_COLLAPSE_DROP_RATIO = 0.10

    def __init__(
        self,
        shared_state=None,
        save_dir=None,
        immobility_seconds: float = DEFAULT_IMMOBILITY_SECONDS,
    ):
        self.shared_state = shared_state if shared_state is not None else {}
        if save_dir is None:
            save_dir = os.getenv("FITMOTION_EMERGENCY_DIR", "").strip() or os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "data", "errorimages", "emergency"
            )
        self.save_dir = os.path.abspath(save_dir)
        os.makedirs(self.save_dir, exist_ok=True)

        self.immobility_seconds = min(10.0, max(4.0, float(immobility_seconds)))
        self.prev_points = None
        self.prev_ts = None
        self.prev_lying = False
        self.last_normal_activity_ts = 0.0
        self.last_upright_ts = 0.0
        self.last_recovery_ts = 0.0
        self.tracking_lost_ts = None
        self.collapse_candidate_ts = None
        self.candidate_last_motion_ts = None
        self.last_saved_ts = 0.0
        self.armed = False
        self.last_reset_version = int(self.shared_state.get("emergency_reset_version", 0) or 0)

        self.required_idx = [5, 6, 11, 12, 15, 16]

        if "active" not in self.shared_state:
            self.reset_shared_state()
        else:
            self._publish_debug_state("unknown", False, 0.0, False)

    def reset_shared_state(self):
        self.shared_state.update({
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
            "emergency_required_seconds": float(self.immobility_seconds),
            "emergency_exercise": "",
            "emergency_phase": "",
            "emergency_reset_version": 0,
            "workout_paused": False,
            "pause_reason": "",
        })

    def reset_runtime(self) -> None:
        self.prev_points = None
        self.prev_ts = None
        self.prev_lying = False
        self.last_normal_activity_ts = 0.0
        self.last_upright_ts = 0.0
        self.last_recovery_ts = 0.0
        self.tracking_lost_ts = None
        self.collapse_candidate_ts = None
        self.candidate_last_motion_ts = None
        self.armed = False
        self._publish_debug_state("unknown", False, 0.0, False)

    def _publish_debug_state(
        self,
        posture: str,
        candidate: bool,
        immobile_for: float,
        armed: bool,
        context: Mapping | None = None,
    ) -> None:
        context = context or {}
        self.shared_state.update({
            "emergency_posture": str(posture),
            "emergency_candidate": bool(candidate),
            "emergency_armed": bool(armed),
            "emergency_immobile_seconds": round(max(0.0, float(immobile_for)), 1),
            "emergency_required_seconds": float(self.immobility_seconds),
            "emergency_exercise": str(context.get("exercise", "") or ""),
            "emergency_phase": str(context.get("phase", "") or ""),
        })

    def _cancel_candidate(self, posture="unknown", context=None) -> None:
        self.collapse_candidate_ts = None
        self.candidate_last_motion_ts = None
        self._publish_debug_state(posture, False, 0.0, self.armed, context)

    @staticmethod
    def _valid_point(conf, idx, threshold=0.22):
        if idx >= len(conf):
            return False
        return bool(float(conf[idx]) >= threshold)

    @staticmethod
    def _midpoint(p1, p2):
        return (
            float((p1[0] + p2[0]) / 2.0),
            float((p1[1] + p2[1]) / 2.0),
        )

    @staticmethod
    def _body_angle_from_vertical(shoulder_mid, hip_mid):
        dx = float(shoulder_mid[0] - hip_mid[0])
        dy = float(shoulder_mid[1] - hip_mid[1])
        if abs(dx) < 1e-6 and abs(dy) < 1e-6:
            return 0.0
        return float(math.degrees(math.atan2(abs(dx), abs(dy) + 1e-6)))

    @staticmethod
    def _avg_motion(current_pts, prev_pts):
        total = 0.0
        count = 0
        for name in ("shoulder", "hip", "ankle", "knee"):
            if name in current_pts and name in prev_pts:
                total += abs(float(current_pts[name][0]) - float(prev_pts[name][0]))
                total += abs(float(current_pts[name][1]) - float(prev_pts[name][1]))
                count += 1
        return float(total / max(count, 1))

    @staticmethod
    def _posture_name(upright: bool, lying: bool, low_posture: bool) -> str:
        if lying:
            return "lying"
        if upright:
            return "upright"
        if low_posture:
            return "low_exercise"
        return "other"

    def _save_emergency_frame(self, frame):
        ts = time.strftime("%Y%m%d_%H%M%S")
        full_path = os.path.join(self.save_dir, f"emergency_{ts}.jpg")
        cv2.imwrite(full_path, frame)
        return full_path.replace("\\", "/")

    def trigger(self, reason, raw_frame=None):
        if bool(self.shared_state.get("active", False)):
            return

        image_path = ""
        now = float(time.time())
        if raw_frame is not None and (now - float(self.last_saved_ts)) > 3:
            image_path = self._save_emergency_frame(raw_frame)
            self.last_saved_ts = now

        # A raised emergency pauses AI inference and rep counting immediately.
        self.shared_state.update({
            "active": True,
            "message": "CẢNH BÁO KHẨN CẤP",
            "reason": str(reason),
            "updated_at": now,
            "image_path": str(image_path),
            "low_posture": True,
            "emergency_candidate": False,
            "emergency_immobile_seconds": float(self.immobility_seconds),
            "workout_paused": True,
            "pause_reason": "emergency",
        })

    def update(self, kp, conf, frame_shape, raw_frame=None, context=None):
        context = context or {}
        exercise = str(context.get("exercise", "") or "").lower()
        phase = str(context.get("phase", "") or "").upper()

        reset_version = int(self.shared_state.get("emergency_reset_version", 0) or 0)
        if reset_version != self.last_reset_version:
            self.last_reset_version = reset_version
            self.reset_runtime()
        if bool(self.shared_state.get("active", False)):
            return self.shared_state

        has_left_shoulder = self._valid_point(conf, 5)
        has_right_shoulder = self._valid_point(conf, 6)
        has_left_hip = self._valid_point(conf, 11)
        has_right_hip = self._valid_point(conf, 12)

        now = float(time.time())
        has_torso = (has_left_shoulder or has_right_shoulder) and (has_left_hip or has_right_hip)
        if not has_torso:
            if self.collapse_candidate_ts is not None:
                if self.tracking_lost_ts is None:
                    self.tracking_lost_ts = now
                elif (now - self.tracking_lost_ts) > 1.5:
                    self._cancel_candidate("tracking_lost", context)
                    self.tracking_lost_ts = None
                    self.prev_points = None
                    self.prev_ts = None
            else:
                self._cancel_candidate("tracking_lost", context)
                self.prev_points = None
                self.prev_ts = None
            return self.shared_state

        self.tracking_lost_ts = None
        h = max(float(frame_shape[0]), 1.0)

        if has_left_shoulder and has_right_shoulder:
            shoulder_mid = self._midpoint(kp[5], kp[6])
        else:
            shoulder_mid = kp[5] if has_left_shoulder else kp[6]

        if has_left_hip and has_right_hip:
            hip_mid = self._midpoint(kp[11], kp[12])
        else:
            hip_mid = kp[11] if has_left_hip else kp[12]

        has_ankles = self._valid_point(conf, 15) and self._valid_point(conf, 16)
        has_knees = self._valid_point(conf, 13) and self._valid_point(conf, 14)

        current_pts = {"shoulder": shoulder_mid, "hip": hip_mid}
        if has_ankles:
            ankle_mid = self._midpoint(kp[15], kp[16])
            current_pts["ankle"] = ankle_mid
        elif has_knees:
            knee_mid = self._midpoint(kp[13], kp[14])
            current_pts["knee"] = knee_mid

        body_angle = float(self._body_angle_from_vertical(shoulder_mid, hip_mid))
        shoulder_y_ratio = float(shoulder_mid[1]) / h
        hip_y_ratio = float(hip_mid[1]) / h

        # Upright: body mostly vertical, shoulder above hip, not too low
        upright = bool(
            body_angle <= 42.0
            and shoulder_mid[1] < hip_mid[1]
            and shoulder_y_ratio < 0.52
        )

        # Lying or collapsed: horizontal or both points very low in frame
        lying = bool(
            body_angle >= 55.0
            or (shoulder_y_ratio >= 0.55 and hip_y_ratio >= 0.55)
            or (shoulder_y_ratio >= 0.50 and hip_y_ratio >= 0.50 and body_angle >= 40.0)
        )

        low_posture = bool(hip_y_ratio > 0.68 or shoulder_y_ratio > 0.52)

        self.shared_state["body_angle"] = round(body_angle, 1)
        self.shared_state["low_posture"] = bool(low_posture)

        # Init last_motion_ts on first frame
        if not hasattr(self, "last_motion_ts") or self.last_motion_ts == 0.0:
            self.last_motion_ts = now

        if self.prev_points is None or self.prev_ts is None:
            self.prev_points = current_pts
            self.prev_ts = now
            self.prev_lying = lying
            self.last_motion_ts = now
            if upright:
                self.armed = True
                self.last_normal_activity_ts = now
            self._publish_debug_state(
                self._posture_name(upright, lying, low_posture), False, 0.0, self.armed, context
            )
            return self.shared_state

        dt = max(now - float(self.prev_ts), 1e-3)
        if dt > self.MAX_OBSERVATION_GAP_SECONDS:
            self._cancel_candidate("gap", context)
            self.prev_points = current_pts
            self.prev_ts = now
            self.prev_lying = lying
            self.last_motion_ts = now
            return self.shared_state

        avg_motion = float(self._avg_motion(current_pts, self.prev_points))
        avg_motion_ratio = avg_motion / h

        hip_drop = float(current_pts["hip"][1]) - float(self.prev_points["hip"][1])
        shoulder_drop = float(current_pts["shoulder"][1]) - float(self.prev_points["shoulder"][1])

        # Arm the monitor whenever person is upright or actively moving
        if upright:
            self.armed = True
            self.last_normal_activity_ts = now
            self.last_motion_ts = now
        elif avg_motion_ratio >= self.ACTIVE_MOTION_RATIO and not lying:
            self.armed = True
            self.last_normal_activity_ts = now
            self.last_motion_ts = now

        # Push-up is horizontal by design - arm on active motion
        if exercise == "pushup" and avg_motion_ratio >= self.ACTIVE_MOTION_RATIO:
            self.armed = True
            self.last_normal_activity_ts = now

        # Reset immobility clock when there is real motion (not while lying)
        significant_motion = avg_motion > 18 or hip_drop > 20 or shoulder_drop > 20
        if significant_motion and not lying:
            self.last_motion_ts = now

        posture = self._posture_name(upright, lying, low_posture)

        # Detect rapid fall
        rapid_drop = bool(
            (hip_drop / h) >= self.RAPID_HIP_DROP_RATIO
            or (shoulder_drop / h) >= self.RAPID_SHOULDER_DROP_RATIO
            or (hip_drop / dt) > 180
            or (shoulder_drop / dt) > 180
        )

        transitioned_to_lying = bool(lying and not self.prev_lying)
        recent_recovery = bool(
            self.last_recovery_ts and (now - self.last_recovery_ts) <= 20.0
        )

        # Start a collapse candidate:
        # - On rapid drop into lying
        # - On lying down on the floor (any posture, without requiring previous standing)
        # - On being low/horizontal and immobile >= 1.0s
        if self.collapse_candidate_ts is None and not recent_recovery:
            if rapid_drop and lying:
                self.collapse_candidate_ts = now
                self.candidate_last_motion_ts = now
                self.last_motion_ts = now
            elif lying:
                self.collapse_candidate_ts = now
                self.candidate_last_motion_ts = now
            elif low_posture and body_angle >= 45.0:
                elapsed_immobile = now - float(self.last_motion_ts or now)
                if elapsed_immobile >= 1.0:
                    self.collapse_candidate_ts = now
                    self.candidate_last_motion_ts = now

        if self.collapse_candidate_ts is not None:
            if upright:
                self.last_recovery_ts = now
                self._cancel_candidate(posture, context)
            elif avg_motion_ratio >= self.RECOVERY_MOTION_RATIO:
                elapsed = now - float(self.collapse_candidate_ts)
                if elapsed <= self.CANDIDATE_SETTLE_SECONDS:
                    self.candidate_last_motion_ts = now
                    self._publish_debug_state(posture, True, 0.0, self.armed, context)
                else:
                    self.last_recovery_ts = now
                    self._cancel_candidate(posture, context)
            else:
                start = float(self.candidate_last_motion_ts or self.collapse_candidate_ts)
                immobile_for_candidate = max(0.0, now - start)
                self._publish_debug_state(posture, True, immobile_for_candidate, self.armed, context)
                if immobile_for_candidate >= self.immobility_seconds:
                    self.trigger(
                        "Phát hiện người tập ngã xuống tư thế nằm và bất động liên tục "
                        f"trong {int(self.immobility_seconds)} giây",
                        raw_frame,
                    )
        else:
            self._publish_debug_state(posture, False, 0.0, self.armed, context)

        self.prev_points = current_pts
        self.prev_ts = now
        self.prev_lying = lying
        return self.shared_state
