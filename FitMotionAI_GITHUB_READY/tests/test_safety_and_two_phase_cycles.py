from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np

from emergency import EmergencyMonitor


class _Clock:
    def __init__(self, start=1000.0):
        self.now = float(start)

    def time(self):
        return self.now

    def advance(self, seconds):
        self.now += float(seconds)


def _pose(shoulder_left, shoulder_right, hip_left, hip_right, ankle_left, ankle_right):
    kp = np.zeros((17, 2), dtype=float)
    kp[5], kp[6] = shoulder_left, shoulder_right
    kp[11], kp[12] = hip_left, hip_right
    kp[15], kp[16] = ankle_left, ankle_right
    conf = np.ones(17, dtype=float)
    return kp, conf


STANDING = _pose(
    (280, 90), (360, 90),
    (290, 220), (350, 220),
    (295, 430), (345, 430),
)

SQUAT_LOW = _pose(
    (280, 200), (360, 200),
    (285, 325), (355, 325),
    (295, 430), (345, 430),
)

PUSHUP_TOP = _pose(
    (120, 270), (120, 310),
    (310, 280), (310, 320),
    (520, 290), (520, 330),
)

PUSHUP_BOTTOM = _pose(
    (120, 300), (120, 340),
    (310, 310), (310, 350),
    (520, 320), (520, 360),
)

FALLEN = _pose(
    (120, 350), (120, 390),
    (310, 360), (310, 400),
    (520, 370), (520, 410),
)


class EmergencySafetyTests(unittest.TestCase):
    def _update(self, monitor, clock, pose, context=None, advance=0.2):
        clock.advance(advance)
        kp, conf = pose
        return monitor.update(
            kp,
            conf,
            (480, 640, 3),
            raw_frame=None,
            context=context or {},
        )

    def test_squat_low_posture_never_triggers_emergency(self):
        clock = _Clock()
        state = {}
        with patch("emergency.time.time", side_effect=clock.time):
            monitor = EmergencyMonitor(state, immobility_seconds=6)
            for _ in range(8):
                self._update(monitor, clock, STANDING, {"exercise": "squat", "phase": "STANDING"})
            for _ in range(20):
                self._update(monitor, clock, SQUAT_LOW, {"exercise": "squat", "phase": "BOTTOM"}, advance=1.0)
        self.assertFalse(state["active"])
        self.assertFalse(state["emergency_candidate"])
        self.assertNotEqual(state["emergency_posture"], "lying")

    def test_normal_pushup_motion_never_triggers_emergency(self):
        clock = _Clock()
        state = {}
        with patch("emergency.time.time", side_effect=clock.time):
            monitor = EmergencyMonitor(state, immobility_seconds=6)
            for _ in range(6):
                self._update(monitor, clock, PUSHUP_TOP, {"exercise": "pushup", "phase": "PLANK_UP"})
            for _ in range(15):
                self._update(monitor, clock, PUSHUP_BOTTOM, {"exercise": "pushup", "phase": "BOTTOM"}, advance=0.5)
                self._update(monitor, clock, PUSHUP_TOP, {"exercise": "pushup", "phase": "PLANK_UP"}, advance=0.5)
        self.assertFalse(state["active"])
        self.assertFalse(state["emergency_candidate"])

    def test_fall_requires_full_six_seconds_of_continuous_immobility(self):
        clock = _Clock()
        state = {}
        with patch("emergency.time.time", side_effect=clock.time):
            monitor = EmergencyMonitor(state, immobility_seconds=6)
            for _ in range(8):
                self._update(monitor, clock, STANDING, {"exercise": "squat", "phase": "STANDING"})
            self._update(monitor, clock, FALLEN, {"exercise": "squat", "phase": "DESCENDING"})
            self.assertTrue(state["emergency_candidate"])
            for _ in range(5):
                self._update(monitor, clock, FALLEN, {"exercise": "squat", "phase": "DESCENDING"}, advance=1.0)
            self.assertFalse(state["active"])
            self._update(monitor, clock, FALLEN, {"exercise": "squat", "phase": "DESCENDING"}, advance=1.1)
        self.assertTrue(state["active"])
        self.assertIn("6 giây", state["reason"])

    def test_interaction_or_recovery_cancels_pending_alert(self):
        clock = _Clock()
        state = {}
        with patch("emergency.time.time", side_effect=clock.time):
            monitor = EmergencyMonitor(state, immobility_seconds=6)
            for _ in range(8):
                self._update(monitor, clock, STANDING)
            self._update(monitor, clock, FALLEN)
            for _ in range(5):
                self._update(monitor, clock, FALLEN, advance=1.0)
            # Meaningful movement/interaction with the camera cancels the timer.
            moved = _pose(
                (170, 350), (170, 390),
                (360, 360), (360, 400),
                (570, 370), (570, 410),
            )
            self._update(monitor, clock, moved, advance=0.5)
            self.assertFalse(state["emergency_candidate"])
            for _ in range(15):
                self._update(monitor, clock, moved, advance=1.0)
        self.assertFalse(state["active"])


if __name__ == "__main__":
    unittest.main()
