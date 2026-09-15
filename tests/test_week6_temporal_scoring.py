from __future__ import annotations

import unittest

from ai_temporal import TemporalBank, TemporalSignal, movement_stability
from quality_scoring import score_curl, score_pushup, score_squat


class TemporalProcessingTests(unittest.TestCase):
    def test_ema_reduces_single_frame_noise(self):
        signal = TemporalSignal(alpha=0.30, window_size=10)
        for value in [170, 170, 170, 100, 170]:
            filtered = signal.update(value)
        self.assertGreater(filtered, 140)
        self.assertLess(filtered, 170)

    def test_hysteresis_requires_consecutive_frames(self):
        bank = TemporalBank()
        self.assertFalse(bank.stable("enter", True, frames=3))
        self.assertFalse(bank.stable("enter", False, frames=3))
        self.assertFalse(bank.stable("enter", True, frames=3))
        self.assertFalse(bank.stable("enter", True, frames=3))
        self.assertTrue(bank.stable("enter", True, frames=3))

    def test_stability_penalizes_jitter(self):
        smooth = movement_stability([170, 160, 150, 140, 130, 120, 110])
        jitter = movement_stability([170, 130, 165, 120, 160, 110, 155])
        self.assertGreater(smooth, jitter)

    def test_signal_trend_tracks_direction(self):
        signal = TemporalSignal(alpha=1.0, window_size=6)
        for value in [170, 160, 150, 140]:
            signal.update(value)
        self.assertLess(signal.trend, 0)
        for value in [150, 160, 170, 175]:
            signal.update(value)
        self.assertGreater(signal.trend, 0)


class QualityScoringTests(unittest.TestCase):
    def test_good_squat_scores_higher_than_shallow_squat(self):
        good = score_squat(92, 165, 0.92, 90)
        bad = score_squat(136, 125, 0.55, 45)
        self.assertGreaterEqual(good.total, 80)
        self.assertLess(bad.total, 55)
        self.assertIn("depth", good.components)
        self.assertAlmostEqual(sum(good.weights.values()), 1.0)

    def test_good_pushup_scores_higher_than_bad_alignment(self):
        good = score_pushup(82, 174, 0.92, 90)
        bad = score_pushup(126, 130, 0.55, 45)
        self.assertGreaterEqual(good.total, 80)
        self.assertLess(bad.total, 55)

    def test_good_curl_scores_higher_than_incomplete_curl(self):
        good = score_curl(58, 165, 10, 0.91, 90)
        bad = score_curl(115, 128, 65, 0.55, 45)
        self.assertGreaterEqual(good.total, 80)
        self.assertLess(bad.total, 55)
        self.assertIn("elbow_control", good.components)

    def test_low_confidence_reduces_score(self):
        high = score_squat(95, 165, 0.95, 90)
        low = score_squat(95, 165, 0.35, 90)
        self.assertGreater(high.total, low.total)


if __name__ == "__main__":
    unittest.main()
