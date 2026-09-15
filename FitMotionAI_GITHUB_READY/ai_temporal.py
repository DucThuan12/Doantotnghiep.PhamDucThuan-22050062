"""Temporal processing utilities for FitMotion AI.

The module intentionally uses lightweight filters so the real-time webcam
pipeline can run on a normal laptop. It provides:
- exponential moving average (EMA) smoothing;
- short rolling histories and motion trend;
- consecutive-frame hysteresis counters;
- a per-repetition stability score in the 0-100 range.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, Iterable

import numpy as np


def clamp(value: float, low: float = 0.0, high: float = 100.0) -> float:
    return max(low, min(high, float(value)))


def movement_stability(values: Iterable[float], acceleration_scale: float = 4.0) -> float:
    """Return a 0-100 stability score for one repetition.

    The score is based on second-order changes of the smoothed joint-angle
    sequence. A controlled movement has smaller abrupt acceleration changes and
    therefore receives a higher score. Fewer than four samples are not enough
    to evaluate jitter reliably and return a neutral score of 100.
    """
    data = np.asarray(list(values), dtype=np.float32)
    if data.size < 4:
        return 100.0

    acceleration = np.diff(data, n=2)
    mean_abs_acceleration = float(np.mean(np.abs(acceleration)))
    penalty = mean_abs_acceleration * float(acceleration_scale)
    return clamp(100.0 - penalty)


@dataclass
class TemporalSignal:
    """EMA-smoothed scalar signal with a bounded rolling history."""

    alpha: float = 0.35
    window_size: int = 10
    value: float | None = None
    history: Deque[float] = field(default_factory=deque)

    def __post_init__(self) -> None:
        self.alpha = float(self.alpha) / 100.0 if float(self.alpha) > 1.0 else float(self.alpha)
        self.alpha = max(0.01, min(1.0, self.alpha))
        self.window_size = max(4, int(self.window_size))
        self.history = deque(maxlen=self.window_size)

    def update(self, raw_value: float) -> float:
        raw_value = float(raw_value)
        if not np.isfinite(raw_value):
            if self.value is None:
                raise ValueError("TemporalSignal received a non-finite first value")
            return float(self.value)

        if self.value is None:
            self.value = raw_value
        else:
            self.value = self.alpha * raw_value + (1.0 - self.alpha) * self.value

        self.history.append(float(self.value))
        return float(self.value)

    @property
    def trend(self) -> float:
        """Average per-frame change over the recent smoothed samples."""
        if len(self.history) < 3:
            return 0.0
        values = np.asarray(self.history, dtype=np.float32)
        return float(np.mean(np.diff(values)))

    def reset(self) -> None:
        self.value = None
        self.history.clear()


class TemporalBank:
    """Named temporal signals and consecutive-condition counters."""

    def __init__(self, alpha: float = 0.35, window_size: int = 10) -> None:
        self.alpha = float(alpha)
        self.window_size = int(window_size)
        self.signals: Dict[str, TemporalSignal] = {}
        self.counters: Dict[str, int] = {}

    def update(self, name: str, value: float) -> float:
        signal = self.signals.get(name)
        if signal is None:
            signal = TemporalSignal(alpha=self.alpha, window_size=self.window_size)
            self.signals[name] = signal
        return signal.update(value)

    def trend(self, name: str) -> float:
        signal = self.signals.get(name)
        return signal.trend if signal else 0.0

    def stable(self, key: str, condition: bool, frames: int = 3) -> bool:
        """Apply frame-based hysteresis to a boolean condition."""
        required = max(1, int(frames))
        self.counters[key] = self.counters.get(key, 0) + 1 if condition else 0
        return self.counters[key] >= required

    def reset_counter(self, key: str) -> None:
        self.counters[key] = 0

    def reset_counters(self, *keys: str) -> None:
        if keys:
            for key in keys:
                self.counters[key] = 0
        else:
            self.counters.clear()

    def reset_signal(self, name: str) -> None:
        signal = self.signals.get(name)
        if signal is not None:
            signal.reset()

    def reset_all(self) -> None:
        for signal in self.signals.values():
            signal.reset()
        self.counters.clear()
