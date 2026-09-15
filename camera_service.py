"""Shared, reconnecting webcam capture for the local FitMotion demo.

The old source opened ``cv2.VideoCapture`` inside every MJPEG request. Reloading
the page or creating a second request could leave the webcam locked and produce
an interrupted stream. This service owns one capture device per Python process,
reads frames in a background thread and reconnects after transient failures.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass(frozen=True)
class CameraSnapshot:
    sequence: int
    frame: object | None
    ready: bool
    error: str
    backend: str
    last_frame_age: float


class CameraService:
    def __init__(self, width: int = 960, height: int = 540, target_fps: float = 20.0):
        self.width = int(width)
        self.height = int(height)
        self.target_fps = max(1.0, float(target_fps))

        self._lock = threading.RLock()
        self._condition = threading.Condition(self._lock)
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._restart_event = threading.Event()

        self._capture = None
        self._frame = None
        self._sequence = 0
        self._error = "Camera chưa khởi động."
        self._backend = ""
        self._last_frame_at = 0.0
        self._open_attempts = 0

    def ensure_started(self) -> None:
        with self._lock:
            if self._thread and self._thread.is_alive():
                return
            self._stop_event.clear()
            self._thread = threading.Thread(
                target=self._capture_loop,
                name="fitmotion-camera",
                daemon=True,
            )
            self._thread.start()

    def _release_capture(self) -> None:
        cap = None
        with self._lock:
            cap, self._capture = self._capture, None
        if cap is not None:
            try:
                cap.release()
            except Exception:
                pass

    def _open_capture(self):
        import cv2

        candidates = [
            (0, cv2.CAP_DSHOW, "camera-0/dshow"),
            (1, cv2.CAP_DSHOW, "camera-1/dshow"),
            (0, cv2.CAP_ANY, "camera-0/auto"),
            (1, cv2.CAP_ANY, "camera-1/auto"),
        ]

        for index, backend, label in candidates:
            cap = None
            try:
                cap = cv2.VideoCapture(index, backend)
                if cap is not None and cap.isOpened():
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                    try:
                        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    except Exception:
                        pass
                    with self._lock:
                        self._capture = cap
                        self._backend = label
                        self._error = ""
                        self._open_attempts = 0
                    return cap
            except Exception as exc:
                with self._lock:
                    self._error = f"Lỗi mở {label}: {exc}"
            finally:
                if cap is not None and cap is not self._capture:
                    try:
                        cap.release()
                    except Exception:
                        pass

        with self._lock:
            self._open_attempts += 1
            self._error = (
                "Không mở được webcam. Hãy đóng Zoom/Teams/Camera, "
                "kiểm tra quyền camera rồi bấm Kết nối lại."
            )
        return None

    def _capture_loop(self) -> None:
        frame_period = 1.0 / self.target_fps
        consecutive_failures = 0

        while not self._stop_event.is_set():
            if self._restart_event.is_set():
                self._restart_event.clear()
                self._release_capture()
                with self._condition:
                    self._frame = None
                    self._error = "Đang kết nối lại camera..."
                    self._condition.notify_all()

            with self._lock:
                cap = self._capture

            if cap is None or not cap.isOpened():
                cap = self._open_capture()
                if cap is None:
                    time.sleep(min(2.0, 0.35 + self._open_attempts * 0.15))
                    continue

            started = time.monotonic()
            try:
                success, frame = cap.read()
            except Exception as exc:
                success, frame = False, None
                with self._lock:
                    self._error = f"Lỗi đọc webcam: {exc}"

            if not success or frame is None:
                consecutive_failures += 1
                with self._condition:
                    self._error = "Camera tạm ngắt; hệ thống đang tự kết nối lại."
                    self._condition.notify_all()
                if consecutive_failures >= 5:
                    self._release_capture()
                    consecutive_failures = 0
                time.sleep(0.10)
                continue

            consecutive_failures = 0
            with self._condition:
                self._frame = frame
                self._sequence += 1
                self._last_frame_at = time.monotonic()
                self._error = ""
                self._condition.notify_all()

            elapsed = time.monotonic() - started
            if elapsed < frame_period:
                time.sleep(frame_period - elapsed)

        self._release_capture()

    def get_frame(
        self,
        last_sequence: int = -1,
        timeout: float = 1.5,
    ) -> CameraSnapshot:
        self.ensure_started()
        deadline = time.monotonic() + max(0.05, float(timeout))

        with self._condition:
            while (
                self._sequence == last_sequence
                and time.monotonic() < deadline
                and not self._stop_event.is_set()
            ):
                self._condition.wait(timeout=max(0.01, deadline - time.monotonic()))

            frame = self._frame.copy() if self._frame is not None else None
            age = (
                max(0.0, time.monotonic() - self._last_frame_at)
                if self._last_frame_at
                else 999.0
            )
            ready = frame is not None and age < 2.5
            return CameraSnapshot(
                sequence=int(self._sequence),
                frame=frame,
                ready=bool(ready),
                error=str(self._error),
                backend=str(self._backend),
                last_frame_age=float(age),
            )

    def status(self) -> dict:
        snapshot = self.get_frame(timeout=0.05)
        return {
            "ready": snapshot.ready,
            "error": snapshot.error,
            "backend": snapshot.backend,
            "sequence": snapshot.sequence,
            "last_frame_age": round(snapshot.last_frame_age, 3),
        }

    def restart(self) -> None:
        self.ensure_started()
        self._restart_event.set()

    def stop(self) -> None:
        self._stop_event.set()
        with self._condition:
            self._condition.notify_all()
        thread = self._thread
        if thread and thread.is_alive():
            thread.join(timeout=1.0)
        self._release_capture()


camera_service = CameraService()
