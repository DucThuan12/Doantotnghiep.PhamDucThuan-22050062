"""Standalone webcam runner using the same processors as the Flask application."""
from __future__ import annotations

import cv2

from workoutlogic import CurlProcessor, PushupProcessor, SquatProcessor, UnsupportedExerciseProcessor


def create_processor(slug: str, shared_state: dict | None = None):
    slug = str(slug or "").strip().lower()
    if slug == "squat":
        return SquatProcessor(shared_state=shared_state)
    if slug == "pushup":
        return PushupProcessor(shared_state=shared_state)
    if slug in {"curl", "curl-left"}:
        return CurlProcessor("left", shared_state=shared_state)
    if slug == "curl-right":
        return CurlProcessor("right", shared_state=shared_state)
    return UnsupportedExerciseProcessor(slug, shared_state=shared_state)


def run_camera(slug: str, camera_index: int = 0) -> dict:
    state: dict = {}
    processor = create_processor(slug, state)
    if isinstance(processor, UnsupportedExerciseProcessor):
        raise ValueError(f"Bài tập chưa có bộ phân tích AI: {slug}")

    camera = cv2.VideoCapture(int(camera_index))
    if not camera.isOpened():
        raise RuntimeError(f"Không mở được camera index {camera_index}")

    window_name = f"FitMotion AI - {slug}"
    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                raise RuntimeError("Camera không trả về frame hợp lệ")
            output = processor.process(frame)
            cv2.imshow(window_name, output)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()
    return state
