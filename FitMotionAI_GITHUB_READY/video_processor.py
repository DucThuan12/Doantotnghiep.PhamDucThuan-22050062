"""Uploaded-video evaluation using the canonical realtime processors.

This service intentionally reuses ``workoutlogic`` instead of maintaining a
second set of thresholds. The returned JSON includes auditable per-rep evidence
and the same quality/confidence fields used by live webcam sessions.
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

import cv2
from flask import Flask, jsonify, request

from standalone_runner import create_processor
from workoutlogic import UnsupportedExerciseProcessor
from training_levels import DEFAULT_LEVEL, normalize_training_level, training_level_label

app = Flask(__name__)
SUPPORTED = {"squat", "pushup", "curl", "curl-left", "curl-right"}


def process_video_file(video_path: str, exercise_slug: str, training_level: str = DEFAULT_LEVEL) -> dict:
    normalized_level = normalize_training_level(training_level)
    state: dict = {"training_level": normalized_level}
    processor = create_processor(exercise_slug, state)
    if isinstance(processor, UnsupportedExerciseProcessor):
        raise ValueError(f"Bài tập không được hỗ trợ: {exercise_slug}")

    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        raise ValueError("Không thể mở file video")

    frame_count = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            processor.process(frame)
            frame_count += 1
    finally:
        capture.release()

    return {
        "exercise": exercise_slug,
        "training_level": normalized_level,
        "training_level_label": training_level_label(normalized_level),
        "processed_frames": frame_count,
        "total_reps": int(state.get("total_rep", 0) or 0),
        "correct_reps": int(state.get("good_rep", 0) or 0),
        "incorrect_reps": int(state.get("bad_rep", 0) or 0),
        "quality_score_avg": float(state.get("quality_score_avg", 0) or 0),
        "keypoint_confidence_avg": float(state.get("keypoint_confidence_avg", 0) or 0),
        "rep_records": state.get("rep_records", []),
        "error_code_counts": state.get("error_code_counts", {}),
        "tracking_abort_count": int(state.get("tracking_abort_count", 0) or 0),
    }


@app.post("/process_video")
def process_video():
    upload = request.files.get("video")
    exercise = str(request.form.get("exercise", "")).strip().lower()
    training_level = normalize_training_level(request.form.get("training_level", DEFAULT_LEVEL))
    if upload is None or not upload.filename:
        return jsonify({"success": False, "error": "Thiếu file video"}), 400
    if exercise not in SUPPORTED:
        return jsonify({"success": False, "error": f"Bài tập không được hỗ trợ: {exercise}"}), 400

    suffix = Path(upload.filename).suffix.lower() or ".mp4"
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(prefix="fitmotion_", suffix=suffix, delete=False) as temp_file:
            temp_path = temp_file.name
        upload.save(temp_path)
        result = process_video_file(temp_path, exercise, training_level=training_level)
        return jsonify({"success": True, **result})
    except ValueError as exc:
        return jsonify({"success": False, "error": str(exc)}), 422
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5001, debug=True)
