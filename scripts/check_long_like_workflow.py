"""Static preflight for the Lê Phi Long-like admin workflows."""
from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)
    print(f"[PASS] {message}")


def text(relative: str) -> str:
    return (ROOT / relative).read_text(encoding="utf-8")


def main() -> int:
    viewer = text("static/exercise_model_viewer.js")
    admin = text("templates/admin_exercise_detail.html")
    workout = text("templates/user_workout.html")
    app = text("app.py")
    logic = text("workoutlogic.py")
    tts = text("tts_service.py")

    require("FBXLoader" in viewer and "AnimationMixer" in viewer, "Viewer tải FBX và chạy animation thật")
    require("buildProceduralHumanoid" not in viewer and "animateProcedural" not in viewer, "Không còn procedural humanoid fallback")
    require("Huấn luyện mô hình trên web" in admin and "/train-model" in app and "/training-status" in app, "Admin train model trực tiếp trên web")
    require('f"{slug}_start"' in app and 'f"{slug}_middle"' in app and 'f"{slug}_end"' in app, "Bài tập mới có start/middle/end riêng")
    require("LearnedExerciseProcessor" in logic and "admin_criteria_rules" in logic, "Processor bài tập mới đọc model và luật Admin")
    require('name="joint_a"' in admin and 'name="joint_b"' in admin and 'name="joint_c"' in admin, "Admin cấu hình luật góc bằng 3 keypoint A-B-C")
    require("synthesize_feedback" in app and "setup_piper_vi.py" in text("docs/CAI_PIPER_TTS_WINDOWS.md"), "Tiêu chí tự sinh WAV và có hướng dẫn Piper")
    require("feedback_audio_url" in app and "generatedFeedbackAudio" in workout, "Workout ưu tiên phát WAV đã sinh theo error_code")
    require("vi_VN-vais1000-medium" in tts and '"-m"' in tts and '"-f"' in tts, "TTS hỗ trợ Piper CLI hiện hành với voice tiếng Việt")

    print("\nLONG-LIKE WORKFLOW PREFLIGHT: PASS")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"[FAIL] {exc}")
        raise
