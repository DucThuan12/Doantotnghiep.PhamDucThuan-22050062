"""One-time Piper Vietnamese setup for FitMotion AI.

Run inside the project's activated virtual environment:
    python scripts/setup_piper_vi.py

The script installs piper-tts and downloads the Vietnamese voice into
models/piper.  Once complete, Admin -> Add criterion can generate WAV files
without extra environment variables.
"""
from __future__ import annotations

from pathlib import Path
import subprocess
import sys

VOICE = "vi_VN-vais1000-medium"
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "models" / "piper"


def run(command: list[str]) -> None:
    print(">", " ".join(command))
    subprocess.check_call(command)


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    run([sys.executable, "-m", "pip", "install", "piper-tts"])
    run([
        sys.executable,
        "-m",
        "piper.download_voices",
        "--data-dir",
        str(DATA_DIR),
        VOICE,
    ])
    model = DATA_DIR / f"{VOICE}.onnx"
    config = DATA_DIR / f"{VOICE}.onnx.json"
    if not model.is_file() or not config.is_file():
        raise SystemExit("Piper đã chạy nhưng chưa tìm thấy đủ file .onnx/.onnx.json.")
    print("\nPiper tiếng Việt đã sẵn sàng:")
    print(" model:", model)
    print(" config:", config)
    print("Bây giờ vào Admin -> Tiêu chí động tác -> Lưu tiêu chí & tạo âm thanh.")


if __name__ == "__main__":
    main()
