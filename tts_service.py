"""Offline text-to-speech service for exercise feedback.

FitMotion follows the benchmark thesis workflow: when Admin saves an expert
rule, its Vietnamese feedback is materialized into an audio file once. During
training the browser plays that stored file, so the rule and the spoken message
remain traceable together.

Provider order:
1. Piper (preferred, matching the benchmark thesis idea).
2. pyttsx3 as an offline system-voice fallback.

The current Piper package supports ``python -m piper``.  For compatibility with
older installations, this module also tries the legacy ``piper --model ...``
CLI. No fake audio file is reported as success: a provider is accepted only
when a non-empty WAV file exists after synthesis.
"""
from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import shutil
import subprocess
import sys


DEFAULT_VOICE_NAME = "vi_VN-vais1000-medium"
SOURCE_ROOT = Path(__file__).resolve().parent
DEFAULT_PIPER_DATA_DIR = SOURCE_ROOT / "models" / "piper"
DEFAULT_PIPER_MODEL = DEFAULT_PIPER_DATA_DIR / f"{DEFAULT_VOICE_NAME}.onnx"


@dataclass(frozen=True)
class TTSResult:
    success: bool
    provider: str = ""
    output_path: str = ""
    error: str = ""


def _valid_wav(path: Path) -> bool:
    try:
        return path.is_file() and path.stat().st_size > 44
    except OSError:
        return False


def _piper_model_value() -> tuple[str, Path | None]:
    configured = os.getenv("PIPER_MODEL", "").strip()
    if configured:
        path = Path(configured).expanduser()
        if path.is_file():
            return str(path.resolve()), path.resolve()
        # The modern Piper CLI also accepts a voice/model name when its data
        # directory contains the downloaded voice.
        return configured, None
    if DEFAULT_PIPER_MODEL.is_file():
        return str(DEFAULT_PIPER_MODEL), DEFAULT_PIPER_MODEL
    return "", None


def _run_process(command: list[str], *, stdin_text: str | None = None, timeout: int = 120) -> tuple[bool, str]:
    try:
        completed = subprocess.run(
            command,
            input=stdin_text,
            text=True,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
        message = (completed.stderr or completed.stdout or f"exit={completed.returncode}").strip()
        return completed.returncode == 0, message
    except Exception as exc:  # pragma: no cover - environment dependent
        return False, str(exc)


def _run_piper(text: str, output_path: Path) -> TTSResult:
    model_value, model_path = _piper_model_value()
    if not model_value:
        return TTSResult(
            False,
            "piper",
            error=(
                "Chưa có giọng Piper tiếng Việt. Chạy scripts/setup_piper_vi.py "
                "hoặc cấu hình PIPER_MODEL."
            ),
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    data_dir = Path(os.getenv("PIPER_DATA_DIR", "").strip() or DEFAULT_PIPER_DATA_DIR)
    errors: list[str] = []

    # Current OHF Piper CLI: python -m piper -m MODEL -f out.wav -- 'text'
    modern = [
        sys.executable,
        "-m",
        "piper",
        "-m",
        model_value,
        "-f",
        str(output_path),
    ]
    if model_path is None and data_dir:
        modern.extend(["--data-dir", str(data_dir)])
    modern.extend(["--", text])
    ok, message = _run_process(modern)
    if ok and _valid_wav(output_path):
        return TTSResult(True, "piper", str(output_path))
    errors.append(message or "Piper modern CLI không tạo WAV.")

    # Legacy Piper CLI compatibility for older Windows/local installations.
    exe = os.getenv("PIPER_EXE", "").strip() or shutil.which("piper") or ""
    if exe and (model_path is not None or Path(model_value).is_file()):
        legacy_model = str(model_path or model_value)
        legacy_attempts = [
            ([exe, "--model", legacy_model, "--output_file", str(output_path), "--text", text], None),
            ([exe, "--model", legacy_model, "--output_file", str(output_path)], text),
        ]
        for command, stdin_text in legacy_attempts:
            ok, message = _run_process(command, stdin_text=stdin_text)
            if ok and _valid_wav(output_path):
                return TTSResult(True, "piper", str(output_path))
            errors.append(message or "Piper legacy CLI không tạo WAV.")

    return TTSResult(False, "piper", error=" | ".join(item for item in errors if item)[:1500])


def _run_pyttsx3(text: str, output_path: Path) -> TTSResult:
    try:
        import pyttsx3

        output_path.parent.mkdir(parents=True, exist_ok=True)
        engine = pyttsx3.init()
        engine.setProperty("rate", 165)
        engine.save_to_file(text, str(output_path))
        engine.runAndWait()
        try:
            engine.stop()
        except Exception:
            pass
        if _valid_wav(output_path):
            return TTSResult(True, "pyttsx3", str(output_path))
        return TTSResult(False, "pyttsx3", error="pyttsx3 không tạo được file WAV hợp lệ.")
    except Exception as exc:  # pragma: no cover - environment dependent
        return TTSResult(False, "pyttsx3", error=str(exc))


def _run_gtts(text: str, output_path: Path) -> TTSResult:
    """Tổng hợp tiếng Việt bằng Google TTS (gTTS) - giọng Việt chuẩn.

    Yêu cầu kết nối internet và thư viện gtts, imageio-ffmpeg.
    Được ưu tiên hơn pyttsx3 vì hệ thống Windows thường chỉ có giọng tiếng Anh.
    """
    try:
        from gtts import gTTS  # type: ignore
        import subprocess
        import tempfile
        import os as _os

        try:
            import imageio_ffmpeg  # type: ignore
            ffmpeg_bin = imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:
            ffmpeg_bin = "ffmpeg"

        output_path.parent.mkdir(parents=True, exist_ok=True)

        tts = gTTS(text=text, lang="vi", slow=False)
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
            tmp_path = tmp.name
        tts.save(tmp_path)

        cmd = [
            ffmpeg_bin, "-y",
            "-i", tmp_path,
            "-ar", "22050",
            "-ac", "1",
            "-sample_fmt", "s16",
            str(output_path),
        ]
        proc = subprocess.run(cmd, capture_output=True, timeout=60)
        try:
            _os.unlink(tmp_path)
        except Exception:
            pass

        if proc.returncode == 0 and _valid_wav(output_path):
            return TTSResult(True, "gtts", str(output_path))
        err = (proc.stderr or b"").decode("utf-8", errors="ignore")[-300:]
        return TTSResult(False, "gtts", error=f"ffmpeg exit={proc.returncode}: {err}")

    except ImportError:
        return TTSResult(False, "gtts", error="Thiếu thư viện gtts hoặc imageio-ffmpeg.")
    except Exception as exc:  # pragma: no cover - environment dependent
        return TTSResult(False, "gtts", error=str(exc))


def _run_edge_tts(text: str, output_path: Path) -> TTSResult:
    """Tổng hợp giọng nam tiếng Việt bằng Microsoft Edge TTS (vi-VN-NamMinhNeural)."""
    try:
        import asyncio
        import edge_tts  # type: ignore
        import subprocess
        import tempfile
        import os as _os

        try:
            import imageio_ffmpeg  # type: ignore
            ffmpeg_bin = imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:
            ffmpeg_bin = "ffmpeg"

        output_path.parent.mkdir(parents=True, exist_ok=True)

        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
            tmp_mp3 = tmp.name

        async def _speak():
            comm = edge_tts.Communicate(text, "vi-VN-NamMinhNeural")
            await comm.save(tmp_mp3)

        try:
            asyncio.run(_speak())
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(_speak())
            loop.close()

        cmd = [
            ffmpeg_bin, "-y",
            "-i", tmp_mp3,
            "-ar", "22050",
            "-ac", "1",
            "-sample_fmt", "s16",
            str(output_path),
        ]
        proc = subprocess.run(cmd, capture_output=True, timeout=30)
        try:
            _os.remove(tmp_mp3)
        except Exception:
            pass

        if proc.returncode == 0 and _valid_wav(output_path):
            return TTSResult(True, "edge-tts (Nam)", str(output_path))
        err = (proc.stderr or b"").decode("utf-8", errors="ignore")[-300:]
        return TTSResult(False, "edge-tts", error=f"ffmpeg exit={proc.returncode}: {err}")

    except ImportError:
        return TTSResult(False, "edge-tts", error="Thiếu thư viện edge-tts.")
    except Exception as exc:  # pragma: no cover - environment dependent
        return TTSResult(False, "edge-tts", error=str(exc))


def synthesize_feedback(text: str, output_path: str | Path) -> TTSResult:
    """Generate one durable WAV file for an expert-rule message.

    Provider order:
    1. Piper     - offline Vietnamese neural voice (requires onnx model).
    2. Edge TTS  - Vietnamese male voice (vi-VN-NamMinhNeural, natural and clear).
    3. gTTS      - Google TTS online Vietnamese voice.
    4. pyttsx3   - system voice fallback.
    """
    clean = " ".join(str(text or "").split()).strip()
    if not clean:
        return TTSResult(False, error="Nội dung âm thanh đang trống.")

    destination = Path(output_path)
    if destination.suffix.lower() != ".wav":
        destination = destination.with_suffix(".wav")
    try:
        if destination.exists():
            destination.unlink()
    except OSError:
        pass

    piper = _run_piper(clean, destination)
    if piper.success:
        return piper

    edge_result = _run_edge_tts(clean, destination)
    if edge_result.success:
        return edge_result

    gtts_result = _run_gtts(clean, destination)
    if gtts_result.success:
        return gtts_result

    fallback = _run_pyttsx3(clean, destination)
    if fallback.success:
        return fallback

    error = (
        f"Piper: {piper.error or 'không khả dụng'}; "
        f"Edge TTS: {edge_result.error or 'không khả dụng'}; "
        f"gTTS: {gtts_result.error or 'không khả dụng'}; "
        f"pyttsx3: {fallback.error or 'không khả dụng'}"
    )
    return TTSResult(False, error=error[:2000])

