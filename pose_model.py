"""Thread-safe lazy loading of the YOLO pose model.

Keeping one model instance per Python process avoids repeatedly loading the
weights whenever a workout stream is created.
"""
from __future__ import annotations

from threading import Lock

from ultralytics import YOLO

from config import POSE_IMGSZ, POSE_CONF, POSE_USE_HALF

_MODEL = None
_MODEL_PATH = None
_MODEL_LOCK = Lock()
_INFERENCE_LOCK = Lock()


def get_pose_model(model_path: str):
    global _MODEL, _MODEL_PATH
    with _MODEL_LOCK:
        if _MODEL is None or _MODEL_PATH != model_path:
            _MODEL = YOLO(model_path)
            _MODEL_PATH = model_path
    return _MODEL


def run_pose_inference(model, frame):
    """Run low-latency pose inference on the newest camera frame.

    The browser preview can stay 960x540, but YOLO does not need to infer at
    that full size. Ultralytics will use CUDA automatically when available;
    half precision is requested only on CUDA-capable deployments.
    """
    kwargs = {
        "verbose": False,
        "imgsz": int(POSE_IMGSZ),
        "conf": float(POSE_CONF),
    }
    try:
        import torch
        if bool(POSE_USE_HALF) and torch.cuda.is_available():
            kwargs.update({"device": 0, "half": True})
    except Exception:
        pass
    with _INFERENCE_LOCK:
        if hasattr(model, "predict"):
            try:
                return model.predict(source=frame, **kwargs)
            except Exception:
                pass
        if callable(model):
            try:
                return model(frame, verbose=False)
            except TypeError:
                return model(frame)
        return model.predict(source=frame, **kwargs)
