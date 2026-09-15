"""Directly derive FitMotion scoring references from a rigged Mixamo GLB.

This module makes the 3D trainer measurable instead of merely decorative.
For the four thesis exercises, a GLB uploaded by Admin can be sampled directly
from its embedded Mixamo skeleton animation and converted to the same
``fitmotion.reference.v1`` JSON consumed by :mod:`reference_motion`.

Only GLB is auto-parsed here. FBX/GLTF uploads remain valid visual assets, but
FBX references should still be exported with the Blender helper because FBX is
not a self-contained glTF container.
"""
from __future__ import annotations

import json
import math
import struct
from pathlib import Path
from typing import Any

import numpy as np

from reference_motion import SCHEMA, validate_reference_payload


class GLBReferenceError(ValueError):
    """Raised when a GLB cannot be trusted as a measurable reference asset."""


_COMPONENT_DTYPES = {
    5120: np.int8,
    5121: np.uint8,
    5122: np.int16,
    5123: np.uint16,
    5125: np.uint32,
    5126: np.float32,
}
_TYPE_COMPONENTS = {
    "SCALAR": 1,
    "VEC2": 2,
    "VEC3": 3,
    "VEC4": 4,
    "MAT2": 4,
    "MAT3": 9,
    "MAT4": 16,
}

BONES = {
    "left_shoulder": "mixamorig:LeftArm",
    "left_elbow": "mixamorig:LeftForeArm",
    "left_wrist": "mixamorig:LeftHand",
    "right_shoulder": "mixamorig:RightArm",
    "right_elbow": "mixamorig:RightForeArm",
    "right_wrist": "mixamorig:RightHand",
    "left_hip": "mixamorig:LeftUpLeg",
    "left_knee": "mixamorig:LeftLeg",
    "left_ankle": "mixamorig:LeftFoot",
    "right_hip": "mixamorig:RightUpLeg",
    "right_knee": "mixamorig:RightLeg",
    "right_ankle": "mixamorig:RightFoot",
}
CORE_EXERCISES = {"squat", "pushup", "curl-left", "curl-right"}


def _normalize_exercise(value: str) -> str:
    value = str(value or "").strip().lower().replace("_", "-")
    return {"push-up": "pushup", "push up": "pushup"}.get(value, value)


def _read_glb(path: str | Path) -> tuple[dict, bytes]:
    path = Path(path)
    if path.suffix.lower() != ".glb":
        raise GLBReferenceError("Tự động tạo reference hiện chỉ hỗ trợ file .glb.")
    raw = path.read_bytes()
    if len(raw) < 20:
        raise GLBReferenceError("File GLB quá nhỏ hoặc bị hỏng.")
    magic, version, declared_length = struct.unpack_from("<4sII", raw, 0)
    if magic != b"glTF" or version != 2:
        raise GLBReferenceError("Chỉ hỗ trợ GLB glTF 2.0.")
    if declared_length > len(raw):
        raise GLBReferenceError("GLB bị thiếu dữ liệu so với header.")

    offset = 12
    document = None
    binary = None
    while offset + 8 <= declared_length:
        chunk_length, chunk_type = struct.unpack_from("<II", raw, offset)
        offset += 8
        chunk = raw[offset: offset + chunk_length]
        offset += chunk_length
        if len(chunk) != chunk_length:
            raise GLBReferenceError("GLB có chunk bị cắt ngắn.")
        if chunk_type == 0x4E4F534A:  # JSON
            try:
                document = json.loads(chunk.decode("utf-8").rstrip("\x00 \t\r\n"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise GLBReferenceError("Không đọc được JSON chunk của GLB.") from exc
        elif chunk_type == 0x004E4942:  # BIN\0
            binary = bytes(chunk)

    if not isinstance(document, dict) or binary is None:
        raise GLBReferenceError("GLB phải có cả JSON chunk và BIN chunk.")
    return document, binary


def inspect_glb_asset(path: str | Path) -> dict:
    """Return structural evidence that an upload is a real animated 3D asset."""
    document, _ = _read_glb(path)
    animations = [str(item.get("name", "") or "") for item in document.get("animations", [])]
    summary = {
        "asset_version": str((document.get("asset") or {}).get("version", "")),
        "generator": str((document.get("asset") or {}).get("generator", "")),
        "nodes": len(document.get("nodes", [])),
        "meshes": len(document.get("meshes", [])),
        "skins": len(document.get("skins", [])),
        "materials": len(document.get("materials", [])),
        "textures": len(document.get("textures", [])),
        "images": len(document.get("images", [])),
        "animations": animations,
    }
    if summary["meshes"] < 1:
        raise GLBReferenceError("GLB không có mesh nhân vật.")
    if summary["skins"] < 1:
        raise GLBReferenceError("GLB không có skin/rig xương.")
    if not animations:
        raise GLBReferenceError("GLB không có animation clip.")
    return summary


class _GLBAnimation:
    def __init__(self, document: dict, binary: bytes, animation_key: str):
        self.document = document
        self.binary = binary
        self.nodes = document.get("nodes", [])
        animations = document.get("animations", [])
        if not animations:
            raise GLBReferenceError("GLB không chứa animation.")

        key = str(animation_key or "").strip().lower()
        exact = next((a for a in animations if str(a.get("name", "")).strip().lower() == key), None) if key else None
        contains = next((a for a in animations if key and key in str(a.get("name", "")).strip().lower()), None)
        self.animation = exact or contains or animations[0]
        self.animation_name = str(self.animation.get("name", "") or animation_key or "animation")

        self.parents: dict[int, int] = {}
        for parent_idx, node in enumerate(self.nodes):
            for child_idx in node.get("children", []) or []:
                self.parents[int(child_idx)] = parent_idx

        self.channels: dict[tuple[int, str], dict] = {}
        self.start_time = math.inf
        self.end_time = -math.inf
        samplers = self.animation.get("samplers", [])
        for channel in self.animation.get("channels", []) or []:
            target = channel.get("target") or {}
            if "node" not in target or target.get("path") not in {"translation", "rotation", "scale"}:
                continue
            sampler_idx = int(channel.get("sampler", -1))
            if not 0 <= sampler_idx < len(samplers):
                continue
            sampler = samplers[sampler_idx]
            times = self._accessor(int(sampler["input"])).astype(np.float64).reshape(-1)
            values = self._accessor(int(sampler["output"])).astype(np.float64)
            interpolation = str(sampler.get("interpolation", "LINEAR") or "LINEAR").upper()
            if interpolation not in {"LINEAR", "STEP"}:
                raise GLBReferenceError(
                    f"Animation {self.animation_name} dùng {interpolation}; hãy export lại Mixamo GLB với LINEAR/STEP."
                )
            if times.size == 0:
                continue
            self.start_time = min(self.start_time, float(times[0]))
            self.end_time = max(self.end_time, float(times[-1]))
            self.channels[(int(target["node"]), str(target["path"]))] = {
                "times": times,
                "values": values,
                "interpolation": interpolation,
            }
        if not math.isfinite(self.start_time) or not math.isfinite(self.end_time) or self.end_time <= self.start_time:
            raise GLBReferenceError("Animation GLB không có khoảng thời gian hợp lệ.")

        self.name_to_index = {}
        for index, node in enumerate(self.nodes):
            name = str(node.get("name", "") or "")
            if name:
                self.name_to_index[name.lower()] = index
                self.name_to_index.setdefault(name.split(":")[-1].lower(), index)

    def _accessor(self, accessor_index: int) -> np.ndarray:
        accessors = self.document.get("accessors", [])
        views = self.document.get("bufferViews", [])
        if not 0 <= accessor_index < len(accessors):
            raise GLBReferenceError("Accessor GLB không hợp lệ.")
        accessor = accessors[accessor_index]
        if "sparse" in accessor:
            raise GLBReferenceError("GLB reference chưa hỗ trợ sparse accessor.")
        view_index = accessor.get("bufferView")
        if view_index is None or not 0 <= int(view_index) < len(views):
            raise GLBReferenceError("Accessor GLB thiếu bufferView.")
        view = views[int(view_index)]
        dtype = _COMPONENT_DTYPES.get(int(accessor.get("componentType", 0)))
        components = _TYPE_COMPONENTS.get(str(accessor.get("type", "")))
        if dtype is None or components is None:
            raise GLBReferenceError("Accessor GLB dùng component/type chưa hỗ trợ.")
        count = int(accessor.get("count", 0))
        item_size = np.dtype(dtype).itemsize * components
        stride = int(view.get("byteStride", item_size) or item_size)
        start = int(view.get("byteOffset", 0) or 0) + int(accessor.get("byteOffset", 0) or 0)
        if stride == item_size:
            end = start + count * item_size
            if end > len(self.binary):
                raise GLBReferenceError("Accessor vượt quá BIN chunk.")
            arr = np.frombuffer(self.binary, dtype=dtype, count=count * components, offset=start)
            return arr.reshape(count, components) if components > 1 else arr.reshape(count)

        rows = []
        for idx in range(count):
            row_start = start + idx * stride
            row_end = row_start + item_size
            if row_end > len(self.binary):
                raise GLBReferenceError("Accessor có byteStride vượt quá BIN chunk.")
            row = np.frombuffer(self.binary, dtype=dtype, count=components, offset=row_start)
            rows.append(np.array(row, copy=True))
        arr = np.asarray(rows)
        return arr.reshape(count) if components == 1 else arr

    @staticmethod
    def _quat_slerp(a: np.ndarray, b: np.ndarray, alpha: float) -> np.ndarray:
        a = np.asarray(a, dtype=np.float64)
        b = np.asarray(b, dtype=np.float64)
        a /= max(1e-12, float(np.linalg.norm(a)))
        b /= max(1e-12, float(np.linalg.norm(b)))
        dot = float(np.dot(a, b))
        if dot < 0.0:
            b = -b
            dot = -dot
        dot = max(-1.0, min(1.0, dot))
        if dot > 0.9995:
            out = a + alpha * (b - a)
            return out / max(1e-12, float(np.linalg.norm(out)))
        theta_0 = math.acos(dot)
        sin_theta_0 = math.sin(theta_0)
        theta = theta_0 * alpha
        s0 = math.sin(theta_0 - theta) / sin_theta_0
        s1 = math.sin(theta) / sin_theta_0
        return s0 * a + s1 * b

    def _sample_channel(self, node_idx: int, path: str, time_value: float, default: np.ndarray) -> np.ndarray:
        channel = self.channels.get((node_idx, path))
        if channel is None:
            return np.asarray(default, dtype=np.float64)
        times = channel["times"]
        values = channel["values"]
        if time_value <= times[0]:
            return np.asarray(values[0], dtype=np.float64)
        if time_value >= times[-1]:
            return np.asarray(values[-1], dtype=np.float64)
        right = int(np.searchsorted(times, time_value, side="right"))
        left = max(0, right - 1)
        right = min(right, len(times) - 1)
        if channel["interpolation"] == "STEP" or right == left:
            return np.asarray(values[left], dtype=np.float64)
        span = max(1e-12, float(times[right] - times[left]))
        alpha = max(0.0, min(1.0, float((time_value - times[left]) / span)))
        if path == "rotation":
            return self._quat_slerp(values[left], values[right], alpha)
        return np.asarray(values[left] + (values[right] - values[left]) * alpha, dtype=np.float64)

    @staticmethod
    def _quat_matrix(quat: np.ndarray) -> np.ndarray:
        x, y, z, w = [float(v) for v in quat]
        norm = math.sqrt(x*x + y*y + z*z + w*w)
        if norm <= 1e-12:
            return np.eye(4, dtype=np.float64)
        x, y, z, w = x/norm, y/norm, z/norm, w/norm
        xx, yy, zz = x*x, y*y, z*z
        xy, xz, yz = x*y, x*z, y*z
        wx, wy, wz = w*x, w*y, w*z
        return np.array([
            [1 - 2*(yy+zz), 2*(xy-wz), 2*(xz+wy), 0],
            [2*(xy+wz), 1 - 2*(xx+zz), 2*(yz-wx), 0],
            [2*(xz-wy), 2*(yz+wx), 1 - 2*(xx+yy), 0],
            [0, 0, 0, 1],
        ], dtype=np.float64)

    def _local_matrix(self, node_idx: int, time_value: float) -> np.ndarray:
        node = self.nodes[node_idx]
        # glTF forbids matrix together with animated TRS. Preserve a static
        # matrix when present; real Mixamo exports here use TRS.
        if "matrix" in node and not any((node_idx, p) in self.channels for p in ("translation", "rotation", "scale")):
            # glTF matrix arrays are column-major.
            return np.asarray(node["matrix"], dtype=np.float64).reshape((4, 4), order="F")

        translation = self._sample_channel(
            node_idx, "translation", time_value,
            np.asarray(node.get("translation", [0.0, 0.0, 0.0]), dtype=np.float64),
        )
        rotation = self._sample_channel(
            node_idx, "rotation", time_value,
            np.asarray(node.get("rotation", [0.0, 0.0, 0.0, 1.0]), dtype=np.float64),
        )
        scale = self._sample_channel(
            node_idx, "scale", time_value,
            np.asarray(node.get("scale", [1.0, 1.0, 1.0]), dtype=np.float64),
        )
        t = np.eye(4, dtype=np.float64)
        t[:3, 3] = translation[:3]
        s = np.eye(4, dtype=np.float64)
        s[0, 0], s[1, 1], s[2, 2] = [float(v) for v in scale[:3]]
        return t @ self._quat_matrix(rotation) @ s

    def _global_matrix(self, node_idx: int, time_value: float, cache: dict[int, np.ndarray]) -> np.ndarray:
        if node_idx in cache:
            return cache[node_idx]
        local = self._local_matrix(node_idx, time_value)
        parent = self.parents.get(node_idx)
        result = self._global_matrix(parent, time_value, cache) @ local if parent is not None else local
        cache[node_idx] = result
        return result

    def point(self, bone_name: str, time_value: float, cache: dict[int, np.ndarray]) -> np.ndarray:
        candidates = [bone_name.lower(), bone_name.split(":")[-1].lower()]
        node_idx = next((self.name_to_index.get(name) for name in candidates if name in self.name_to_index), None)
        if node_idx is None:
            raise GLBReferenceError(f"Thiếu bone Mixamo bắt buộc: {bone_name}")
        matrix = self._global_matrix(int(node_idx), time_value, cache)
        return np.asarray(matrix[:3, 3], dtype=np.float64)


def _angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    ba = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    bc = np.asarray(c, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    nba = float(np.linalg.norm(ba))
    nbc = float(np.linalg.norm(bc))
    if nba <= 1e-9 or nbc <= 1e-9:
        return 0.0
    dot = float(np.dot(ba / nba, bc / nbc))
    return math.degrees(math.acos(max(-1.0, min(1.0, dot))))


def _frame_metrics(animation: _GLBAnimation, exercise: str, time_value: float, first_positions: dict[str, np.ndarray]) -> dict:
    cache: dict[int, np.ndarray] = {}
    p = {name: animation.point(bone, time_value, cache) for name, bone in BONES.items()}
    if exercise == "squat":
        left_knee = _angle(p["left_hip"], p["left_knee"], p["left_ankle"])
        right_knee = _angle(p["right_hip"], p["right_knee"], p["right_ankle"])
        shoulder_mid = (p["left_shoulder"] + p["right_shoulder"]) / 2.0
        hip_mid = (p["left_hip"] + p["right_hip"]) / 2.0
        ankle_mid = (p["left_ankle"] + p["right_ankle"]) / 2.0
        return {"knee_angle": (left_knee + right_knee) / 2.0, "torso_angle": _angle(shoulder_mid, hip_mid, ankle_mid)}
    if exercise == "pushup":
        elbows = [
            _angle(p["left_shoulder"], p["left_elbow"], p["left_wrist"]),
            _angle(p["right_shoulder"], p["right_elbow"], p["right_wrist"]),
        ]
        shoulder_mid = (p["left_shoulder"] + p["right_shoulder"]) / 2.0
        hip_mid = (p["left_hip"] + p["right_hip"]) / 2.0
        ankle_mid = (p["left_ankle"] + p["right_ankle"]) / 2.0
        return {"elbow_angle": sum(elbows) / 2.0, "body_angle": _angle(shoulder_mid, hip_mid, ankle_mid)}
    side = "left" if exercise == "curl-left" else "right"
    elbow_angle = _angle(p[f"{side}_shoulder"], p[f"{side}_elbow"], p[f"{side}_wrist"])
    upper = max(0.001, float(np.linalg.norm(p[f"{side}_shoulder"] - p[f"{side}_elbow"])))
    shift = float(np.linalg.norm(p[f"{side}_elbow"] - first_positions[f"{side}_elbow"])) / upper * 100.0
    return {"elbow_angle": elbow_angle, "elbow_shift": shift}


def build_reference_from_glb(
    path: str | Path,
    exercise: str,
    animation_key: str = "",
    *,
    samples: int = 40,
    min_score: float = 55.0,
) -> dict:
    """Sample a Mixamo GLB and return a validated FitMotion reference payload."""
    exercise = _normalize_exercise(exercise)
    if exercise not in CORE_EXERCISES:
        raise GLBReferenceError("Chỉ tự tạo reference cho squat, pushup, curl-left và curl-right.")
    path = Path(path)
    inspect_glb_asset(path)
    document, binary = _read_glb(path)
    animation = _GLBAnimation(document, binary, animation_key)
    sample_count = max(8, min(240, int(samples)))
    times = np.linspace(animation.start_time, animation.end_time, sample_count)

    first_cache: dict[int, np.ndarray] = {}
    first_positions = {
        name: animation.point(bone, float(times[0]), first_cache)
        for name, bone in BONES.items()
    }
    frames = []
    for index, time_value in enumerate(times):
        metrics = _frame_metrics(animation, exercise, float(time_value), first_positions)
        frames.append({
            "t": round(index / float(sample_count - 1), 6),
            **{key: round(float(value), 4) for key, value in metrics.items()},
        })

    # Canonicalize a complete repetition to the same phase order used by the
    # realtime state machines: extended/top -> lowering/curling -> return.
    # Some Mixamo clips (notably the supplied push-up) are authored starting
    # at the bottom position. A cyclic shift keeps the exact animation shape
    # while avoiding a false DTW penalty purely because the clip starts at a
    # different point in the cycle.
    primary_metric = "knee_angle" if exercise == "squat" else "elbow_angle"
    if frames and primary_metric in frames[0]:
        start_idx = max(range(len(frames)), key=lambda idx: float(frames[idx][primary_metric]))
        if start_idx:
            frames = frames[start_idx:] + frames[:start_idx]
        for idx, frame in enumerate(frames):
            frame["t"] = round(idx / float(len(frames) - 1), 6)

    metric_names = [key for key in frames[0] if key != "t"]
    metric_scales = {}
    for metric in metric_names:
        values = [float(frame[metric]) for frame in frames]
        metric_scales[metric] = round(max(15.0, max(values) - min(values)), 4)

    payload = {
        "schema": SCHEMA,
        "exercise": exercise,
        "animation_key": animation.animation_name,
        "version": "mixamo-glb-v1",
        "source": {
            "type": "mixamo-blender",
            "asset": path.name,
            "action": animation.animation_name,
            "generator": str((document.get("asset") or {}).get("generator", "")),
            "reference_method": "direct-glb-skeleton-sampling",
        },
        "metrics": metric_names,
        "metric_scales": metric_scales,
        "min_score": max(0.0, min(100.0, float(min_score))),
        "frames": frames,
    }
    return validate_reference_payload(payload, expected_exercise=exercise)


def write_reference_from_glb(
    glb_path: str | Path,
    output_path: str | Path,
    exercise: str,
    animation_key: str = "",
    *,
    samples: int = 40,
    min_score: float = 55.0,
) -> dict:
    payload = build_reference_from_glb(
        glb_path, exercise, animation_key, samples=samples, min_score=min_score
    )
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload
