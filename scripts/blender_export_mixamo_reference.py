"""Run inside Blender to export a Mixamo action as FitMotion reference JSON.

Example:
  blender -b FitMotionTrainer.blend \
    --python scripts/blender_export_mixamo_reference.py -- \
    --exercise squat --action Squat --output static/references/squat.json

Expected Mixamo bone names use the common ``mixamorig:`` prefix. Use
``--armature`` when the armature object has a non-default name.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import bpy
from mathutils import Vector

SCHEMA = "fitmotion.reference.v1"

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
    "spine": "mixamorig:Spine2",
}


def parse_args():
    raw = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--exercise", required=True, choices=["squat", "pushup", "curl-left", "curl-right"])
    parser.add_argument("--action", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--armature", default="")
    parser.add_argument("--samples", type=int, default=40)
    parser.add_argument("--min-score", type=float, default=55.0)
    return parser.parse_args(raw)


def find_armature(name=""):
    if name:
        obj = bpy.data.objects.get(name)
        if obj and obj.type == "ARMATURE":
            return obj
    for obj in bpy.context.scene.objects:
        if obj.type == "ARMATURE":
            return obj
    raise RuntimeError("Không tìm thấy Armature trong Blender scene.")


def point(armature, bone_name):
    bone = armature.pose.bones.get(bone_name)
    if bone is None:
        # Accept rigs whose exporter removed the mixamorig prefix.
        bone = armature.pose.bones.get(bone_name.split(":")[-1])
    if bone is None:
        raise RuntimeError(f"Thiếu bone: {bone_name}")
    return armature.matrix_world @ bone.head


def angle(a: Vector, b: Vector, c: Vector):
    ba = (a - b).normalized()
    bc = (c - b).normalized()
    dot = max(-1.0, min(1.0, ba.dot(bc)))
    return math.degrees(math.acos(dot))


def frame_metrics(armature, exercise, first_positions):
    p = {name: point(armature, bone) for name, bone in BONES.items()}
    if exercise == "squat":
        left_knee = angle(p["left_hip"], p["left_knee"], p["left_ankle"])
        right_knee = angle(p["right_hip"], p["right_knee"], p["right_ankle"])
        shoulder_mid = (p["left_shoulder"] + p["right_shoulder"]) / 2.0
        hip_mid = (p["left_hip"] + p["right_hip"]) / 2.0
        ankle_mid = (p["left_ankle"] + p["right_ankle"]) / 2.0
        return {"knee_angle": (left_knee + right_knee) / 2.0, "torso_angle": angle(shoulder_mid, hip_mid, ankle_mid)}
    if exercise == "pushup":
        elbows = [
            angle(p["left_shoulder"], p["left_elbow"], p["left_wrist"]),
            angle(p["right_shoulder"], p["right_elbow"], p["right_wrist"]),
        ]
        shoulder_mid = (p["left_shoulder"] + p["right_shoulder"]) / 2.0
        hip_mid = (p["left_hip"] + p["right_hip"]) / 2.0
        ankle_mid = (p["left_ankle"] + p["right_ankle"]) / 2.0
        return {"elbow_angle": sum(elbows) / 2.0, "body_angle": angle(shoulder_mid, hip_mid, ankle_mid)}
    side = "left" if exercise == "curl-left" else "right"
    elbow_angle = angle(p[f"{side}_shoulder"], p[f"{side}_elbow"], p[f"{side}_wrist"])
    upper = max(0.001, (p[f"{side}_shoulder"] - p[f"{side}_elbow"]).length)
    shift = (p[f"{side}_elbow"] - first_positions[f"{side}_elbow"]).length / upper * 100.0
    return {"elbow_angle": elbow_angle, "elbow_shift": shift}


def main():
    args = parse_args()
    armature = find_armature(args.armature)
    action = bpy.data.actions.get(args.action)
    if action is None:
        raise RuntimeError(f"Không tìm thấy action: {args.action}")
    if armature.animation_data is None:
        armature.animation_data_create()
    armature.animation_data.action = action

    start, end = action.frame_range
    sample_count = max(8, int(args.samples))
    scene = bpy.context.scene
    first_frame = int(round(start))
    scene.frame_set(first_frame)
    first_positions = {name: point(armature, bone) for name, bone in BONES.items()}

    frames = []
    for i in range(sample_count):
        t = i / float(sample_count - 1)
        frame_no = start + (end - start) * t
        scene.frame_set(int(round(frame_no)))
        metrics = frame_metrics(armature, args.exercise, first_positions)
        frames.append({"t": round(t, 6), **{key: round(float(value), 4) for key, value in metrics.items()}})

    metrics = [key for key in frames[0] if key != "t"]
    scales = {}
    for metric in metrics:
        values = [item[metric] for item in frames]
        scales[metric] = round(max(15.0, max(values) - min(values)), 4)

    payload = {
        "schema": SCHEMA,
        "exercise": args.exercise,
        "animation_key": args.action,
        "version": "mixamo-v1",
        "source": {
            "type": "mixamo-blender",
            "blend_file": Path(bpy.data.filepath).name,
            "action": args.action,
        },
        "metrics": metrics,
        "metric_scales": scales,
        "min_score": max(0.0, min(100.0, args.min_score)),
        "frames": frames,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print("FITMOTION_REFERENCE_EXPORTED", output)


if __name__ == "__main__":
    main()
