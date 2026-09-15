"""Inspect a GLB and optionally export the FitMotion reference JSON.

Example Windows (PowerShell):
  python scripts/check_glb_reference.py --input "C:\\...\\remy_air_squat.glb" --exercise squat --animation squat --output "squat-reference.json"
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from glb_reference import inspect_glb_asset, write_reference_from_glb


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--exercise", required=True, choices=["squat", "pushup", "curl-left", "curl-right"])
    parser.add_argument("--animation", default="")
    parser.add_argument("--output", default="")
    parser.add_argument("--samples", type=int, default=40)
    args = parser.parse_args()

    info = inspect_glb_asset(args.input)
    print("[GLB]", json.dumps(info, ensure_ascii=False, indent=2))
    if args.output:
        payload = write_reference_from_glb(
            args.input, args.output, args.exercise, args.animation,
            samples=args.samples,
        )
        print(f"[PASS] Đã xuất reference: {Path(args.output).resolve()}")
        print("[REFERENCE]", json.dumps({
            "exercise": payload["exercise"],
            "animation_key": payload["animation_key"],
            "metrics": payload["metrics"],
            "metric_scales": payload["metric_scales"],
            "frames": len(payload["frames"]),
        }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
