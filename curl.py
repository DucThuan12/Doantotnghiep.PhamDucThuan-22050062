"""Compatibility launcher for Bicep Curl using the unified week-6 pipeline."""
from standalone_runner import run_camera


def runcurl(side: str = "left", camera_index: int = 0):
    slug = "curl-right" if str(side).lower() == "right" else "curl-left"
    return run_camera(slug, camera_index)


if __name__ == "__main__":
    runcurl("left")
