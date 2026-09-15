"""Compatibility launcher for Squat using the unified week-6 pipeline."""
from standalone_runner import run_camera


def runsquat(camera_index: int = 0):
    return run_camera("squat", camera_index)


if __name__ == "__main__":
    runsquat()
