"""Compatibility launcher for Push-up using the unified week-6 pipeline."""
from standalone_runner import run_camera


def runpushup(camera_index: int = 0):
    return run_camera("pushup", camera_index)


if __name__ == "__main__":
    runpushup()
