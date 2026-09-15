"""Static validation that does not require Flask, Ultralytics or a webcam."""
from __future__ import annotations

import json
import os
import re
import sqlite3
import subprocess
import sys
import tempfile
from pathlib import Path

from jinja2 import Environment, FileSystemLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from storage import prepare_persistent_database


def result(ok: bool, label: str, detail: str = "") -> bool:
    print(f"[{'PASS' if ok else 'FAIL'}] {label}" + (f" - {detail}" if detail else ""))
    return ok


def validate_python() -> bool:
    command = [sys.executable, "-m", "compileall", "-q", str(ROOT)]
    completed = subprocess.run(command, check=False)
    return result(completed.returncode == 0, "Python compileall")


def validate_jinja() -> bool:
    environment = Environment(loader=FileSystemLoader(str(ROOT / "templates")))
    files = sorted((ROOT / "templates").glob("*.html"))
    failures = []
    for path in files:
        try:
            environment.parse(path.read_text(encoding="utf-8"))
        except Exception as exc:  # pragma: no cover - diagnostic path
            failures.append(f"{path.name}: {exc}")
    return result(not failures, "Jinja templates", f"{len(files)} parsed" if not failures else "; ".join(failures))


def _sanitize_jinja(js: str) -> str:
    js = re.sub(r"\{\{.*?\}\}", "x", js, flags=re.S)
    js = re.sub(r"\{%.*?%\}", "", js, flags=re.S)
    js = re.sub(r"\{#.*?#\}", "", js, flags=re.S)
    return js


def validate_inline_javascript() -> bool:
    node = subprocess.run(["bash", "-lc", "command -v node"], capture_output=True, text=True).stdout.strip()
    if not node:
        return result(True, "Inline JavaScript", "SKIP: Node.js not installed")

    blocks = []
    for template in sorted((ROOT / "templates").glob("*.html")):
        text = template.read_text(encoding="utf-8")
        matches = re.finditer(
            r"<script(?P<attrs>[^>]*)>(?P<body>.*?)</script>",
            text,
            flags=re.S | re.I,
        )
        for index, match in enumerate(matches, start=1):
            attrs = match.group("attrs") or ""
            block = match.group("body") or ""
            # Import maps and JSON configuration blocks are data, not
            # executable JavaScript. External files are validated separately.
            type_match = re.search(r'type=["\']([^"\']+)["\']', attrs, flags=re.I)
            script_type = type_match.group(1).strip().lower() if type_match else ""
            if re.search(r'\bsrc=["\']', attrs, flags=re.I):
                continue
            if script_type and script_type not in {"text/javascript", "application/javascript", "module"}:
                continue
            if block.strip():
                blocks.append((template.name, index, _sanitize_jinja(block)))

    failures = []
    for name, index, block in blocks:
        with tempfile.NamedTemporaryFile("w", suffix=".js", encoding="utf-8", delete=False) as handle:
            handle.write(block)
            temp_path = Path(handle.name)
        completed = subprocess.run([node, "--check", str(temp_path)], capture_output=True, text=True, check=False)
        temp_path.unlink(missing_ok=True)
        if completed.returncode != 0:
            failures.append(f"{name}#{index}: {completed.stderr.strip()}")
    return result(not failures, "Inline JavaScript", f"{len(blocks)} blocks" if not failures else " | ".join(failures))


def validate_database() -> bool:
    with tempfile.TemporaryDirectory() as temp_dir:
        previous = os.environ.get("FITMOTION_DATA_DIR")
        os.environ["FITMOTION_DATA_DIR"] = temp_dir
        try:
            _, path, _ = prepare_persistent_database(ROOT)
        finally:
            if previous is None:
                os.environ.pop("FITMOTION_DATA_DIR", None)
            else:
                os.environ["FITMOTION_DATA_DIR"] = previous
        connection = sqlite3.connect(path)
        try:
            integrity = connection.execute("PRAGMA integrity_check").fetchone()[0]
            draft_exists = bool(connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='workout_session_drafts'"
            ).fetchone())
            draft_columns = {row[1] for row in connection.execute("PRAGMA table_info(workout_session_drafts)")}
            schedule_item_columns = {row[1] for row in connection.execute("PRAGMA table_info(workout_schedule_items)")}
            level_config_columns = {row[1] for row in connection.execute("PRAGMA table_info(exercise_level_configs)")}
        finally:
            connection.close()

    # workout_sessions is created by SQLAlchemy at app boot, while this script
    # intentionally runs without Flask. Verify its ORM evidence fields in source
    # and verify the storage-only tables directly in SQLite.
    model_source = (ROOT / "models.py").read_text(encoding="utf-8")
    session_fields = {
        "keypoint_confidence_avg", "quality_score_avg", "rep_details_json",
        "error_codes_json", "training_level", "set_count", "rep_target",
        "duration_seconds", "effectiveness_status", "session_summary_json",
    }
    orm_session_ok = all(name in model_source for name in session_fields)
    level_storage_ok = (
        "training_level" in draft_columns
        and "training_level" in schedule_item_columns
        and {
            "training_level", "set_count", "rep_target", "rest_seconds",
            "max_idle_seconds", "max_session_seconds", "min_good_rep_ratio",
            "min_quality_score", "min_confidence", "min_stability_score",
            "max_tracking_abort_count",
        }.issubset(level_config_columns)
    )
    return result(
        integrity == "ok" and draft_exists and level_storage_ok and orm_session_ok,
        "SQLite persistent integrity/schema",
        f"integrity={integrity}, draft_table={draft_exists}, level_storage={level_storage_ok}, orm_session={orm_session_ok}",
    )


def validate_json_data() -> bool:
    files = sorted((ROOT / "data" / "validation" / "week6").glob("*.json"))
    failures = []
    for path in files:
        try:
            json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            failures.append(f"{path.name}: {exc}")
    return result(not failures, "Validation JSON", f"{len(files)} files" if not failures else "; ".join(failures))


def validate_single_pipeline() -> bool:
    app_source = (ROOT / "app.py").read_text(encoding="utf-8")
    web_source = (ROOT / "webapp.py").read_text(encoding="utf-8")
    video_source = (ROOT / "video_processor.py").read_text(encoding="utf-8")
    ok = (
        "return LearnedExerciseProcessor(slug, shared_state)" in app_source
        and "from app import app" in web_source
        and "from standalone_runner import create_processor" in video_source
        and 'data.get("total_rep"' not in app_source[app_source.index("def save_session_api"):app_source.index("def live_workout_api")]
    )
    return result(ok, "Canonical pipeline and server-authoritative metrics")


def main() -> int:
    checks = [
        validate_python(),
        validate_jinja(),
        validate_inline_javascript(),
        validate_database(),
        validate_json_data(),
        validate_single_pipeline(),
    ]
    print("\nSTATIC VALIDATION:", "PASS" if all(checks) else "FAIL")
    return 0 if all(checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
