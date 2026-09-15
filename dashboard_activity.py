"""Pure helpers for weekly workout activity aggregation.

The dashboard must represent both what was planned and what was actually
performed.  Keeping this module free of Flask/SQLAlchemy makes the aggregation
rules independently testable.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable, Mapping


def _value(item: Any, name: str, default: Any = None) -> Any:
    if isinstance(item, Mapping):
        return item.get(name, default)
    return getattr(item, name, default)


def build_week_activity(
    week_dates: Iterable[str],
    plans: Iterable[Any],
    sessions: Iterable[Any],
    drafts: Iterable[Any],
    live_items: Iterable[Any],
    exercise_calories: Mapping[int, float],
) -> dict[str, dict[str, Any]]:
    """Aggregate planned and actual activity per day without double counting.

    A day card counts distinct exercises from any source.  Calories use actual
    reps when a session/draft/live workout exists for an exercise; otherwise
    they use the planned target.  This ensures an ad-hoc completed workout is
    visible even if no plan row existed before the session started.
    """
    activity: dict[str, dict[str, Any]] = {
        str(day): {
            "exercise_ids": set(),
            "planned_kcal": defaultdict(float),
            "actual_kcal": defaultdict(float),
            "completed_exercise_ids": set(),
        }
        for day in week_dates
    }

    def ensure(day: str) -> dict[str, Any] | None:
        return activity.get(str(day))

    for plan in plans:
        day = str(_value(plan, "workout_date", "") or "")
        bucket = ensure(day)
        if bucket is None:
            continue
        exercise_id = int(_value(plan, "exercise_id", 0) or 0)
        if exercise_id <= 0:
            continue
        reps_per_set = max(0, int(_value(plan, "rep_target", 0) or 0))
        set_count = max(1, int(_value(plan, "set_count", 1) or 1))
        target = reps_per_set * set_count
        bucket["exercise_ids"].add(exercise_id)
        bucket["planned_kcal"][exercise_id] += max(
            0.0, float(exercise_calories.get(exercise_id, 0.0) or 0.0)
        ) * target

    def add_actual(items: Iterable[Any], date_field: str) -> None:
        for item in items:
            day = str(_value(item, date_field, "") or "")
            bucket = ensure(day)
            if bucket is None:
                continue
            exercise_id = int(_value(item, "exercise_id", 0) or 0)
            if exercise_id <= 0:
                continue
            reps = max(0, int(_value(item, "total_rep", 0) or 0))
            bucket["exercise_ids"].add(exercise_id)
            if reps > 0:
                bucket["completed_exercise_ids"].add(exercise_id)
                bucket["actual_kcal"][exercise_id] += max(
                    0.0, float(exercise_calories.get(exercise_id, 0.0) or 0.0)
                ) * reps

    add_actual(sessions, "session_date")
    add_actual(drafts, "session_date")
    add_actual(live_items, "session_date")

    result: dict[str, dict[str, Any]] = {}
    for day, bucket in activity.items():
        kcal = 0.0
        for exercise_id in bucket["exercise_ids"]:
            actual = float(bucket["actual_kcal"].get(exercise_id, 0.0) or 0.0)
            planned = float(bucket["planned_kcal"].get(exercise_id, 0.0) or 0.0)
            kcal += actual if actual > 0 else planned
        result[day] = {
            "count": len(bucket["exercise_ids"]),
            "completed_count": len(bucket["completed_exercise_ids"]),
            "kcal": round(kcal, 2),
        }
    return result

def build_schedule_adherence(
    plans: Iterable[Any],
    sessions: Iterable[Any],
    drafts: Iterable[Any],
    live_items: Iterable[Any],
    today: str,
) -> dict[str, Any]:
    """Summarize recurring-schedule adherence from one shared data model.

    Only plans linked to ``workout_schedules`` are counted as registered
    schedule items. A finalized session, durable draft, or live workout with
    at least one rep marks the matching date/exercise as performed.
    """
    planned_keys: set[tuple[str, int]] = set()
    for plan in plans:
        schedule_id = int(_value(plan, "schedule_id", 0) or 0)
        exercise_id = int(_value(plan, "exercise_id", 0) or 0)
        day = str(_value(plan, "workout_date", "") or "")
        if schedule_id > 0 and exercise_id > 0 and day:
            planned_keys.add((day, exercise_id))

    performed_keys: set[tuple[str, int]] = set()

    def add_performed(items: Iterable[Any], date_field: str) -> None:
        for item in items:
            exercise_id = int(_value(item, "exercise_id", 0) or 0)
            day = str(_value(item, date_field, "") or "")
            reps = int(_value(item, "total_rep", 0) or 0)
            if exercise_id > 0 and day and reps > 0:
                performed_keys.add((day, exercise_id))

    add_performed(sessions, "session_date")
    add_performed(drafts, "session_date")
    add_performed(live_items, "session_date")

    performed_planned = planned_keys & performed_keys
    missed = {key for key in planned_keys if key[0] < str(today) and key not in performed_planned}
    upcoming = planned_keys - performed_planned - missed
    planned_count = len(planned_keys)
    performed_count = len(performed_planned)
    return {
        "planned_count": planned_count,
        "performed_count": performed_count,
        "missed_count": len(missed),
        "upcoming_count": len(upcoming),
        "completion_percent": round((performed_count / planned_count) * 100, 2) if planned_count else 0.0,
    }

