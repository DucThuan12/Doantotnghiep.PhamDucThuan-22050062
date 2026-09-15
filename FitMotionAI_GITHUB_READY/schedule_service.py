"""Database services for durable weekly/monthly workout schedules."""
from __future__ import annotations

from collections import defaultdict
from datetime import date, timedelta

from database import db
from models import (
    WorkoutExercise,
    WorkoutPlan,
    WorkoutSchedule,
    WorkoutScheduleItem,
    WorkoutSession,
    WorkoutProgram,
)
from schedule_logic import (
    WEEKDAY_LABELS,
    build_month_grid,
    month_bounds,
    recurrence_dates,
)
from training_levels import DEFAULT_LEVEL, normalize_training_level, training_level_label


def materialize_schedule(schedule: WorkoutSchedule, from_date: date | None = None) -> int:
    """Create durable WorkoutPlan occurrences for every configured weekday.

    Existing rows are kept. This makes the operation idempotent and safe to run
    after app restarts or when a user revisits the schedule screen.
    """
    if schedule.status != "active":
        return 0

    start = date.fromisoformat(schedule.start_date)
    end = date.fromisoformat(schedule.end_date)
    if from_date and from_date > start:
        start = from_date
    if end < start:
        return 0

    created = 0
    for item in schedule.items:
        for occurrence_date in recurrence_dates(start, end, [item.weekday]):
            day_text = occurrence_date.isoformat()
            exists = WorkoutPlan.query.filter_by(
                user_id=schedule.user_id,
                exercise_id=item.exercise_id,
                workout_date=day_text,
            ).order_by(WorkoutPlan.created_at.asc()).first()
            if exists:
                # Adopt an old manual pending plan rather than creating a
                # duplicate. Plans belonging to another recurring schedule are
                # preserved and the overlapping occurrence is skipped.
                if exists.schedule_id is None and (exists.status or "pending") == "pending":
                    exists.schedule_id = schedule.id
                    exists.set_count = item.set_count
                    exists.rep_target = item.rep_target
                    exists.training_level = normalize_training_level(getattr(item, "training_level", DEFAULT_LEVEL))
                continue
            db.session.add(WorkoutPlan(
                user_id=schedule.user_id,
                exercise_id=item.exercise_id,
                workout_date=day_text,
                set_count=item.set_count,
                rep_target=item.rep_target,
                training_level=normalize_training_level(getattr(item, "training_level", DEFAULT_LEVEL)),
                status="pending",
                schedule_id=schedule.id,
            ))
            created += 1
    return created


def replace_future_occurrences(schedule: WorkoutSchedule, today: date | None = None) -> int:
    """Regenerate future pending occurrences while preserving history."""
    today = today or date.today()
    WorkoutPlan.query.filter(
        WorkoutPlan.schedule_id == schedule.id,
        WorkoutPlan.workout_date >= today.isoformat(),
        WorkoutPlan.status == "pending",
    ).delete(synchronize_session=False)
    return materialize_schedule(schedule, from_date=today)


def archive_schedule(schedule: WorkoutSchedule, today: date | None = None) -> None:
    today = today or date.today()
    schedule.status = "archived"
    WorkoutPlan.query.filter(
        WorkoutPlan.schedule_id == schedule.id,
        WorkoutPlan.workout_date >= today.isoformat(),
        WorkoutPlan.status == "pending",
    ).delete(synchronize_session=False)


def get_user_schedules(user_id: int) -> list[WorkoutSchedule]:
    schedules = WorkoutSchedule.query.filter_by(user_id=user_id).order_by(
        WorkoutSchedule.status.asc(),
        WorkoutSchedule.start_date.desc(),
        WorkoutSchedule.id.desc(),
    ).all()
    for schedule in schedules:
        # Keep future materialized occurrences healthy after source updates.
        materialize_schedule(schedule, from_date=date.today())
    return schedules


def schedule_to_dict(schedule: WorkoutSchedule, exercise_map: dict[int, WorkoutExercise]) -> dict:
    grouped: dict[int, list[dict]] = defaultdict(list)
    source_program = db.session.get(WorkoutProgram, schedule.source_program_id) if getattr(schedule, "source_program_id", None) else None
    for item in schedule.items:
        exercise = exercise_map.get(item.exercise_id)
        grouped[item.weekday].append({
            "id": item.id,
            "exercise_id": item.exercise_id,
            "exercise_name": exercise.name if exercise else "Bài tập đã ẩn",
            "set_count": item.set_count,
            "rep_target": item.rep_target,
            "training_level": normalize_training_level(getattr(item, "training_level", DEFAULT_LEVEL)),
            "training_level_label": training_level_label(getattr(item, "training_level", DEFAULT_LEVEL)),
        })
    return {
        "id": schedule.id,
        "name": schedule.name,
        "start_date": schedule.start_date,
        "end_date": schedule.end_date,
        "status": schedule.status,
        "source_program_id": getattr(schedule, "source_program_id", None),
        "source_program_title": source_program.title if source_program else "",
        "weekday_groups": [
            {
                "weekday": weekday,
                "weekday_label": WEEKDAY_LABELS[weekday],
                "exercises": grouped[weekday],
            }
            for weekday in sorted(grouped)
        ],
    }


def build_month_calendar(user_id: int, year: int, month: int) -> dict:
    start, end = month_bounds(year, month)
    plans = WorkoutPlan.query.filter(
        WorkoutPlan.user_id == user_id,
        WorkoutPlan.workout_date >= start.isoformat(),
        WorkoutPlan.workout_date <= end.isoformat(),
    ).order_by(WorkoutPlan.workout_date.asc(), WorkoutPlan.id.asc()).all()
    sessions = WorkoutSession.query.filter(
        WorkoutSession.user_id == user_id,
        WorkoutSession.session_date >= start.isoformat(),
        WorkoutSession.session_date <= end.isoformat(),
    ).all()

    exercise_ids = {item.exercise_id for item in plans}
    exercise_ids.update(item.exercise_id for item in sessions)
    exercises = WorkoutExercise.query.filter(WorkoutExercise.id.in_(exercise_ids)).all() if exercise_ids else []
    exercise_map = {item.id: item for item in exercises}

    completed_keys = {(item.session_date, item.exercise_id) for item in sessions}
    day_events: dict[str, list[dict]] = defaultdict(list)
    today = date.today()

    for plan in plans:
        event_date = date.fromisoformat(plan.workout_date)
        status = plan.status or "pending"
        if (plan.workout_date, plan.exercise_id) in completed_keys and status not in {"partial", "completed"}:
            status = "completed"
        elif event_date < today and status == "pending":
            status = "missed"
        exercise = exercise_map.get(plan.exercise_id)
        day_events[plan.workout_date].append({
            "plan_id": plan.id,
            "schedule_id": plan.schedule_id,
            "exercise_id": plan.exercise_id,
            "exercise_name": exercise.name if exercise else "Bài tập đã ẩn",
            "exercise_slug": exercise.slug if exercise else "",
            "set_count": plan.set_count,
            "rep_target": plan.rep_target,
            "training_level": normalize_training_level(getattr(plan, "training_level", DEFAULT_LEVEL)),
            "training_level_label": training_level_label(getattr(plan, "training_level", DEFAULT_LEVEL)),
            "status": status,
        })

    # Show ad-hoc completed sessions even if no plan was created for that day.
    planned_keys = {(item.workout_date, item.exercise_id) for item in plans}
    for session in sessions:
        key = (session.session_date, session.exercise_id)
        if key in planned_keys:
            continue
        exercise = exercise_map.get(session.exercise_id)
        day_events[session.session_date].append({
            "plan_id": None,
            "schedule_id": getattr(session, "schedule_id", None),
            "exercise_id": session.exercise_id,
            "exercise_name": exercise.name if exercise else "Bài tập đã ẩn",
            "exercise_slug": exercise.slug if exercise else "",
            "set_count": max(1, int(getattr(session, "set_count", 1) or 1)),
            "rep_target": max(1, int(getattr(session, "rep_target", session.total_rep or 1) or 1)),
            "training_level": normalize_training_level(getattr(session, "training_level", DEFAULT_LEVEL)),
            "training_level_label": training_level_label(getattr(session, "training_level", DEFAULT_LEVEL)),
            "status": "completed",
        })

    weeks = []
    for week in build_month_grid(year, month):
        rendered_week = []
        for cell in week:
            if cell is None:
                rendered_week.append(None)
                continue
            day_text = cell.isoformat()
            rendered_week.append({
                "date": day_text,
                "day": cell.day,
                "is_today": cell == today,
                "events": day_events.get(day_text, []),
            })
        weeks.append(rendered_week)

    return {
        "year": year,
        "month": month,
        "month_key": f"{year:04d}-{month:02d}",
        "label": f"Tháng {month:02d}/{year}",
        "weeks": weeks,
    }
