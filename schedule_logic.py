"""Pure scheduling helpers for FitMotion AI.

This module intentionally has no Flask/SQLAlchemy imports so recurrence and
calendar behaviour can be regression-tested without booting the web app.
"""
from __future__ import annotations

import calendar
from datetime import date, datetime, timedelta
from typing import Iterable, Sequence

WEEKDAY_LABELS = (
    "Thứ Hai",
    "Thứ Ba",
    "Thứ Tư",
    "Thứ Năm",
    "Thứ Sáu",
    "Thứ Bảy",
    "Chủ Nhật",
)


def parse_iso_date(value: str) -> date:
    """Parse YYYY-MM-DD or raise ValueError with a stable message."""
    try:
        return datetime.strptime(str(value or "").strip(), "%Y-%m-%d").date()
    except ValueError as exc:
        raise ValueError("Ngày phải có định dạng YYYY-MM-DD.") from exc


def validate_schedule_range(start: date, end: date, max_days: int = 366) -> None:
    if end < start:
        raise ValueError("Ngày kết thúc phải bằng hoặc sau ngày bắt đầu.")
    span_days = (end - start).days + 1
    if span_days > max_days:
        raise ValueError(f"Một lịch tập chỉ được kéo dài tối đa {max_days} ngày.")


def recurrence_dates(start: date, end: date, weekdays: Iterable[int]) -> list[date]:
    """Return every date in range whose weekday is selected (Monday=0)."""
    normalized = sorted({int(item) for item in weekdays})
    if any(item < 0 or item > 6 for item in normalized):
        raise ValueError("Thứ trong tuần phải nằm trong khoảng 0 đến 6.")
    if not normalized:
        return []

    result: list[date] = []
    current = start
    while current <= end:
        if current.weekday() in normalized:
            result.append(current)
        current += timedelta(days=1)
    return result


def parse_month(value: str | None, today: date | None = None) -> tuple[int, int]:
    today = today or date.today()
    raw = str(value or "").strip()
    if not raw:
        return today.year, today.month
    try:
        parsed = datetime.strptime(raw, "%Y-%m")
    except ValueError as exc:
        raise ValueError("Tháng phải có định dạng YYYY-MM.") from exc
    return parsed.year, parsed.month


def shift_month(year: int, month: int, delta: int) -> tuple[int, int]:
    absolute = year * 12 + (month - 1) + int(delta)
    return absolute // 12, absolute % 12 + 1


def month_bounds(year: int, month: int) -> tuple[date, date]:
    last_day = calendar.monthrange(year, month)[1]
    return date(year, month, 1), date(year, month, last_day)


def build_month_grid(year: int, month: int) -> list[list[date | None]]:
    """Build a Monday-first rectangular calendar grid."""
    matrix = calendar.Calendar(firstweekday=0).monthdatescalendar(year, month)
    return [
        [cell if cell.month == month else None for cell in week]
        for week in matrix
    ]


def summarize_weekdays(weekdays: Sequence[int]) -> str:
    labels = [WEEKDAY_LABELS[int(day)] for day in sorted(set(weekdays))]
    return ", ".join(labels)
