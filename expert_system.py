"""Rule-based expert feedback for FitMotion AI.

The design follows the thesis principle used by the benchmark project:
exercise measurements are converted into explicit rule violations, then the
system selects one high-priority message to speak after a repetition while
retaining every violation for reports and later evaluation.

This module contains no pose-estimation logic. It only normalizes, prioritizes
and serializes rule results so the inference pipeline remains testable.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable, Mapping, Sequence


@dataclass(frozen=True)
class RuleDefinition:
    code: str
    exercise: str
    phase: str
    priority: int
    severity: str
    title: str
    advice: str


@dataclass(frozen=True)
class RuleViolation:
    code: str
    exercise: str
    phase: str
    priority: int
    severity: str
    title: str
    advice: str

    def to_dict(self) -> dict:
        return asdict(self)


_RULES: tuple[RuleDefinition, ...] = (
    RuleDefinition("backlean", "squat", "middle", 10, "high", "LUNG DO VE PHIA TRUOC", "GIU NGUC MO VA LUNG THANG HON"),
    RuleDefinition("notlow", "squat", "middle", 20, "medium", "XUONG CHUA DU THAP", "HA THAP HONG XUONG THEM"),
    RuleDefinition("bodyline", "pushup", "middle", 10, "high", "THAN NGUOI CHUA THANG", "GIU THANG VAI - HONG - CHAN"),
    RuleDefinition("notlow", "pushup", "middle", 20, "medium", "HA NGUOI CHUA DU THAP", "GAP KHUYU TAY SAU HON TRUOC KHI DAY LEN"),
    RuleDefinition("elbowshift", "curl", "middle", 10, "high", "KHUYU TAY BI LECH", "GIU KHUYU TAY GAN THAN NGUOI"),
    RuleDefinition("notstraight", "curl", "end", 20, "medium", "CHUA VE GAN GOC BAT DAU", "HA TA VE GAN TU THE BAT DAU THOAI MAI, KHONG KHOA CUNG KHUYU"),
    RuleDefinition("notbend", "curl", "middle", 30, "medium", "GAP TAY CHUA DU", "NANG TA LEN CAO HON"),
    RuleDefinition("qualitylow", "common", "middle", 90, "low", "DIEM CHAT LUONG CHUA DAT", "GIAM TOC DO VA KIEM SOAT BIEN DO"),
    RuleDefinition("tracking_lost", "common", "middle", 5, "high", "MAT DIEM KHOP", "TRO LAI KHUNG HINH VA THUC HIEN LAI"),
    RuleDefinition("wrongside", "curl", "middle", 4, "high", "SAI BEN TAY", "HAY TAP DUNG BEN TAY DA CHON VA GIU TAY DOI DIEN YEN"),
    RuleDefinition("exercise_mismatch", "common", "middle", 6, "high", "CHUA XAC NHAN DUNG BAI TAP", "HAY THUC HIEN LAI DUNG BAI VA DUNG PHA DONG TAC"),
    RuleDefinition("reference_mismatch", "common", "middle", 12, "medium", "CHUA KHOP DONG TAC MAU 3D", "DIEU CHINH Nhip VA BIEN DO DE GAN VOI ANIMATION THAM CHIEU"),
)

ERROR_CATALOG: Mapping[tuple[str, str], RuleDefinition] = {
    (rule.exercise, rule.code): rule for rule in _RULES
}


def get_rule(exercise: str, code: str) -> RuleDefinition | None:
    """Find an exercise-specific rule, then a common fallback."""
    exercise = str(exercise or "common").lower()
    code = str(code or "unknown").lower()
    return ERROR_CATALOG.get((exercise, code)) or ERROR_CATALOG.get(("common", code))


def build_violation(
    exercise: str,
    code: str,
    title: str | None = None,
    advice: str | None = None,
    phase: str | None = None,
) -> RuleViolation:
    """Create a normalized violation without discarding runtime messages."""
    rule = get_rule(exercise, code)
    if rule is None:
        return RuleViolation(
            code=str(code),
            exercise=str(exercise),
            phase=str(phase or "middle"),
            priority=80,
            severity="medium",
            title=str(title or code),
            advice=str(advice or "Kiểm tra lại kỹ thuật thực hiện."),
        )

    return RuleViolation(
        code=rule.code,
        exercise=str(exercise),
        phase=str(phase or rule.phase),
        priority=int(rule.priority),
        severity=rule.severity,
        title=str(title or rule.title),
        advice=str(advice or rule.advice),
    )


def normalize_error_candidates(exercise: str, candidates: Iterable[Sequence]) -> list[RuleViolation]:
    """Normalize tuple candidates and remove duplicate error codes.

    Accepted tuple shape is ``(code, title, advice, phase)``. The first
    occurrence supplies the runtime wording, while catalog priority controls
    which message is spoken to the user.
    """
    unique: dict[str, RuleViolation] = {}
    for candidate in candidates:
        if not candidate:
            continue
        values = list(candidate)
        code = str(values[0])
        title = str(values[1]) if len(values) > 1 else None
        advice = str(values[2]) if len(values) > 2 else None
        phase = str(values[3]) if len(values) > 3 else None
        if code not in unique:
            unique[code] = build_violation(exercise, code, title, advice, phase)
    return sorted(unique.values(), key=lambda item: (item.priority, item.code))


def select_primary_violation(violations: Iterable[RuleViolation]) -> RuleViolation | None:
    items = list(violations)
    if not items:
        return None
    return min(items, key=lambda item: (item.priority, item.code))


def serialize_violations(violations: Iterable[RuleViolation]) -> list[dict]:
    return [item.to_dict() for item in violations]
