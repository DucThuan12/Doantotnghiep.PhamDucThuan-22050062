"""Evaluation helpers for the FitMotion AI experimental protocol.

This module does not fabricate model accuracy. It only evaluates a CSV that has
both ground-truth annotations and outputs collected from an actual run.
"""
from __future__ import annotations

import csv
import json
from collections import Counter
from dataclasses import dataclass, asdict
from pathlib import Path
from statistics import mean
from typing import Iterable


def _number(value, default=0.0) -> float:
    try:
        if value is None or str(value).strip() == "":
            return float(default)
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _labels(value: str | None) -> set[str]:
    if not value:
        return set()
    return {part.strip().lower() for part in str(value).split(";") if part.strip()}


@dataclass(frozen=True)
class EvaluationSummary:
    sample_count: int
    rep_count_mae: float
    rep_count_exact_accuracy: float
    good_rep_mae: float
    good_rep_exact_accuracy: float
    error_precision: float
    error_recall: float
    error_f1: float
    average_fps: float | None
    average_latency_ms: float | None
    average_keypoint_confidence: float | None
    average_quality_score: float | None

    def to_dict(self) -> dict:
        return asdict(self)


def evaluate_rows(rows: Iterable[dict]) -> EvaluationSummary:
    records = list(rows)
    if not records:
        raise ValueError("Không có dòng dữ liệu thực nghiệm để đánh giá.")

    rep_errors: list[float] = []
    rep_exact: list[float] = []
    good_errors: list[float] = []
    good_exact: list[float] = []
    true_positive = false_positive = false_negative = 0
    fps_values: list[float] = []
    latency_values: list[float] = []
    confidence_values: list[float] = []
    quality_values: list[float] = []

    for row in records:
        gt_total = int(round(_number(row.get("gt_total_rep"))))
        pred_total = int(round(_number(row.get("pred_total_rep"))))
        gt_good = int(round(_number(row.get("gt_good_rep"))))
        pred_good = int(round(_number(row.get("pred_good_rep"))))

        rep_errors.append(abs(pred_total - gt_total))
        rep_exact.append(float(pred_total == gt_total))
        good_errors.append(abs(pred_good - gt_good))
        good_exact.append(float(pred_good == gt_good))

        gt_labels = _labels(row.get("gt_error_codes"))
        pred_labels = _labels(row.get("pred_error_codes"))
        true_positive += len(gt_labels & pred_labels)
        false_positive += len(pred_labels - gt_labels)
        false_negative += len(gt_labels - pred_labels)

        optional_fields = [
            ("avg_fps", fps_values),
            ("avg_latency_ms", latency_values),
            ("avg_keypoint_confidence", confidence_values),
            ("avg_quality_score", quality_values),
        ]
        for field, bucket in optional_fields:
            value = row.get(field)
            if value is not None and str(value).strip() != "":
                bucket.append(_number(value))

    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    optional_mean = lambda values: round(mean(values), 4) if values else None
    return EvaluationSummary(
        sample_count=len(records),
        rep_count_mae=round(mean(rep_errors), 4),
        rep_count_exact_accuracy=round(mean(rep_exact), 4),
        good_rep_mae=round(mean(good_errors), 4),
        good_rep_exact_accuracy=round(mean(good_exact), 4),
        error_precision=round(precision, 4),
        error_recall=round(recall, 4),
        error_f1=round(f1, 4),
        average_fps=optional_mean(fps_values),
        average_latency_ms=optional_mean(latency_values),
        average_keypoint_confidence=optional_mean(confidence_values),
        average_quality_score=optional_mean(quality_values),
    )


def evaluate_csv(csv_path: str | Path) -> EvaluationSummary:
    path = Path(csv_path)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return evaluate_rows(rows)


def write_json(summary: EvaluationSummary, output_path: str | Path) -> None:
    Path(output_path).write_text(
        json.dumps(summary.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
