from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from evaluation_metrics import evaluate_csv, evaluate_rows


class EvaluationMetricTests(unittest.TestCase):
    def test_rep_and_error_metrics(self):
        rows = [
            {
                "gt_total_rep": "10",
                "pred_total_rep": "10",
                "gt_good_rep": "8",
                "pred_good_rep": "7",
                "gt_error_codes": "notlow;backlean",
                "pred_error_codes": "notlow",
                "avg_fps": "18",
                "avg_latency_ms": "45",
                "avg_keypoint_confidence": "0.9",
                "avg_quality_score": "82",
            },
            {
                "gt_total_rep": "5",
                "pred_total_rep": "4",
                "gt_good_rep": "4",
                "pred_good_rep": "4",
                "gt_error_codes": "bodyline",
                "pred_error_codes": "bodyline;notlow",
                "avg_fps": "20",
                "avg_latency_ms": "40",
                "avg_keypoint_confidence": "0.8",
                "avg_quality_score": "75",
            },
        ]
        summary = evaluate_rows(rows)
        self.assertEqual(summary.sample_count, 2)
        self.assertEqual(summary.rep_count_mae, 0.5)
        self.assertEqual(summary.rep_count_exact_accuracy, 0.5)
        self.assertEqual(summary.good_rep_mae, 0.5)
        self.assertAlmostEqual(summary.error_precision, 2 / 3, places=4)
        self.assertAlmostEqual(summary.error_recall, 2 / 3, places=4)
        self.assertEqual(summary.average_fps, 19.0)

    def test_empty_dataset_is_rejected(self):
        with self.assertRaises(ValueError):
            evaluate_rows([])

    def test_csv_loader(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.csv"
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=[
                    "gt_total_rep", "pred_total_rep", "gt_good_rep", "pred_good_rep",
                    "gt_error_codes", "pred_error_codes",
                ])
                writer.writeheader()
                writer.writerow({
                    "gt_total_rep": 3,
                    "pred_total_rep": 3,
                    "gt_good_rep": 2,
                    "pred_good_rep": 2,
                    "gt_error_codes": "notlow",
                    "pred_error_codes": "notlow",
                })
            summary = evaluate_csv(path)
            self.assertEqual(summary.rep_count_exact_accuracy, 1.0)
            self.assertEqual(summary.error_f1, 1.0)


if __name__ == "__main__":
    unittest.main()
