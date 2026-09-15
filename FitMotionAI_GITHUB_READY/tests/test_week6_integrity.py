from __future__ import annotations

import ast
import json
import sqlite3
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class SourceIntegrityTests(unittest.TestCase):
    def test_good_rep_normalization_only_clamps(self):
        source = (ROOT / "app.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        node = next(item for item in tree.body if isinstance(item, ast.FunctionDef) and item.name == "normalize_display_good_rep")
        module = ast.Module(body=[node], type_ignores=[])
        namespace = {}
        exec(compile(module, "app.py", "exec"), namespace)
        normalize = namespace["normalize_display_good_rep"]
        self.assertEqual(normalize(10, 2), 2)
        self.assertEqual(normalize(10, 12), 10)
        self.assertEqual(normalize(10, -3), 0)

    def test_unknown_exercise_never_falls_back_to_squat(self):
        source = (ROOT / "app.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        node = next(item for item in tree.body if isinstance(item, ast.FunctionDef) and item.name == "_build_processor")
        final_return = node.body[-1]
        self.assertIsInstance(final_return, ast.Return)
        self.assertEqual(final_return.value.func.id, "LearnedExerciseProcessor")
        self.assertNotEqual(final_return.value.func.id, "SquatProcessor")

    def test_database_contains_week6_columns(self):
        # The production DB is intentionally stored outside the source tree, so
        # source-integrity tests must not require a bundled aifitness.db file.
        source = (ROOT / "models.py").read_text(encoding="utf-8")
        expected = {
            "keypoint_confidence_avg",
            "quality_score_avg",
            "rep_details_json",
            "error_codes_json",
        }
        self.assertTrue(all(name in source for name in expected))

    def test_models_define_separate_confidence_and_quality(self):
        source = (ROOT / "models.py").read_text(encoding="utf-8")
        self.assertIn("keypoint_confidence_avg", source)
        self.assertIn("quality_score_avg", source)
        self.assertIn("rep_details_json", source)

    def test_rep_record_schema_is_json_serializable(self):
        record = {
            "index": 1,
            "score": 84,
            "is_good": True,
            "confidence": 0.91,
            "stability": 88.0,
            "components": {"depth": 90},
            "error_codes": [],
        }
        encoded = json.dumps([record], ensure_ascii=False)
        self.assertEqual(json.loads(encoded)[0]["score"], 84)

    def test_validation_dataset_has_required_scenarios(self):
        data = json.loads((ROOT / "data" / "validation" / "week6" / "scenarios.json").read_text(encoding="utf-8"))
        scenarios = data.get("scenarios", [])
        self.assertGreaterEqual(len(scenarios), 10)
        ids = {item["id"] for item in scenarios}
        required = {"SQ-GOOD-01", "SQ-SHALLOW-01", "PU-GOOD-01", "PU-BODY-01", "CU-GOOD-01", "CU-EXT-01", "TRACK-LOWCONF-01"}
        self.assertTrue(required.issubset(ids))
        for item in scenarios:
            self.assertIn("exercise", item)
            self.assertIn("expected_total_rep", item)
            self.assertIn("expected_good_rep", item)



if __name__ == "__main__":
    unittest.main()
