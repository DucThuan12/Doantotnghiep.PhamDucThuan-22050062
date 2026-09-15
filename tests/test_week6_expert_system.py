from __future__ import annotations

import unittest

from expert_system import (
    build_violation,
    normalize_error_candidates,
    select_primary_violation,
    serialize_violations,
)


class ExpertSystemTests(unittest.TestCase):
    def test_squat_safety_error_has_priority_over_depth(self):
        violations = normalize_error_candidates("squat", [
            ("notlow", "Nông", "Hạ thấp hơn", "middle"),
            ("backlean", "Nghiêng lưng", "Giữ lưng thẳng", "middle"),
        ])
        primary = select_primary_violation(violations)
        self.assertEqual(primary.code, "backlean")
        self.assertEqual([item.code for item in violations], ["backlean", "notlow"])

    def test_duplicate_code_is_recorded_once(self):
        violations = normalize_error_candidates("pushup", [
            ("bodyline", "Sai thân", "Giữ thẳng", "middle"),
            ("bodyline", "Lặp lại", "Lặp lại", "middle"),
        ])
        self.assertEqual(len(violations), 1)
        self.assertEqual(violations[0].title, "Sai thân")

    def test_runtime_wording_is_preserved(self):
        violation = build_violation("curl", "notbend", "GẬP CHƯA ĐỦ", "GẬP THÊM", "middle")
        self.assertEqual(violation.title, "GẬP CHƯA ĐỦ")
        self.assertEqual(violation.advice, "GẬP THÊM")
        self.assertEqual(violation.priority, 30)

    def test_unknown_rule_has_safe_fallback(self):
        violation = build_violation("squat", "unknown-code")
        self.assertEqual(violation.code, "unknown-code")
        self.assertEqual(violation.priority, 80)
        self.assertTrue(violation.advice)

    def test_serialized_violation_contains_auditable_fields(self):
        payload = serialize_violations([build_violation("pushup", "bodyline")])[0]
        self.assertEqual(payload["code"], "bodyline")
        self.assertEqual(payload["exercise"], "pushup")
        self.assertIn("priority", payload)
        self.assertIn("phase", payload)
        self.assertIn("severity", payload)


if __name__ == "__main__":
    unittest.main()
