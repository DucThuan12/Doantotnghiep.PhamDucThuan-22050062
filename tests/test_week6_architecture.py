from __future__ import annotations

import ast
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class ArchitectureConsistencyTests(unittest.TestCase):
    def test_legacy_web_entrypoint_reuses_canonical_app(self):
        source = (ROOT / "webapp.py").read_text(encoding="utf-8")
        self.assertIn("from app import app", source)
        self.assertNotIn("return SquatProcessor()", source)

    def test_standalone_factory_uses_safe_unsupported_processor(self):
        tree = ast.parse((ROOT / "standalone_runner.py").read_text(encoding="utf-8"))
        factory = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "create_processor")
        final_return = factory.body[-1]
        self.assertIsInstance(final_return, ast.Return)
        self.assertEqual(final_return.value.func.id, "UnsupportedExerciseProcessor")

    def test_uploaded_video_uses_canonical_processor_factory(self):
        source = (ROOT / "video_processor.py").read_text(encoding="utf-8")
        self.assertIn("from standalone_runner import create_processor", source)
        self.assertNotIn("trangthai = \"UP\"", source)
        self.assertIn('"rep_records"', source)

    def test_compatibility_save_endpoint_does_not_trust_client_rep_counts(self):
        source = (ROOT / "app.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "save_session_api")
        function_source = ast.get_source_segment(source, function) or ""
        self.assertIn('shared_state.get("total_rep"', function_source)
        self.assertNotIn('data.get("total_rep"', function_source)
        self.assertNotIn('data.get("good_rep"', function_source)


if __name__ == "__main__":
    unittest.main()
