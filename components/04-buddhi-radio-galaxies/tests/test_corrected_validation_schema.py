"""Exercise the actual Step 20 preflight schema assertion without importing it.

AST extraction prevents checkpoint/dataset loading or inference during this test.
"""
import ast
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / (
    "analysis_reports/step20_corrected_validation_classification/evaluate_fixed_checkpoint.py"
)


def validate_mapping(mapping):
    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
    checks = [node for node in ast.walk(tree) if isinstance(node, ast.Assert)
              and isinstance(node.test, ast.Compare)
              and ast.unparse(node.test.left) == "manifest['class_mapping']"]
    assert len(checks) == 1
    code = compile(ast.fix_missing_locations(ast.Module(body=checks, type_ignores=[])),
                   str(SCRIPT), "exec")
    exec(code, {"manifest": {"class_mapping": mapping}})


def test_frozen_index_string_to_class_name_schema_passes():
    validate_mapping({"0": "FR-I", "1": "FR-II"})


@pytest.mark.parametrize("mapping", [
    {"FR-I": 0, "FR-II": 1},
    {"0": "FR-II", "1": "FR-I"},
    {0: "FR-I", 1: "FR-II"},
    {"0": "FR-I"},
    {"0": "FR-I", "1": "FR-II", "2": "unknown"},
    None,
])
def test_wrong_schema_or_class_definitions_fail(mapping):
    with pytest.raises(AssertionError):
        validate_mapping(mapping)
