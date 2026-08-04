from __future__ import annotations

import sys
from pathlib import Path

from jsonschema import Draft202012Validator

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from output_parser import deterministic_unfence, parse_and_validate  # noqa: E402
from pilot_utils import (  # noqa: E402
    PRIMARY_FIELD_PATHS,
    compare_prediction,
    ground_truth_from_case,
    load_cases,
    read_json,
)


def test_case_matrix_is_eight_semantic_cases_by_two_templates() -> None:
    cases = load_cases()
    assert len(cases) == 8
    assert len({case["case_id"] for case in cases}) == 8


def test_generated_ground_truth_conforms_to_schema() -> None:
    schema = read_json(ROOT / "schema" / "extraction.schema.json")
    validator = Draft202012Validator(schema)
    for case in load_cases():
        for template_id in ("A", "B"):
            truth = ground_truth_from_case(case, template_id)
            assert list(validator.iter_errors(truth)) == []


def test_perfect_prediction_scores_all_atomic_fields() -> None:
    truth = ground_truth_from_case(load_cases()[0], "A")
    metrics = compare_prediction(truth, truth)
    assert metrics["field_exact_correct"] == len(PRIMARY_FIELD_PATHS)
    assert metrics["field_exact_match"] == 1.0
    assert metrics["f1"] == 1.0
    assert metrics["unsupported_count"] == 0
    assert metrics["document_id_correct"] is True
    assert metrics["complete_report"] is True


def test_unsupported_value_is_counted() -> None:
    truth = ground_truth_from_case(load_cases()[-1], "A")
    prediction = {
        **truth,
        "breslow_thickness_mm": 0.9,
        "breslow_qualifier": "exact",
        "staging": {**truth["staging"], "pT": "pT1b"},
    }
    metrics = compare_prediction(truth, prediction)
    assert metrics["unsupported_count"] == 3
    assert {
        "breslow_thickness_mm",
        "breslow_qualifier",
        "staging.pT",
    }.issubset(metrics["unsupported_fields"])
    assert metrics["complete_report"] is False


def test_only_one_outer_markdown_fence_is_removed() -> None:
    raw = '```json\n{"a": 1}\n```'
    assert deterministic_unfence(raw) == '{"a": 1}'
    assert deterministic_unfence('prefix {"a": 1}') == 'prefix {"a": 1}'


def test_schema_rejects_extra_properties() -> None:
    truth = ground_truth_from_case(load_cases()[0], "A")
    truth["invented"] = "not allowed"
    parsed, metadata = parse_and_validate(__import__("json").dumps(truth))
    assert parsed is not None
    assert metadata["parse_valid"] is True
    assert metadata["schema_valid"] is False


def test_wrong_document_id_prevents_complete_report() -> None:
    truth = ground_truth_from_case(load_cases()[0], "A")
    prediction = {**truth, "document_id": "MEL-999-A"}
    metrics = compare_prediction(truth, prediction)
    assert metrics["field_exact_match"] == 1.0
    assert metrics["document_id_correct"] is False
    assert "document_id" in metrics["mismatched_fields"]
    assert metrics["complete_report"] is False


def test_schema_requires_numeric_qualifier_pairs() -> None:
    schema = read_json(ROOT / "schema" / "extraction.schema.json")
    validator = Draft202012Validator(schema)
    truth = ground_truth_from_case(load_cases()[0], "A")

    missing_breslow_qualifier = {**truth, "breslow_qualifier": None}
    assert list(validator.iter_errors(missing_breslow_qualifier))

    missing_mitotic_value = {
        **truth,
        "mitotic_rate_per_mm2": None,
        "mitotic_qualifier": "exact",
    }
    assert list(validator.iter_errors(missing_mitotic_value))
