from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

from jsonschema import Draft202012Validator

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from output_parser import deterministic_unfence, parse_and_validate  # noqa: E402
from pilot_utils import (  # noqa: E402
    FORMAL_DOCUMENT_CONDITION_COUNT,
    FORMAL_OUTPUT_COUNT,
    PRIMARY_FIELD_PATHS,
    compare_prediction,
    ground_truth_from_case,
    load_cases,
    read_json,
)
from run_inference import extract_assistant_text, manifest_rows, output_paths  # noqa: E402


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
    parsed, metadata = parse_and_validate(json.dumps(truth))
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


def test_formal_matrix_excludes_entire_development_case() -> None:
    formal = manifest_rows("all", "formal")
    development = manifest_rows("all", "development")
    assert len(formal) == FORMAL_DOCUMENT_CONDITION_COUNT
    assert len(formal) * 2 == FORMAL_OUTPUT_COUNT
    assert {row["semantic_case_id"] for row in development} == {"MEL-001"}
    assert len(development) == 4
    assert all(row["semantic_case_id"] != "MEL-001" for row in formal)


def test_development_outputs_are_routed_outside_formal_results() -> None:
    formal = output_paths("google/medgemma-1.5-4b-it", "clean", "MEL-002-A")
    development = output_paths(
        "google/medgemma-1.5-4b-it",
        "clean",
        "MEL-001-A",
        scope="development",
    )
    assert "results/raw/" in formal["raw"].as_posix()
    assert "results/development/raw/" in development["raw"].as_posix()


def test_generated_assistant_text_supports_string_and_typed_content() -> None:
    assert extract_assistant_text([{"role": "assistant", "content": '{"a": 1}'}]) == '{"a": 1}'
    assert (
        extract_assistant_text(
            [
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "{"},
                        {"type": "text", "text": '"a": 1}'},
                    ],
                }
            ]
        )
        == '{"a": 1}'
    )


def test_missing_key_does_not_score_as_explicit_null() -> None:
    truth = ground_truth_from_case(load_cases()[-1], "A")
    prediction = copy.deepcopy(truth)
    del prediction["staging"]["pN"]
    metrics = compare_prediction(truth, prediction)
    assert metrics["field_exact_correct"] == len(PRIMARY_FIELD_PATHS) - 1
    assert "staging.pN" in metrics["mismatched_fields"]


def test_boolean_does_not_equal_numeric_zero() -> None:
    truth = ground_truth_from_case(load_cases()[0], "A")
    assert truth["mitotic_rate_per_mm2"] == 0.0
    prediction = copy.deepcopy(truth)
    prediction["mitotic_rate_per_mm2"] = False
    metrics = compare_prediction(truth, prediction)
    assert "mitotic_rate_per_mm2" in metrics["mismatched_fields"]


def test_parser_rejects_duplicate_keys_and_non_finite_numbers() -> None:
    parsed, duplicate = parse_and_validate('{"document_id":"A","document_id":"B"}')
    assert parsed is None
    assert "Duplicate JSON key" in duplicate["parse_error"]

    parsed, non_finite = parse_and_validate('{"value": NaN}')
    assert parsed is None
    assert "Non-finite JSON number" in non_finite["parse_error"]


def test_parser_does_not_strip_reasoning_envelopes() -> None:
    raw = '<unused94>thought\nanalysis<unused95>{"document_id":"MEL-002-A"}'
    parsed, metadata = parse_and_validate(raw)
    assert parsed is None
    assert metadata["parse_valid"] is False
