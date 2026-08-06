from __future__ import annotations

import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

from jsonschema import Draft202012Validator
from PIL import Image
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from generate_v3_reports import degradation_parameters as v3_degradation_parameters  # noqa: E402
from generate_v4_reports import (  # noqa: E402
    ALL_CASE_IDS,
    CASE_ORIGINS,
    CONDITIONS,
    DATASET_VERSION,
    DEVELOPMENT_CASE_IDS,
    FORMAL_CASE_IDS,
    GENERATOR_VERSION,
    NEW_ARTIFACT_ORIGIN,
    REUSED_ARTIFACT_ORIGIN,
    REUSED_V3_CASE_IDS,
    TEMPLATE_IDS,
    degradation_parameters,
    load_v4_cases,
)
from pilot_utils import (  # noqa: E402
    PRIMARY_FIELD_PATHS,
    flatten_json,
    ground_truth_from_case,
    sha256_file,
)


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def indexed_cases(path: Path) -> dict[str, dict[str, str]]:
    return {case["case_id"]: case for case in load_csv(path)}


def semantic_fingerprint(case: dict[str, str]) -> tuple[str, ...]:
    fields = (
        "specimen_site",
        "laterality",
        "diagnosis",
        "breslow_thickness_mm",
        "breslow_qualifier",
        "ulceration",
        "mitotic_rate_per_mm2",
        "mitotic_qualifier",
        "invasive_peripheral",
        "invasive_deep",
        "in_situ_peripheral",
        "in_situ_deep",
        "pT",
        "pN",
        "pM",
        "stage_group",
    )
    return tuple(case[field] for field in fields)


def test_v4_split_and_origins_are_explicit() -> None:
    cases = load_v4_cases()
    assert tuple(case["case_id"] for case in cases) == ALL_CASE_IDS
    assert DEVELOPMENT_CASE_IDS == ("MEL-104",)
    assert FORMAL_CASE_IDS == ("MEL-105", "MEL-106", "MEL-107", "MEL-108", "MEL-109")
    assert {case["case_id"] for case in cases if case["split"] == "development"} == set(
        DEVELOPMENT_CASE_IDS
    )
    assert {case["case_id"] for case in cases if case["split"] == "formal"} == set(FORMAL_CASE_IDS)
    assert set(DEVELOPMENT_CASE_IDS).isdisjoint(FORMAL_CASE_IDS)
    assert {case["case_id"]: case["case_origin"] for case in cases} == CASE_ORIGINS
    assert tuple(CASE_ORIGINS) == ALL_CASE_IDS


def test_v4_discloses_exact_v3_case_reuse_and_one_new_case() -> None:
    v3 = indexed_cases(ROOT / "data" / "v3" / "cases.csv")
    v4 = indexed_cases(ROOT / "data" / "v4" / "cases.csv")
    shared_fields = tuple(field for field in v3["MEL-104"] if field not in {"split", "case_origin"})

    assert tuple(REUSED_V3_CASE_IDS) == ("MEL-104", "MEL-105", "MEL-106", "MEL-107", "MEL-108")
    for case_id in REUSED_V3_CASE_IDS:
        assert {field: v4[case_id][field] for field in shared_fields} == {
            field: v3[case_id][field] for field in shared_fields
        }

    assert v3["MEL-104"]["split"] == "formal"
    assert v4["MEL-104"]["split"] == "development"
    assert all(v4[case_id]["split"] == "formal" for case_id in REUSED_V3_CASE_IDS[1:])
    assert "MEL-109" not in v3

    prior_cases = [
        *load_csv(ROOT / "data" / "cases.csv"),
        *load_csv(ROOT / "data" / "v3" / "cases.csv"),
    ]
    prior_fingerprints = {semantic_fingerprint(case) for case in prior_cases}
    assert semantic_fingerprint(v4["MEL-109"]) not in prior_fingerprints


def test_v4_ground_truth_is_schema_valid() -> None:
    schema = json.loads((ROOT / "schema" / "extraction.schema.json").read_text(encoding="utf-8"))
    validator = Draft202012Validator(schema)
    for case in load_v4_cases():
        for template_id in TEMPLATE_IDS:
            truth = ground_truth_from_case(case, template_id)
            assert list(validator.iter_errors(truth)) == []


def test_v4_formal_cases_cover_supported_values_and_abstention() -> None:
    formal = [case for case in load_v4_cases() if case["case_id"] in FORMAL_CASE_IDS]
    flattened = [flatten_json(ground_truth_from_case(case, "A")) for case in formal]

    for path in PRIMARY_FIELD_PATHS:
        values = [truth[path] for truth in flattened]
        assert any(value is not None for value in values), path
        if path not in {"specimen_site", "diagnosis"}:
            assert any(value is None for value in values), path

    assert {truth["laterality"] for truth in flattened} == {
        "left",
        "right",
        "midline",
        None,
    }
    assert {truth["diagnosis"] for truth in flattened} == {
        "invasive_melanoma",
        "melanoma_in_situ",
        "residual_melanoma_in_situ_no_invasive",
    }
    assert {truth["breslow_qualifier"] for truth in flattened} == {
        "exact",
        "at_least",
        "approximate",
        None,
    }
    assert {truth["ulceration"] for truth in flattened} == {
        "present",
        "not_identified",
        "cannot_be_assessed",
        None,
    }
    assert {truth["mitotic_qualifier"] for truth in flattened} == {
        "exact",
        "at_least",
        None,
    }
    unique_null_opportunities = sum(
        truth[path] is None for truth in flattened for path in PRIMARY_FIELD_PATHS
    )
    assert unique_null_opportunities == 31
    assert unique_null_opportunities * len(TEMPLATE_IDS) * len(CONDITIONS) == 124


def test_v4_manifest_is_complete_balanced_and_hash_bound() -> None:
    manifest = load_csv(ROOT / "data" / "v4" / "report_manifest.csv")
    assert len(manifest) == 24
    assert {row["dataset_version"] for row in manifest} == {DATASET_VERSION}
    assert {row["generator_version"] for row in manifest} == {GENERATOR_VERSION}
    assert Counter(row["split"] for row in manifest) == {
        "development": 4,
        "formal": 20,
    }

    cells: dict[str, set[tuple[str, str]]] = defaultdict(set)
    for row in manifest:
        case_id = row["semantic_case_id"]
        cells[case_id].add((row["template_id"], row["condition"]))
        assert row["case_origin"] == CASE_ORIGINS[case_id]
        assert row["artifact_origin"] == (
            REUSED_ARTIFACT_ORIGIN if case_id in REUSED_V3_CASE_IDS else NEW_ARTIFACT_ORIGIN
        )
        assert row["template_id"] in TEMPLATE_IDS
        assert row["condition"] in CONDITIONS
        pdf_path = ROOT / row["pdf_path"]
        image_path = ROOT / row["image_path"]
        ground_truth_path = ROOT / row["ground_truth_path"]
        source_text_path = ROOT / row["source_text_path"]
        assert sha256_file(pdf_path) == row["pdf_sha256"]
        assert sha256_file(image_path) == row["image_sha256"]
        assert ground_truth_path.is_file()
        assert source_text_path.is_file()
        assert "NOT A REAL PATIENT" in source_text_path.read_text(encoding="utf-8")

        reader = PdfReader(str(pdf_path))
        assert len(reader.pages) == 1
        extracted_text = (reader.pages[0].extract_text() or "").strip()
        if row["condition"] == "clean":
            assert "SYNTHETIC" in extracted_text
        else:
            assert extracted_text == ""

        with Image.open(image_path) as image:
            assert image.width >= 1200
            assert image.height >= 1600
            image.verify()

    expected_cells = {
        (template_id, condition) for template_id in TEMPLATE_IDS for condition in CONDITIONS
    }
    assert set(cells) == set(ALL_CASE_IDS)
    assert all(case_cells == expected_cells for case_cells in cells.values())


def test_v4_reused_artifacts_are_byte_identical_to_v3() -> None:
    v3_manifest = load_csv(ROOT / "data" / "v3" / "report_manifest.csv")
    v4_manifest = load_csv(ROOT / "data" / "v4" / "report_manifest.csv")
    v3_rows = {
        (row["document_id"], row["condition"]): row
        for row in v3_manifest
        if row["semantic_case_id"] in REUSED_V3_CASE_IDS
    }
    v4_rows = {
        (row["document_id"], row["condition"]): row
        for row in v4_manifest
        if row["semantic_case_id"] in REUSED_V3_CASE_IDS
    }
    assert set(v4_rows) == set(v3_rows)

    for key, v4_row in v4_rows.items():
        v3_row = v3_rows[key]
        for hash_field in ("pdf_sha256", "image_sha256"):
            assert v4_row[hash_field] == v3_row[hash_field], (key, hash_field)
        for path_field in (
            "pdf_path",
            "image_path",
            "source_text_path",
            "ground_truth_path",
        ):
            assert (ROOT / v4_row[path_field]).read_bytes() == (
                ROOT / v3_row[path_field]
            ).read_bytes()


def test_v4_degradation_is_deterministic_bounded_and_discloses_reuse() -> None:
    for doc_id in ("MEL-104-A", "MEL-105-B", "MEL-108-A"):
        params = degradation_parameters(doc_id)
        assert params == v3_degradation_parameters(doc_id)
        assert params["perturbation_id"] == "scan-v3"

    first = degradation_parameters("MEL-109-A")
    second = degradation_parameters("MEL-109-A")
    assert first == second
    assert first["perturbation_id"] == "scan-v4"
    assert 0.35 <= abs(first["rotation_degrees"]) <= 0.65
    assert 0.55 <= first["blur_radius"] <= 0.85
    assert 0.82 <= first["contrast_factor"] <= 0.90
    assert 58 <= first["jpeg_quality"] <= 72
