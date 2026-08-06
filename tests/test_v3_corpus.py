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

from generate_v3_reports import (  # noqa: E402
    ALL_CASE_IDS,
    CONDITIONS,
    DATASET_VERSION,
    DEVELOPMENT_CASE_IDS,
    FORMAL_CASE_IDS,
    GENERATOR_VERSION,
    TEMPLATE_IDS,
    degradation_parameters,
    load_v3_cases,
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


def test_v3_split_is_explicit_fresh_and_disjoint() -> None:
    cases = load_v3_cases()
    assert tuple(case["case_id"] for case in cases) == ALL_CASE_IDS
    assert {case["case_id"] for case in cases if case["split"] == "development"} == set(
        DEVELOPMENT_CASE_IDS
    )
    assert {case["case_id"] for case in cases if case["split"] == "formal"} == set(FORMAL_CASE_IDS)
    assert set(DEVELOPMENT_CASE_IDS).isdisjoint(FORMAL_CASE_IDS)
    assert set(ALL_CASE_IDS).isdisjoint({f"MEL-{number:03d}" for number in range(1, 9)})


def test_no_v3_case_reuses_a_v2_semantic_fact_vector() -> None:
    legacy = load_csv(ROOT / "data" / "cases.csv")
    v3 = load_v3_cases()
    legacy_fingerprints = {semantic_fingerprint(case) for case in legacy}
    assert all(semantic_fingerprint(case) not in legacy_fingerprints for case in v3)


def test_v3_ground_truth_is_schema_valid() -> None:
    schema = json.loads((ROOT / "schema" / "extraction.schema.json").read_text(encoding="utf-8"))
    validator = Draft202012Validator(schema)
    for case in load_v3_cases():
        for template_id in TEMPLATE_IDS:
            truth = ground_truth_from_case(case, template_id)
            assert list(validator.iter_errors(truth)) == []


def test_formal_cases_cover_supported_values_and_abstention() -> None:
    formal = [case for case in load_v3_cases() if case["case_id"] in FORMAL_CASE_IDS]
    truths = [ground_truth_from_case(case, "A") for case in formal]
    flattened = [flatten_json(truth) for truth in truths]

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
    null_opportunities = sum(value is None for truth in flattened for value in truth.values())
    assert null_opportunities == 31


def test_v3_manifest_is_complete_balanced_and_hash_bound() -> None:
    manifest = load_csv(ROOT / "data" / "v3" / "report_manifest.csv")
    assert len(manifest) == 32
    assert {row["dataset_version"] for row in manifest} == {DATASET_VERSION}
    assert {row["generator_version"] for row in manifest} == {GENERATOR_VERSION}
    assert Counter(row["split"] for row in manifest) == {
        "development": 12,
        "formal": 20,
    }

    cells: dict[str, set[tuple[str, str]]] = defaultdict(set)
    for row in manifest:
        cells[row["semantic_case_id"]].add((row["template_id"], row["condition"]))
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


def test_v3_degradation_is_versioned_deterministic_and_bounded() -> None:
    first = degradation_parameters("MEL-104-A")
    second = degradation_parameters("MEL-104-A")
    assert first == second
    assert first["perturbation_id"] == "scan-v3"
    assert 0.35 <= abs(first["rotation_degrees"]) <= 0.65
    assert 0.55 <= first["blur_radius"] <= 0.85
    assert 0.82 <= first["contrast_factor"] <= 0.90
    assert 58 <= first["jpeg_quality"] <= 72
