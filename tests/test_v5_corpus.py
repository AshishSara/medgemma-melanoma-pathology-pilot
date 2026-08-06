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

import freeze_v5_protocol as freezer  # noqa: E402
from generate_v5_reports import (  # noqa: E402
    ALL_CASE_IDS,
    CONDITIONS,
    DATASET_VERSION,
    DEVELOPMENT_CASE_IDS,
    FORMAL_CASE_IDS,
    GENERATOR_VERSION,
    TEMPLATE_IDS,
    degradation_parameters,
    load_v5_cases,
)
from pilot_utils import PRIMARY_FIELD_PATHS, flatten_json, sha256_file  # noqa: E402


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_v5_split_is_exactly_one_qualification_and_five_formal_vectors() -> None:
    cases = load_v5_cases()
    assert tuple(row["case_id"] for row in cases) == ALL_CASE_IDS
    assert DEVELOPMENT_CASE_IDS == ("MEL-190",)
    assert FORMAL_CASE_IDS == ("MEL-201", "MEL-202", "MEL-203", "MEL-204", "MEL-205")
    assert Counter(row["split"] for row in cases) == {"development": 1, "formal": 5}
    assert all(row["case_origin"] == "fresh-v5" for row in cases)


def test_v5_fingerprints_ids_and_dates_are_exactly_fresh() -> None:
    cases, freshness, reference_hashes = freezer.load_and_validate_cases(ROOT)
    prior = [
        row for relative_path in freezer.PRIOR_CASE_PATHS for row in load_csv(ROOT / relative_path)
    ]
    prior_fingerprints = {freezer.semantic_fingerprint(row) for row in prior}

    assert all(freezer.semantic_fingerprint(row) not in prior_fingerprints for row in cases)
    assert len({freezer.semantic_fingerprint(row) for row in cases}) == 6
    assert {row["case_id"] for row in cases}.isdisjoint(row["case_id"] for row in prior)
    assert {row["report_date"] for row in cases}.isdisjoint(row["report_date"] for row in prior)
    assert freshness["method"] == (
        "exact ordered 16-string tuple comparison; empty strings retained; "
        "no normalization or fuzzy matching"
    )
    assert freshness["fields"] == list(freezer.SEMANTIC_FINGERPRINT_FIELDS)
    assert freshness["prior_row_count"] == 22
    assert freshness["prior_unique_fingerprint_count"] == 17
    assert freshness["v5_unique_fingerprint_count"] == 6
    assert freshness["collision_count"] == 0
    assert set(reference_hashes) == set(freezer.PRIOR_CASE_PATHS)


def test_prelock_mel_202_collision_correction_is_machine_visible() -> None:
    v2_mel_004 = next(
        row for row in load_csv(ROOT / "data" / "cases.csv") if row["case_id"] == "MEL-004"
    )
    v5_mel_202 = next(
        row for row in load_csv(ROOT / "data" / "v5" / "cases.csv") if row["case_id"] == "MEL-202"
    )
    assert v5_mel_202["specimen_site"] == "abdomen"
    assert freezer.semantic_fingerprint(v5_mel_202) != freezer.semantic_fingerprint(v2_mel_004)

    disclosed_draft = dict(v5_mel_202)
    disclosed_draft["specimen_site"] = "shoulder"
    assert freezer.semantic_fingerprint(disclosed_draft) == freezer.semantic_fingerprint(v2_mel_004)


def test_v5_manifest_has_the_exact_ordered_4_plus_20_matrix() -> None:
    cases = load_v5_cases()
    manifest = freezer.load_and_validate_manifest(ROOT, cases)
    assert len(manifest) == 24
    assert Counter(row["split"] for row in manifest) == {
        "development": 4,
        "formal": 20,
    }
    assert {row["dataset_version"] for row in manifest} == {DATASET_VERSION}
    assert {row["generator_version"] for row in manifest} == {GENERATOR_VERSION}

    cells: dict[str, set[tuple[str, str]]] = defaultdict(set)
    for row in manifest:
        cells[row["semantic_case_id"]].add((row["template_id"], row["condition"]))
    expected_cells = {
        (template_id, condition) for template_id in TEMPLATE_IDS for condition in CONDITIONS
    }
    assert tuple(cells) == ALL_CASE_IDS
    assert all(observed == expected_cells for observed in cells.values())


def test_v5_ground_truth_is_deterministic_schema_valid_with_exact_denominators() -> None:
    cases = load_v5_cases()
    manifest = freezer.load_and_validate_manifest(ROOT, cases)
    counts = freezer._validate_ground_truth_and_denominators(ROOT, manifest, cases)
    assert counts["development"]["field_opportunities"] == 64
    assert counts["development"]["ground_truth_non_null_opportunities"] == 52
    assert counts["development"]["ground_truth_null_opportunities"] == 12
    assert counts["formal"]["field_opportunities"] == 320
    assert counts["formal"]["ground_truth_non_null_opportunities"] == 196
    assert counts["formal"]["ground_truth_null_opportunities"] == 124
    assert all(
        counts["formal"]["by_condition"][condition]["field_opportunities"] == 160
        for condition in CONDITIONS
    )

    schema = json.loads((ROOT / "schema" / "extraction.schema.json").read_text(encoding="utf-8"))
    validator = Draft202012Validator(schema)
    for path in sorted((ROOT / "data" / "v5" / "ground_truth").glob("*.json")):
        truth = json.loads(path.read_text(encoding="utf-8"))
        assert list(validator.iter_errors(truth)) == []
        flattened = flatten_json(truth)
        assert all(field in flattened for field in PRIMARY_FIELD_PATHS)


def test_v5_manifest_hashes_and_exact_corpus_file_inventory() -> None:
    cases = load_v5_cases()
    manifest = freezer.load_and_validate_manifest(ROOT, cases)
    operational, evaluation_only = freezer.expected_corpus_paths(manifest)
    expected_data = {path for path in evaluation_only if path.startswith("data/v5/")} | {
        freezer.V5_MANIFEST_RELATIVE_PATH,
        freezer.RUNTIME_PROFILE_RELATIVE_PATH,
    }
    expected_output = {path for path in operational if path.startswith("output/v5/")}

    actual_data = {
        path.relative_to(ROOT).as_posix()
        for path in (ROOT / "data" / "v5").rglob("*")
        if path.is_file() and path.name != "protocol_lock.json"
    }
    actual_output = {
        path.relative_to(ROOT).as_posix()
        for path in (ROOT / "output" / "v5").rglob("*")
        if path.is_file()
    }
    assert actual_data == expected_data
    assert actual_output == expected_output
    assert not any(path.is_symlink() for path in (ROOT / "data" / "v5").rglob("*"))
    assert not any(path.is_symlink() for path in (ROOT / "output" / "v5").rglob("*"))

    for row in manifest:
        assert sha256_file(ROOT / row["pdf_path"]) == row["pdf_sha256"]
        assert sha256_file(ROOT / row["image_path"]) == row["image_sha256"]


def test_v5_reports_are_single_page_and_degradation_is_deterministic() -> None:
    manifest = load_csv(ROOT / "data" / "v5" / "report_manifest.csv")
    for row in manifest:
        source = (ROOT / row["source_text_path"]).read_text(encoding="utf-8")
        assert "SYNTHETIC" in source
        assert "NOT A REAL PATIENT" in source

        reader = PdfReader(str(ROOT / row["pdf_path"]))
        assert len(reader.pages) == 1
        extracted = (reader.pages[0].extract_text() or "").strip()
        if row["condition"] == "clean":
            assert "SYNTHETIC" in extracted
        else:
            assert extracted == ""

        with Image.open(ROOT / row["image_path"]) as image:
            assert image.width >= 1200
            assert image.height >= 1600
            image.verify()

    for document_id in ("MEL-190-A", "MEL-202-B", "MEL-205-A"):
        first = degradation_parameters(document_id)
        assert first == degradation_parameters(document_id)
        assert first["perturbation_id"] == "scan-v5"
        assert 0.35 <= abs(first["rotation_degrees"]) <= 0.65
        assert 0.55 <= first["blur_radius"] <= 0.85
        assert 0.82 <= first["contrast_factor"] <= 0.90
        assert 58 <= first["jpeg_quality"] <= 72
