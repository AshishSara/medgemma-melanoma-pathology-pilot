#!/usr/bin/env python3
from __future__ import annotations

import csv
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List

from jsonschema import Draft202012Validator
from PIL import Image
from pilot_utils import (
    CONDITIONS,
    DEVELOPMENT_CASE_IDS,
    FORMAL_BASE_REPORT_COUNT,
    FORMAL_DOCUMENT_CONDITION_COUNT,
    FORMAL_OUTPUT_COUNT,
    FORMAL_SEMANTIC_CASE_COUNT,
    MODEL_IDS,
    PRIMARY_FIELD_PATHS,
    PROTOCOL_VERSION,
    ROOT,
    TEMPLATE_IDS,
    flatten_json,
    formal_manifest_rows,
    read_json,
    sha256_file,
)
from pypdf import PdfReader


def load_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def validate() -> Dict[str, Any]:
    errors: List[str] = []
    manifest = load_csv(ROOT / "data" / "report_manifest.csv")
    if len(manifest) != 32:
        errors.append(f"Manifest should have 32 rows, found {len(manifest)}")

    doc_conditions: Dict[str, set] = defaultdict(set)
    case_templates: Dict[str, set] = defaultdict(set)
    condition_counts = Counter()
    template_condition_counts = Counter()
    schema = read_json(ROOT / "schema" / "extraction.schema.json")
    validator = Draft202012Validator(schema)
    seen_truth = set()
    truth_field_values = {path: set() for path in PRIMARY_FIELD_PATHS}

    for row in manifest:
        doc_id = row["document_id"]
        condition = row["condition"]
        doc_conditions[doc_id].add(condition)
        case_templates[row["semantic_case_id"]].add(row["template_id"])
        condition_counts[condition] += 1
        template_condition_counts[(row["template_id"], condition)] += 1

        for key in ("pdf_path", "image_path", "ground_truth_path", "source_text_path"):
            path = ROOT / row[key]
            if not path.exists():
                errors.append(f"Missing {key}: {row[key]}")

        pdf_path = ROOT / row["pdf_path"]
        image_path = ROOT / row["image_path"]
        if pdf_path.exists() and sha256_file(pdf_path) != row["pdf_sha256"]:
            errors.append(f"PDF hash mismatch: {row['pdf_path']}")
        if image_path.exists() and sha256_file(image_path) != row["image_sha256"]:
            errors.append(f"Image hash mismatch: {row['image_path']}")

        if pdf_path.exists():
            reader = PdfReader(str(pdf_path))
            if len(reader.pages) != 1:
                errors.append(f"Expected one-page PDF: {row['pdf_path']}")
            text = (reader.pages[0].extract_text() or "").strip()
            if condition == "clean" and "SYNTHETIC" not in text:
                errors.append(f"Clean PDF lacks expected text layer: {row['pdf_path']}")
            if condition == "ocr_degraded" and text:
                errors.append(f"Degraded PDF unexpectedly has a text layer: {row['pdf_path']}")

        if image_path.exists():
            with Image.open(image_path) as image:
                if image.width < 1200 or image.height < 1600:
                    errors.append(f"Rendered image unexpectedly small: {row['image_path']}")
                image.verify()

        truth_path = ROOT / row["ground_truth_path"]
        if truth_path.exists() and truth_path not in seen_truth:
            truth = read_json(truth_path)
            for error in validator.iter_errors(truth):
                errors.append(f"Ground truth schema error in {truth_path.name}: {error.message}")
            flattened = flatten_json(truth)
            for path in PRIMARY_FIELD_PATHS:
                truth_field_values[path].add(repr(flattened.get(path)))
            seen_truth.add(truth_path)

        source_path = ROOT / row["source_text_path"]
        if source_path.exists():
            source = source_path.read_text(encoding="utf-8")
            if "NOT A REAL PATIENT" not in source:
                errors.append(f"Synthetic banner absent from source text: {source_path.name}")

    if len(doc_conditions) != 16:
        errors.append(f"Expected 16 base document IDs, found {len(doc_conditions)}")
    for doc_id, conditions in doc_conditions.items():
        if conditions != set(CONDITIONS):
            errors.append(f"{doc_id} conditions are {sorted(conditions)}")
    if len(case_templates) != 8:
        errors.append(f"Expected 8 semantic cases, found {len(case_templates)}")
    for case_id, templates in case_templates.items():
        if templates != set(TEMPLATE_IDS):
            errors.append(f"{case_id} templates are {sorted(templates)}")
    if condition_counts != Counter({"clean": 16, "ocr_degraded": 16}):
        errors.append(f"Unexpected condition balance: {dict(condition_counts)}")
    expected_cell_counts = {
        ("A", "clean"): 8,
        ("A", "ocr_degraded"): 8,
        ("B", "clean"): 8,
        ("B", "ocr_degraded"): 8,
    }
    if dict(template_condition_counts) != expected_cell_counts:
        errors.append(f"Unexpected template-condition balance: {dict(template_condition_counts)}")
    if len(seen_truth) != 16:
        errors.append(f"Expected 16 ground-truth files, found {len(seen_truth)}")

    required_missingness = (
        "laterality",
        "breslow_thickness_mm",
        "ulceration",
        "mitotic_rate_per_mm2",
        "margins.invasive_peripheral",
        "margins.in_situ_peripheral",
        "staging.pT",
        "staging.pN",
        "staging.pM",
        "staging.stage_group",
    )
    for path in required_missingness:
        if "None" not in truth_field_values[path]:
            errors.append(f"Expected at least one null ground truth for {path}")

    formal_manifest = formal_manifest_rows(manifest)
    formal_case_ids = {row["semantic_case_id"] for row in formal_manifest}
    formal_document_ids = {row["document_id"] for row in formal_manifest}
    if len(formal_case_ids) != FORMAL_SEMANTIC_CASE_COUNT:
        errors.append(
            f"Expected {FORMAL_SEMANTIC_CASE_COUNT} formal cases, found {len(formal_case_ids)}"
        )
    if formal_case_ids.intersection(DEVELOPMENT_CASE_IDS):
        errors.append("Development cases leaked into the formal manifest subset")
    if len(formal_document_ids) != FORMAL_BASE_REPORT_COUNT:
        errors.append(
            f"Expected {FORMAL_BASE_REPORT_COUNT} formal base reports, "
            f"found {len(formal_document_ids)}"
        )
    if len(formal_manifest) != FORMAL_DOCUMENT_CONDITION_COUNT:
        errors.append(
            f"Expected {FORMAL_DOCUMENT_CONDITION_COUNT} formal document conditions, "
            f"found {len(formal_manifest)}"
        )

    pilot_rows = load_csv(ROOT / "results" / "pilot_table.csv")
    review_rows = load_csv(ROOT / "results" / "manual_review.csv")
    intended = FORMAL_OUTPUT_COUNT
    if len(pilot_rows) != FORMAL_OUTPUT_COUNT:
        errors.append(
            f"Pilot table should have {FORMAL_OUTPUT_COUNT} rows, found {len(pilot_rows)}"
        )
    if len(review_rows) != FORMAL_OUTPUT_COUNT:
        errors.append(
            f"Manual review ledger should have {FORMAL_OUTPUT_COUNT} rows, found {len(review_rows)}"
        )

    expected_keys = {
        (model_id, row["document_id"], row["condition"])
        for model_id in MODEL_IDS
        for row in formal_manifest
    }
    for label, rows in (("pilot table", pilot_rows), ("manual review ledger", review_rows)):
        keys = [(row["model_id"], row["document_id"], row["condition"]) for row in rows]
        if len(keys) != len(set(keys)):
            errors.append(f"{label} contains duplicate model/document/condition keys")
        if set(keys) != expected_keys:
            errors.append(f"{label} keys do not exactly match the formal evaluation matrix")
        if any(row["semantic_case_id"] in DEVELOPMENT_CASE_IDS for row in rows):
            errors.append(f"{label} contains a development case")

    prompt_sha = sha256_file(ROOT / "prompts" / "extraction_prompt.txt")
    schema_sha = sha256_file(ROOT / "schema" / "extraction.schema.json")
    run_manifest = read_json(ROOT / "results" / "run_manifest.json")
    metrics = read_json(ROOT / "results" / "metrics.json")
    expected_manifest_values = {
        "protocol_version": PROTOCOL_VERSION,
        "formal_semantic_case_count": FORMAL_SEMANTIC_CASE_COUNT,
        "formal_base_report_count": FORMAL_BASE_REPORT_COUNT,
        "formal_document_condition_count": FORMAL_DOCUMENT_CONDITION_COUNT,
        "intended_output_count": FORMAL_OUTPUT_COUNT,
        "prompt_sha256": prompt_sha,
        "schema_sha256": schema_sha,
    }
    for key, expected in expected_manifest_values.items():
        if run_manifest.get(key) != expected:
            errors.append(
                f"Run manifest {key} should be {expected!r}, found {run_manifest.get(key)!r}"
            )
    if metrics.get("protocol_version") != PROTOCOL_VERSION:
        errors.append("Metrics protocol version is stale")
    if metrics.get("intended_output_count") != FORMAL_OUTPUT_COUNT:
        errors.append("Metrics intended output count is stale")

    completed_rows = [row for row in pilot_rows if row["run_status"] == "COMPLETED"]
    if run_manifest.get("completed_output_count") != len(completed_rows):
        errors.append("Run manifest completed count does not match the pilot table")
    if metrics.get("completed_output_count") != len(completed_rows):
        errors.append("Metrics completed count does not match the pilot table")
    review_by_key = {
        (row["model_id"], row["document_id"], row["condition"]): row for row in review_rows
    }
    for key, row in review_by_key.items():
        if row["review_status"] != "verified":
            continue
        model_id, document_id_value, condition = key
        raw_path = (
            ROOT
            / "results"
            / "raw"
            / model_id.replace("/", "__")
            / condition
            / f"{document_id_value}.txt"
        )
        if not raw_path.exists():
            errors.append(f"Verified review lacks raw output: {key}")
        elif row.get("raw_output_sha256") != sha256_file(raw_path):
            errors.append(f"Verified review hash does not match raw output: {key}")

    summary = {
        "valid": not errors,
        "errors": errors,
        "semantic_cases": len(case_templates),
        "base_reports": len(doc_conditions),
        "document_conditions": len(manifest),
        "formal_semantic_cases": len(formal_case_ids),
        "formal_base_reports": len(formal_document_ids),
        "formal_document_conditions": len(formal_manifest),
        "intended_model_outputs": intended,
        "ground_truth_files": len(seen_truth),
    }
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        raise SystemExit(1)
    print(
        "Validated the 8-case/16-report corpus and the held-out "
        "7-case/14-report/28-condition/56-output evaluation matrix."
    )
    return summary


if __name__ == "__main__":
    validate()
