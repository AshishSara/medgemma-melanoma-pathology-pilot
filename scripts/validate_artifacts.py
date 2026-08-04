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
    MODEL_IDS,
    PRIMARY_FIELD_PATHS,
    ROOT,
    TEMPLATE_IDS,
    flatten_json,
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

    pilot_rows = load_csv(ROOT / "results" / "pilot_table.csv")
    review_rows = load_csv(ROOT / "results" / "manual_review.csv")
    intended = len(MODEL_IDS) * len(doc_conditions) * len(CONDITIONS)
    if len(pilot_rows) != intended:
        errors.append(f"Pilot table should have {intended} rows, found {len(pilot_rows)}")
    if len(review_rows) != intended:
        errors.append(f"Manual review ledger should have {intended} rows, found {len(review_rows)}")

    summary = {
        "valid": not errors,
        "errors": errors,
        "semantic_cases": len(case_templates),
        "base_reports": len(doc_conditions),
        "document_conditions": len(manifest),
        "intended_model_outputs": intended,
        "ground_truth_files": len(seen_truth),
    }
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        raise SystemExit(1)
    print(
        "Validated 8 semantic cases, 16 base reports, 32 document conditions, "
        "16 ground-truth files, and a 64-output review ledger."
    )
    return summary


if __name__ == "__main__":
    validate()
