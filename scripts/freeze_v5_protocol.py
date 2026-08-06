#!/usr/bin/env python3
"""Validate and freeze the prospective v5 protocol before any model inference."""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import os
import re
import subprocess
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path, PurePosixPath
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from jsonschema import Draft202012Validator
from pilot_utils import (
    PRIMARY_FIELD_PATHS,
    ROOT,
    flatten_json,
    ground_truth_from_case,
    sha256_bytes,
    sha256_file,
)

PROTOCOL_VERSION = "fresh-confirmatory-v5"
PROTOCOL_IDENTIFIER = "fresh-heldout-confirmatory-pilot-v5"
MODEL_ID = "google/medgemma-1.5-4b-it"
MODEL_REVISION = "91850547d9f0b2fdd21aa7c5f4f3d1a8a52c243b"
DTYPE_NAME = "bfloat16"
RANDOM_SEED = 0

DEVELOPMENT_OUTPUT_COUNT = 4
FORMAL_OUTPUT_COUNT = 20
EXPECTED_MANIFEST_OUTPUT_COUNT = DEVELOPMENT_OUTPUT_COUNT + FORMAL_OUTPUT_COUNT
DEVELOPMENT_CASE_IDS = ("MEL-190",)
FORMAL_CASE_IDS = ("MEL-201", "MEL-202", "MEL-203", "MEL-204", "MEL-205")
ALL_CASE_IDS = (*DEVELOPMENT_CASE_IDS, *FORMAL_CASE_IDS)
TEMPLATE_IDS = ("A", "B")
CONDITIONS = ("clean", "ocr_degraded")

CANDIDATE_MAX_NEW_TOKENS = 512
AUDIT_MAX_NEW_TOKENS = 1280
CANDIDATE_SERIALIZER_FORCE_JSON_FIELD_ORDER = False
CANDIDATE_SERIALIZER_MAX_CONSECUTIVE_WHITESPACES = 12
AUDIT_SERIALIZER_FORCE_JSON_FIELD_ORDER = True
AUDIT_SERIALIZER_MAX_CONSECUTIVE_WHITESPACES = 0
TESSERACT_LANGUAGE = "eng"
TESSERACT_CONFIG = "--oem 1 --psm 6"
TESSERACT_TIMEOUT_SECONDS = 120

PROTOCOL_LOCK_RELATIVE_PATH = "data/v5/protocol_lock.json"
V5_MANIFEST_RELATIVE_PATH = "data/v5/report_manifest.csv"
RUNTIME_PROFILE_RELATIVE_PATH = "data/v5/runtime_profile.json"
REQUIREMENTS_RELATIVE_PATH = "requirements/colab-v5.txt"

# The ordered tuple is the prospective semantic identity required by the protocol.
# Values are compared exactly as CSV strings; empty strings remain empty strings.
SEMANTIC_FINGERPRINT_FIELDS = (
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

PRIOR_CASE_PATHS = (
    "data/cases.csv",
    "data/v3/cases.csv",
    "data/v4/cases.csv",
)

CASES_FIELDNAMES = (
    "case_id",
    "split",
    "case_origin",
    "report_date",
    "procedure",
    "specimen_site",
    "laterality",
    "diagnosis",
    "histologic_type",
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
    "additional_findings",
    "historical_context",
    "ambiguity_note",
)

MANIFEST_FIELDNAMES = (
    "dataset_version",
    "generator_version",
    "split",
    "case_origin",
    "semantic_case_id",
    "document_id",
    "template_id",
    "layout_scope",
    "condition",
    "perturbation_id",
    "dpi",
    "renderer",
    "rotation_degrees",
    "blur_radius",
    "contrast_factor",
    "jpeg_quality",
    "source_row_sha256",
    "source_text_path",
    "ground_truth_path",
    "pdf_path",
    "image_path",
    "pdf_sha256",
    "image_sha256",
)

REQUIRED_LOCKED_SOURCE_PATHS = (
    V5_MANIFEST_RELATIVE_PATH,
    RUNTIME_PROFILE_RELATIVE_PATH,
    "docs/protocol_v5.md",
    "prompts/v3/audit_prompt.txt",
    "prompts/v3/candidate_prompt.txt",
    "pyproject.toml",
    REQUIREMENTS_RELATIVE_PATH,
    "schema/extraction.schema.json",
    "schema/v3/audit.schema.json",
    "schema/v3/candidate.schema.json",
    "scripts/evaluate_v5.py",
    "scripts/freeze_v5_protocol.py",
    "scripts/generate_reports.py",
    "scripts/generate_v5_reports.py",
    "scripts/pilot_utils.py",
    "scripts/run_v5_inference.py",
    "scripts/v3_pipeline.py",
    "tests/test_v5_corpus.py",
    "tests/test_v5_evaluator.py",
    "tests/test_v5_inference.py",
    "tests/test_v5_protocol.py",
    "uv.lock",
)

EXPECTED_RUNTIME_PROFILE = {
    "cuda_device_count": 1,
    "cuda_device_name": "Tesla T4",
    "cuda_version": "12.8",
    "gpu_memory_mib": 14912,
    "packages": {
        "accelerate": "1.14.0",
        "lm-format-enforcer": "0.11.3",
        "pytesseract": "0.3.13",
        "transformers": "4.57.6",
    },
    "platform": "Linux-6.6.122+-x86_64-with-glibc2.35",
    "profile_captured_before_model_inference": True,
    "python_version": "3.12.13",
    "tesseract_version": "tesseract 4.1.1",
    "torch_version": "2.11.0+cu128",
}

EXPECTED_REQUIREMENTS = (
    "accelerate==1.14.0",
    "lm-format-enforcer==0.11.3",
    "pytesseract==0.3.13",
    "transformers==4.57.6",
)

GATE_THRESHOLDS = {
    "development": {
        "assigned_input_count": 4,
        "field_opportunities": 64,
        "condition_field_opportunities": 32,
        "ground_truth_non_null_opportunities": 52,
        "ground_truth_null_opportunities": 12,
        "pooled_exact_minimum": 58,
        "per_condition_exact_minimum": 28,
        "non_null_recall_minimum": 45,
        "unsupported_maximum": 0,
        "accepted_non_null_evidence_rate_minimum": 1.0,
        "history_carryover_maximum": 0,
        "candidate_cap_hits_maximum": 0,
        "audit_cap_hits_maximum": 0,
    },
    "formal": {
        "assigned_input_count": 20,
        "field_opportunities": 320,
        "condition_field_opportunities": 160,
        "ground_truth_non_null_opportunities": 196,
        "ground_truth_null_opportunities": 124,
        "pooled_exact_minimum": 288,
        "per_condition_exact_minimum": 136,
        "non_null_recall_minimum": 167,
        "unsupported_maximum": 2,
        "accepted_non_null_evidence_rate_minimum": 1.0,
        "history_carryover_maximum": 0,
        "candidate_cap_hits_maximum": 0,
        "audit_cap_hits_maximum": 0,
    },
}

FIXED_CONFIGURATION = {
    "model_id": MODEL_ID,
    "model_revision": MODEL_REVISION,
    "dtype": DTYPE_NAME,
    "random_seed": RANDOM_SEED,
    "candidate_max_new_tokens": CANDIDATE_MAX_NEW_TOKENS,
    "audit_max_new_tokens": AUDIT_MAX_NEW_TOKENS,
    "candidate_serializer_force_json_field_order": (CANDIDATE_SERIALIZER_FORCE_JSON_FIELD_ORDER),
    "candidate_serializer_max_consecutive_whitespaces": (
        CANDIDATE_SERIALIZER_MAX_CONSECUTIVE_WHITESPACES
    ),
    "audit_serializer_force_json_field_order": AUDIT_SERIALIZER_FORCE_JSON_FIELD_ORDER,
    "audit_serializer_max_consecutive_whitespaces": (AUDIT_SERIALIZER_MAX_CONSECUTIVE_WHITESPACES),
    "tesseract_language": TESSERACT_LANGUAGE,
    "tesseract_config": TESSERACT_CONFIG,
    "tesseract_timeout_seconds": TESSERACT_TIMEOUT_SECONDS,
    "image_preprocessing": "none",
    "assistant_prefill": False,
    "do_sample": False,
    "num_beams": 1,
    "candidate_calls_per_input": 1,
    "audit_calls_per_input": 1,
    "development_output_count": DEVELOPMENT_OUTPUT_COUNT,
    "formal_output_count": FORMAL_OUTPUT_COUNT,
    "manifest_order": "exact CSV row order; candidate then blind audit per input",
    "resume_policy": (
        "reuse valid persisted responses; exactly one unchanged infrastructure retry "
        "only after call-start without a durable response"
    ),
}


class ProtocolFreezeError(RuntimeError):
    """Raised when any prospective v5 lock invariant is not satisfied."""


def _fail(message: str) -> None:
    raise ProtocolFreezeError(message)


def _relative(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError as exc:
        raise ProtocolFreezeError(f"Path escapes repository root: {path}") from exc


def _safe_repo_path(root: Path, raw_path: str) -> Path:
    if not isinstance(raw_path, str) or not raw_path:
        _fail("Lock-bound artifact path is missing")
    pure = PurePosixPath(raw_path)
    if pure.is_absolute() or ".." in pure.parts or str(pure) != raw_path:
        _fail(f"Path is not canonical and repository-relative: {raw_path!r}")
    current = root
    for part in pure.parts:
        current = current / part
        if current.is_symlink():
            _fail(f"Lock-bound artifacts and their parents may not be symlinks: {raw_path}")
    return current


def _reject_duplicate_keys(pairs: Sequence[Tuple[str, Any]]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            _fail(f"JSON contains duplicate key: {key}")
        result[key] = value
    return result


def _reject_non_finite(value: str) -> None:
    _fail(f"JSON contains non-finite number: {value}")


def _read_json(path: Path, label: str) -> Dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_non_finite,
        )
    except ProtocolFreezeError:
        raise
    except (OSError, UnicodeError, ValueError) as exc:
        raise ProtocolFreezeError(f"{label} is missing or invalid JSON: {path}") from exc
    if not isinstance(value, dict):
        _fail(f"{label} must be a JSON object: {path}")
    return value


def _load_csv(
    path: Path,
    *,
    label: str,
    exact_fieldnames: Sequence[str] | None = None,
) -> List[Dict[str, str]]:
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if exact_fieldnames is not None and reader.fieldnames != list(exact_fieldnames):
                _fail(
                    f"{label} columns differ from the fixed schema; "
                    f"expected={list(exact_fieldnames)}, found={reader.fieldnames}"
                )
            rows = list(reader)
    except ProtocolFreezeError:
        raise
    except (OSError, UnicodeError, csv.Error) as exc:
        raise ProtocolFreezeError(f"Cannot read {label}: {path}") from exc
    if any(None in row for row in rows):
        _fail(f"{label} contains malformed rows or extra columns")
    return rows


def semantic_fingerprint(row: Mapping[str, str]) -> Tuple[str, ...]:
    try:
        return tuple(row[field] for field in SEMANTIC_FINGERPRINT_FIELDS)
    except KeyError as exc:
        raise ProtocolFreezeError(f"Case row lacks semantic field: {exc.args[0]}") from exc


def _fingerprint_digest(fingerprint: Sequence[str]) -> str:
    encoded = json.dumps(list(fingerprint), ensure_ascii=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return sha256_bytes(encoded)


def load_and_validate_cases(
    root: Path,
) -> Tuple[List[Dict[str, str]], Dict[str, Any], Dict[str, str]]:
    cases_path = root / "data" / "v5" / "cases.csv"
    cases = _load_csv(
        cases_path,
        label="v5 cases",
        exact_fieldnames=CASES_FIELDNAMES,
    )
    if len(cases) != len(ALL_CASE_IDS):
        _fail(f"V5 must contain exactly {len(ALL_CASE_IDS)} semantic fact vectors")
    if tuple(row["case_id"] for row in cases) != ALL_CASE_IDS:
        _fail(f"V5 case order and IDs must be exactly {ALL_CASE_IDS!r}")
    expected_splits = {
        **{case_id: "development" for case_id in DEVELOPMENT_CASE_IDS},
        **{case_id: "formal" for case_id in FORMAL_CASE_IDS},
    }
    if {row["case_id"]: row["split"] for row in cases} != expected_splits:
        _fail("V5 case split assignments differ from the fixed 1-development/5-formal matrix")
    if any(row["case_origin"] != "fresh-v5" for row in cases):
        _fail("Every v5 case must declare case_origin=fresh-v5")

    v5_fingerprints = [semantic_fingerprint(row) for row in cases]
    if len(set(v5_fingerprints)) != len(v5_fingerprints):
        _fail("V5 contains duplicate exact 16-string semantic fingerprints")
    v5_ids = {row["case_id"] for row in cases}
    v5_dates = [row["report_date"] for row in cases]
    if len(set(v5_dates)) != len(v5_dates) or any(not value for value in v5_dates):
        _fail("V5 report dates must be non-empty and unique")

    prior_rows: List[Dict[str, str]] = []
    prior_reference_hashes: Dict[str, str] = {}
    for relative_path in PRIOR_CASE_PATHS:
        path = _safe_repo_path(root, relative_path)
        rows = _load_csv(path, label=f"prior cases {relative_path}")
        first_row = rows[0] if rows else {}
        missing = [field for field in SEMANTIC_FINGERPRINT_FIELDS if field not in first_row]
        if missing:
            _fail(f"Prior cases {relative_path} lack semantic fields: {missing}")
        if any("case_id" not in row or "report_date" not in row for row in rows):
            _fail(f"Prior cases {relative_path} lack case_id or report_date")
        prior_rows.extend(rows)
        prior_reference_hashes[relative_path] = sha256_file(path)

    prior_fingerprints = {semantic_fingerprint(row) for row in prior_rows}
    collisions = [
        row["case_id"]
        for row, fingerprint in zip(cases, v5_fingerprints)
        if fingerprint in prior_fingerprints
    ]
    if collisions:
        _fail(
            "V5 semantic fingerprints must be exactly absent from the v2-v4 union; "
            f"colliding v5 case IDs={collisions}"
        )
    prior_ids = {row["case_id"] for row in prior_rows}
    duplicate_ids = sorted(v5_ids & prior_ids)
    if duplicate_ids:
        _fail(f"V5 case IDs are not fresh versus v2-v4: {duplicate_ids}")
    prior_dates = {row["report_date"] for row in prior_rows}
    duplicate_dates = sorted(set(v5_dates) & prior_dates)
    if duplicate_dates:
        _fail(f"V5 report dates are not fresh versus v2-v4: {duplicate_dates}")

    freshness = {
        "method": (
            "exact ordered 16-string tuple comparison; empty strings retained; "
            "no normalization or fuzzy matching"
        ),
        "fields": list(SEMANTIC_FINGERPRINT_FIELDS),
        "prior_row_count": len(prior_rows),
        "prior_unique_fingerprint_count": len(prior_fingerprints),
        "v5_row_count": len(cases),
        "v5_unique_fingerprint_count": len(set(v5_fingerprints)),
        "collision_count": 0,
        "case_id_collision_count": 0,
        "report_date_collision_count": 0,
        "v5_fingerprint_sha256": {
            row["case_id"]: _fingerprint_digest(fingerprint)
            for row, fingerprint in zip(cases, v5_fingerprints)
        },
    }
    return cases, freshness, dict(sorted(prior_reference_hashes.items()))


def _case_row_sha256(row: Mapping[str, str]) -> str:
    canonical = json.dumps(dict(row), sort_keys=True, separators=(",", ":"))
    return sha256(canonical.encode("utf-8")).hexdigest()


def degradation_parameters(document_id: str) -> Dict[str, str]:
    digest = sha256(f"{document_id}|degradation-v5".encode("utf-8")).digest()
    sign = -1 if digest[0] % 2 else 1
    return {
        "perturbation_id": "scan-v5",
        "rotation_degrees": str(round(sign * (0.35 + (digest[1] / 255) * 0.30), 3)),
        "blur_radius": str(round(0.55 + (digest[2] / 255) * 0.30, 3)),
        "contrast_factor": str(round(0.82 + (digest[3] / 255) * 0.08, 3)),
        "jpeg_quality": str(58 + digest[4] % 15),
    }


def load_and_validate_manifest(
    root: Path,
    cases: Sequence[Mapping[str, str]],
) -> List[Dict[str, str]]:
    rows = _load_csv(
        root / V5_MANIFEST_RELATIVE_PATH,
        label="v5 report manifest",
        exact_fieldnames=MANIFEST_FIELDNAMES,
    )
    if len(rows) != EXPECTED_MANIFEST_OUTPUT_COUNT:
        _fail(f"Expected {EXPECTED_MANIFEST_OUTPUT_COUNT} v5 manifest rows, found {len(rows)}")
    expected_order = [
        (split, case_id, f"{case_id}-{template_id}", template_id, condition)
        for split, case_ids in (
            ("development", DEVELOPMENT_CASE_IDS),
            ("formal", FORMAL_CASE_IDS),
        )
        for case_id in case_ids
        for template_id in TEMPLATE_IDS
        for condition in CONDITIONS
    ]
    observed_order = [
        (
            row.get("split"),
            row.get("semantic_case_id"),
            row.get("document_id"),
            row.get("template_id"),
            row.get("condition"),
        )
        for row in rows
    ]
    if observed_order != expected_order:
        _fail("V5 manifest order or exact case/template/condition matrix is invalid")
    keys = [(row["document_id"], row["condition"]) for row in rows]
    if len(set(keys)) != len(keys):
        _fail("V5 manifest contains duplicate document/condition keys")

    cases_by_id = {row["case_id"]: row for row in cases}
    for row in rows:
        case_id = row["semantic_case_id"]
        document_id = row["document_id"]
        template_id = row["template_id"]
        condition = row["condition"]
        case = cases_by_id[case_id]
        fixed_values = {
            "dataset_version": "pilot-v5",
            "generator_version": "5.0.0",
            "case_origin": "fresh-v5",
            "layout_scope": "known_fixed_layout_family",
            "dpi": "160",
            "renderer": "pypdfium2-5.12.1",
            "source_row_sha256": _case_row_sha256(case),
        }
        for key, expected in fixed_values.items():
            if row.get(key) != expected:
                _fail(f"Manifest {key} is invalid for {document_id} {condition}")
        expected_paths = {
            "source_text_path": f"data/v5/source_text/{document_id}.txt",
            "ground_truth_path": f"data/v5/ground_truth/{document_id}.json",
            "pdf_path": f"output/v5/pdf/{condition}/{document_id}.pdf",
            "image_path": f"output/v5/rendered/{condition}/{document_id}.png",
        }
        for key, expected in expected_paths.items():
            if row.get(key) != expected:
                _fail(f"Manifest {key} is not canonical for {document_id} {condition}")
        if document_id != f"{case_id}-{template_id}":
            _fail(f"Manifest document ID is not deterministic: {document_id}")

        if condition == "clean":
            expected_perturbation = {
                "perturbation_id": "none",
                "rotation_degrees": "0",
                "blur_radius": "0",
                "contrast_factor": "1",
                "jpeg_quality": "",
            }
        else:
            expected_perturbation = degradation_parameters(document_id)
        for key, expected in expected_perturbation.items():
            if row.get(key) != expected:
                _fail(
                    f"Manifest deterministic perturbation {key} is invalid "
                    f"for {document_id} {condition}"
                )
        for key in ("pdf_sha256", "image_sha256"):
            if re.fullmatch(r"[0-9a-f]{64}", row.get(key, "")) is None:
                _fail(f"Manifest {key} is not a lowercase SHA-256 for {document_id} {condition}")
    return rows


def expected_corpus_paths(
    rows: Sequence[Mapping[str, str]],
) -> Tuple[set[str], set[str]]:
    operational = {V5_MANIFEST_RELATIVE_PATH}
    evaluation_only = {"data/v5/cases.csv"}
    for row in rows:
        operational.update((row["pdf_path"], row["image_path"]))
        evaluation_only.update((row["source_text_path"], row["ground_truth_path"]))
    if operational & evaluation_only:
        _fail("Operational and evaluation-only corpus paths overlap")
    return operational, evaluation_only


def _expected_directories(base_relative: str, expected_files: Sequence[str]) -> set[str]:
    expected: set[str] = set()
    base = PurePosixPath(base_relative)
    for raw_path in expected_files:
        parent = PurePosixPath(raw_path).parent
        while parent != base and base in parent.parents:
            expected.add(str(parent))
            parent = parent.parent
    return expected


def validate_exact_namespace(
    root: Path,
    base_relative: str,
    expected_files: Sequence[str],
) -> None:
    base = _safe_repo_path(root, base_relative)
    if not base.is_dir():
        _fail(f"Missing v5 namespace directory: {base_relative}")
    discovered_files: set[str] = set()
    discovered_directories: set[str] = set()
    for path in base.rglob("*"):
        relative = _relative(path, root)
        if path.is_symlink():
            _fail(f"V5 namespace contains a symlink: {relative}")
        if path.is_file():
            discovered_files.add(relative)
        elif path.is_dir():
            discovered_directories.add(relative)
        else:
            _fail(f"V5 namespace contains an unsupported filesystem object: {relative}")
    expected_file_set = set(expected_files)
    expected_directory_set = _expected_directories(base_relative, expected_files)
    if discovered_files != expected_file_set:
        _fail(
            f"{base_relative} file inventory is not exact; "
            f"missing={sorted(expected_file_set - discovered_files)}, "
            f"extra={sorted(discovered_files - expected_file_set)}"
        )
    if discovered_directories != expected_directory_set:
        _fail(
            f"{base_relative} directory inventory is not exact; "
            f"missing={sorted(expected_directory_set - discovered_directories)}, "
            f"extra={sorted(discovered_directories - expected_directory_set)}"
        )


def _validate_pre_inference_namespace(root: Path) -> None:
    results_root = root / "results" / "v5"
    if results_root.is_symlink():
        _fail("results/v5 may not be a symlink")
    if not results_root.exists():
        return
    if not results_root.is_dir():
        _fail("results/v5 is not a directory")
    contents = list(results_root.rglob("*"))
    if contents:
        _fail("V5 inference/result namespace must be pristine before protocol lock")


def _validate_ground_truth_and_denominators(
    root: Path,
    rows: Sequence[Mapping[str, str]],
    cases: Sequence[Mapping[str, str]],
) -> Dict[str, Any]:
    schema = _read_json(root / "schema" / "extraction.schema.json", "canonical schema")
    try:
        Draft202012Validator.check_schema(schema)
    except Exception as exc:
        raise ProtocolFreezeError("Canonical extraction schema is invalid") from exc
    validator = Draft202012Validator(schema)
    cases_by_id = {row["case_id"]: row for row in cases}
    expected_primary_paths = tuple(SEMANTIC_FINGERPRINT_FIELDS[:8]) + (
        "margins.invasive_peripheral",
        "margins.invasive_deep",
        "margins.in_situ_peripheral",
        "margins.in_situ_deep",
        "staging.pT",
        "staging.pN",
        "staging.pM",
        "staging.stage_group",
    )
    if tuple(PRIMARY_FIELD_PATHS) != expected_primary_paths:
        _fail("The locked 16-field scoring path inventory has drifted")

    truth_cache: Dict[str, Dict[str, Any]] = {}
    counts = {
        split: {
            "assigned_input_count": 0,
            "field_opportunities": 0,
            "ground_truth_non_null_opportunities": 0,
            "ground_truth_null_opportunities": 0,
            "by_condition": {
                condition: {
                    "assigned_input_count": 0,
                    "field_opportunities": 0,
                    "ground_truth_non_null_opportunities": 0,
                    "ground_truth_null_opportunities": 0,
                }
                for condition in CONDITIONS
            },
        }
        for split in ("development", "formal")
    }
    for row in rows:
        relative_path = row["ground_truth_path"]
        if relative_path not in truth_cache:
            path = _safe_repo_path(root, relative_path)
            truth = _read_json(path, "v5 ground truth")
            errors = sorted(
                validator.iter_errors(truth),
                key=lambda error: tuple(str(part) for part in error.absolute_path),
            )
            if errors:
                detail = "; ".join(error.message for error in errors[:3])
                _fail(f"Ground truth violates canonical schema: {relative_path}: {detail}")
            expected_truth = ground_truth_from_case(
                cases_by_id[row["semantic_case_id"]],
                row["template_id"],
            )
            if truth != expected_truth:
                _fail(f"Ground truth differs from deterministic case-row truth: {relative_path}")
            truth_cache[relative_path] = truth
        truth = truth_cache[relative_path]
        flattened = flatten_json(truth)
        if any(path not in flattened for path in PRIMARY_FIELD_PATHS):
            _fail(f"Ground truth lacks one or more fixed clinical fields: {relative_path}")
        values = [flattened[path] for path in PRIMARY_FIELD_PATHS]
        split_counts = counts[row["split"]]
        condition_counts = split_counts["by_condition"][row["condition"]]
        for target in (split_counts, condition_counts):
            target["assigned_input_count"] += 1
            target["field_opportunities"] += len(values)
            target["ground_truth_non_null_opportunities"] += sum(
                value is not None for value in values
            )
            target["ground_truth_null_opportunities"] += sum(value is None for value in values)

    expected = {
        "development": {
            "assigned_input_count": 4,
            "field_opportunities": 64,
            "ground_truth_non_null_opportunities": 52,
            "ground_truth_null_opportunities": 12,
            "condition_assigned_input_count": 2,
            "condition_field_opportunities": 32,
        },
        "formal": {
            "assigned_input_count": 20,
            "field_opportunities": 320,
            "ground_truth_non_null_opportunities": 196,
            "ground_truth_null_opportunities": 124,
            "condition_assigned_input_count": 10,
            "condition_field_opportunities": 160,
        },
    }
    for split, expected_values in expected.items():
        observed = counts[split]
        for key in (
            "assigned_input_count",
            "field_opportunities",
            "ground_truth_non_null_opportunities",
            "ground_truth_null_opportunities",
        ):
            if observed[key] != expected_values[key]:
                _fail(
                    f"V5 {split} denominator {key} is {observed[key]}, "
                    f"expected {expected_values[key]}"
                )
        for condition in CONDITIONS:
            condition_counts = observed["by_condition"][condition]
            if (
                condition_counts["assigned_input_count"]
                != expected_values["condition_assigned_input_count"]
                or condition_counts["field_opportunities"]
                != expected_values["condition_field_opportunities"]
            ):
                _fail(f"V5 {split} {condition} denominator matrix is invalid")
    return counts


def collect_and_validate_corpus(
    root: Path,
    rows: Sequence[Mapping[str, str]],
    cases: Sequence[Mapping[str, str]],
) -> Tuple[Dict[str, str], Dict[str, str], Dict[str, Any]]:
    operational_paths, evaluation_only_paths = expected_corpus_paths(rows)
    data_namespace_files = {
        path for path in evaluation_only_paths if path.startswith("data/v5/")
    } | {V5_MANIFEST_RELATIVE_PATH, RUNTIME_PROFILE_RELATIVE_PATH}
    output_namespace_files = {path for path in operational_paths if path.startswith("output/v5/")}
    validate_exact_namespace(root, "data/v5", sorted(data_namespace_files))
    validate_exact_namespace(root, "output/v5", sorted(output_namespace_files))
    _validate_pre_inference_namespace(root)

    all_paths = operational_paths | evaluation_only_paths
    hashes: Dict[str, str] = {}
    for relative_path in sorted(all_paths):
        path = _safe_repo_path(root, relative_path)
        if not path.is_file():
            _fail(f"Missing v5 corpus artifact: {relative_path}")
        hashes[relative_path] = sha256_file(path)

    for row in rows:
        for path_key, hash_key in (("pdf_path", "pdf_sha256"), ("image_path", "image_sha256")):
            relative_path = row[path_key]
            if hashes[relative_path] != row[hash_key]:
                _fail(
                    f"Manifest {hash_key} does not match {relative_path} "
                    f"for {row['document_id']} {row['condition']}"
                )
        source_text = _safe_repo_path(root, row["source_text_path"]).read_text(encoding="utf-8")
        if "SYNTHETIC" not in source_text or "NOT A REAL PATIENT" not in source_text:
            _fail(f"Synthetic disclosure is absent from {row['source_text_path']}")

    denominators = _validate_ground_truth_and_denominators(root, rows, cases)
    operational = {path: hashes[path] for path in sorted(operational_paths)}
    evaluation_only = {path: hashes[path] for path in sorted(evaluation_only_paths)}
    return operational, evaluation_only, denominators


def validate_runtime_profile(root: Path) -> Dict[str, Any]:
    runtime_profile = _read_json(
        root / RUNTIME_PROFILE_RELATIVE_PATH,
        "v5 runtime profile",
    )
    if runtime_profile != EXPECTED_RUNTIME_PROFILE:
        _fail("data/v5/runtime_profile.json differs from the exact protocol runtime profile")
    requirements_path = root / REQUIREMENTS_RELATIVE_PATH
    try:
        requirements_bytes = requirements_path.read_bytes()
    except OSError as exc:
        raise ProtocolFreezeError(f"Missing v5 requirements: {requirements_path}") from exc
    expected_bytes = ("\n".join(EXPECTED_REQUIREMENTS) + "\n").encode("utf-8")
    if requirements_bytes != expected_bytes:
        _fail("requirements/colab-v5.txt differs from the exact fixed package pins")
    return runtime_profile


def validate_implementation_configuration() -> None:
    try:
        runner = importlib.import_module("run_v5_inference")
        evaluator = importlib.import_module("evaluate_v5")
    except (ImportError, OSError) as exc:
        raise ProtocolFreezeError(
            "Cannot import the v5 runner/evaluator for lock validation"
        ) from exc

    expected_runner_constants = {
        "PROTOCOL_VERSION": PROTOCOL_VERSION,
        "PROTOCOL_IDENTIFIER": PROTOCOL_IDENTIFIER,
        "MODEL_ID": MODEL_ID,
        "MODEL_REVISION": MODEL_REVISION,
        "DTYPE_NAME": DTYPE_NAME,
        "RANDOM_SEED": RANDOM_SEED,
        "CANDIDATE_MAX_NEW_TOKENS": CANDIDATE_MAX_NEW_TOKENS,
        "AUDIT_MAX_NEW_TOKENS": AUDIT_MAX_NEW_TOKENS,
        "CANDIDATE_SERIALIZER_FORCE_JSON_FIELD_ORDER": (
            CANDIDATE_SERIALIZER_FORCE_JSON_FIELD_ORDER
        ),
        "CANDIDATE_SERIALIZER_MAX_CONSECUTIVE_WHITESPACES": (
            CANDIDATE_SERIALIZER_MAX_CONSECUTIVE_WHITESPACES
        ),
        "AUDIT_SERIALIZER_FORCE_JSON_FIELD_ORDER": (AUDIT_SERIALIZER_FORCE_JSON_FIELD_ORDER),
        "AUDIT_SERIALIZER_MAX_CONSECUTIVE_WHITESPACES": (
            AUDIT_SERIALIZER_MAX_CONSECUTIVE_WHITESPACES
        ),
        "TESSERACT_LANGUAGE": TESSERACT_LANGUAGE,
        "TESSERACT_CONFIG": TESSERACT_CONFIG,
        "OCR_TIMEOUT_SECONDS": TESSERACT_TIMEOUT_SECONDS,
        "CONDITIONS": CONDITIONS,
    }
    mismatches = {
        key: (expected, getattr(runner, key, None))
        for key, expected in expected_runner_constants.items()
        if getattr(runner, key, None) != expected
    }
    if mismatches:
        _fail(f"V5 runner configuration differs from protocol: {mismatches}")
    runner_paths = tuple(getattr(runner, "REQUIRED_LOCKED_SOURCE_PATHS", ()))
    if len(runner_paths) != len(set(runner_paths)) or set(runner_paths) != set(
        REQUIRED_LOCKED_SOURCE_PATHS
    ):
        _fail("V5 runner locked source inventory differs from the freezer")
    if getattr(evaluator, "GATE_THRESHOLDS", None) != GATE_THRESHOLDS:
        _fail("V5 evaluator thresholds differ from the exact prospective protocol thresholds")


def collect_locked_source_hashes(root: Path) -> Dict[str, str]:
    hashes: Dict[str, str] = {}
    for relative_path in REQUIRED_LOCKED_SOURCE_PATHS:
        path = _safe_repo_path(root, relative_path)
        if not path.is_file():
            _fail(f"Missing required locked source: {relative_path}")
        hashes[relative_path] = sha256_file(path)
    return dict(sorted(hashes.items()))


def _git(
    root: Path,
    *args: str,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            ["git", *args],
            cwd=root,
            check=check,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        detail = ""
        if isinstance(exc, subprocess.CalledProcessError):
            detail = (exc.stderr or exc.stdout or "").strip()
        raise ProtocolFreezeError(
            f"Git validation failed for {' '.join(args)}" + (f": {detail}" if detail else "")
        ) from exc


def _committed_blob(root: Path, commit: str, relative_path: str) -> bytes:
    try:
        return subprocess.check_output(
            ["git", "cat-file", "blob", f"{commit}:{relative_path}"],
            cwd=root,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ProtocolFreezeError(
            f"Could not read {relative_path} from protocol source commit {commit}"
        ) from exc


def validate_protocol_source_commit(
    root: Path,
    protocol_source_commit: str,
    bound_paths: Sequence[str],
) -> None:
    if re.fullmatch(r"[0-9a-f]{40}", protocol_source_commit) is None:
        _fail("--protocol-source-commit must be a 40-character lowercase commit SHA")
    resolved = _git(
        root,
        "rev-parse",
        "--verify",
        f"{protocol_source_commit}^{{commit}}",
    ).stdout.strip()
    if resolved != protocol_source_commit:
        _fail("--protocol-source-commit did not resolve exactly to the supplied SHA")
    head = _git(root, "rev-parse", "HEAD").stdout.strip()
    if head != protocol_source_commit:
        _fail("--protocol-source-commit must equal current HEAD")

    unique_paths = sorted(set(bound_paths))
    if not unique_paths:
        _fail("Protocol lock-bound path inventory is empty")
    status = _git(
        root,
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
        "--",
        *unique_paths,
    ).stdout.strip()
    if status:
        _fail(f"Lock-bound paths are dirty or untracked and must be committed: {status}")

    for relative_path in unique_paths:
        path = _safe_repo_path(root, relative_path)
        if not path.is_file():
            _fail(f"Lock-bound artifact is missing: {relative_path}")
        result = _git(
            root,
            "cat-file",
            "-e",
            f"{protocol_source_commit}:{relative_path}",
            check=False,
        )
        if result.returncode != 0:
            _fail(f"Lock-bound artifact is not in protocol source commit: {relative_path}")
        committed = _committed_blob(root, protocol_source_commit, relative_path)
        if sha256_bytes(committed) != sha256_file(path):
            _fail(f"Working artifact differs from protocol source commit: {relative_path}")


def build_lock_payload(
    *,
    protocol_source_commit: str,
    runtime_profile: Mapping[str, Any],
    locked_source_files: Mapping[str, str],
    corpus_artifacts: Mapping[str, str],
    evaluation_only_corpus_artifacts: Mapping[str, str],
    freshness_reference_artifacts: Mapping[str, str],
    semantic_freshness: Mapping[str, Any],
    denominators: Mapping[str, Any],
) -> Dict[str, Any]:
    operational_paths = sorted(corpus_artifacts)
    evaluation_paths = sorted(evaluation_only_corpus_artifacts)
    return {
        "status": "frozen_pre_inference",
        "protocol_version": PROTOCOL_VERSION,
        "protocol_identifier": PROTOCOL_IDENTIFIER,
        "protocol_source_commit": protocol_source_commit,
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "fixed_configuration": dict(FIXED_CONFIGURATION),
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "resolved_model_revision": MODEL_REVISION,
        "dtype": DTYPE_NAME,
        "random_seed": RANDOM_SEED,
        "development_output_count": DEVELOPMENT_OUTPUT_COUNT,
        "formal_output_count": FORMAL_OUTPUT_COUNT,
        "development_case_ids": list(DEVELOPMENT_CASE_IDS),
        "formal_case_ids": list(FORMAL_CASE_IDS),
        "template_ids": list(TEMPLATE_IDS),
        "conditions": list(CONDITIONS),
        "gate_thresholds": GATE_THRESHOLDS,
        "denominators": denominators,
        "runtime_profile": dict(runtime_profile),
        "runtime_profile_path": RUNTIME_PROFILE_RELATIVE_PATH,
        "requirements_path": REQUIREMENTS_RELATIVE_PATH,
        "locked_source_files": dict(sorted(locked_source_files.items())),
        "corpus_artifacts": dict(sorted(corpus_artifacts.items())),
        "evaluation_only_corpus_artifacts": dict(sorted(evaluation_only_corpus_artifacts.items())),
        "freshness_reference_artifacts": dict(sorted(freshness_reference_artifacts.items())),
        "semantic_freshness": dict(semantic_freshness),
        "corpus_inventory": {
            "operational_count": len(operational_paths),
            "operational_paths": operational_paths,
            "evaluation_only_count": len(evaluation_paths),
            "evaluation_only_paths": evaluation_paths,
            "total_count": len(operational_paths) + len(evaluation_paths),
        },
        "claim_scope": (
            "descriptive in-distribution synthetic-pilot feasibility across five independent "
            "formal fact vectors, two known layouts, and paired clean/OCR-degraded inputs"
        ),
    }


def _write_protocol_lock(root: Path, payload: Mapping[str, Any]) -> Path:
    path = root / PROTOCOL_LOCK_RELATIVE_PATH
    if path.exists() or path.is_symlink():
        _fail(f"Refusing to overwrite existing protocol lock: {path}")
    parent = path.parent
    if not parent.is_dir() or parent.is_symlink():
        _fail(f"Protocol-lock parent is missing or unsafe: {parent}")
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    try:
        with path.open("xb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError as exc:
        raise ProtocolFreezeError(f"Refusing to overwrite existing protocol lock: {path}") from exc
    return path


def freeze_protocol(
    protocol_source_commit: str,
    *,
    root: Path | None = None,
) -> Dict[str, Any]:
    root = ROOT if root is None else root.resolve()
    lock_path = root / PROTOCOL_LOCK_RELATIVE_PATH
    if lock_path.exists() or lock_path.is_symlink():
        _fail(f"Refusing to overwrite existing protocol lock: {lock_path}")

    cases, semantic_freshness, freshness_reference_artifacts = load_and_validate_cases(root)
    rows = load_and_validate_manifest(root, cases)
    runtime_profile = validate_runtime_profile(root)
    validate_implementation_configuration()
    locked_source_files = collect_locked_source_hashes(root)
    corpus_artifacts, evaluation_only_corpus_artifacts, denominators = collect_and_validate_corpus(
        root, rows, cases
    )
    bound_paths = [
        *locked_source_files,
        *corpus_artifacts,
        *evaluation_only_corpus_artifacts,
        *freshness_reference_artifacts,
    ]
    validate_protocol_source_commit(root, protocol_source_commit, bound_paths)
    payload = build_lock_payload(
        protocol_source_commit=protocol_source_commit,
        runtime_profile=runtime_profile,
        locked_source_files=locked_source_files,
        corpus_artifacts=corpus_artifacts,
        evaluation_only_corpus_artifacts=evaluation_only_corpus_artifacts,
        freshness_reference_artifacts=freshness_reference_artifacts,
        semantic_freshness=semantic_freshness,
        denominators=denominators,
    )
    _write_protocol_lock(root, payload)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol-source-commit",
        required=True,
        help="Exact 40-character current HEAD containing every lock-bound source and corpus byte.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        payload = freeze_protocol(args.protocol_source_commit)
    except ProtocolFreezeError as exc:
        raise SystemExit(f"V5 protocol freeze refused: {exc}") from exc
    print(
        f"WROTE {PROTOCOL_LOCK_RELATIVE_PATH} for source commit "
        f"{payload['protocol_source_commit']}."
    )


if __name__ == "__main__":
    main()
