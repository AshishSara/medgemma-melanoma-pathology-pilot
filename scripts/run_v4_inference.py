#!/usr/bin/env python3
"""Run the evidence-gated v4 MedGemma 1.5 protocol."""

from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
import math
import os
import platform
import re
import subprocess
import tempfile
import time
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from jsonschema import Draft202012Validator
from PIL import Image
from pilot_utils import (
    MODEL_REVISIONS,
    ROOT,
    read_json,
    sha256_bytes,
    sha256_file,
    slugify_model_id,
    write_json,
)
from v3_pipeline import compile_prediction, sectionize_ocr_lines

PROTOCOL_VERSION = "evidence-gated-v4"
PROTOCOL_IDENTIFIER = "prespecified-serialization-recovery-pilot-v4"
FORMAL_EXECUTION_ATTEMPT_ID = "sole-formal-attempt"
DEVELOPMENT_ATTEMPT_ID = "sole-development-smoke"
MODEL_ID = "google/medgemma-1.5-4b-it"
MODEL_REVISION = MODEL_REVISIONS[MODEL_ID]
MODEL_SLUG = slugify_model_id(MODEL_ID)
DTYPE_NAME = "bfloat16"
CANDIDATE_MAX_NEW_TOKENS = 512
AUDIT_MAX_NEW_TOKENS = 1280
CANDIDATE_SERIALIZER_FORCE_JSON_FIELD_ORDER = False
CANDIDATE_SERIALIZER_MAX_CONSECUTIVE_WHITESPACES = 12
AUDIT_SERIALIZER_FORCE_JSON_FIELD_ORDER = True
AUDIT_SERIALIZER_MAX_CONSECUTIVE_WHITESPACES = 0
TESSERACT_LANGUAGE = "eng"
TESSERACT_CONFIG = "--oem 1 --psm 6"
OCR_TIMEOUT_SECONDS = 120
RANDOM_SEED = 0
CONDITIONS = ("clean", "ocr_degraded")
SPLITS = ("development", "formal")

V4_MANIFEST = ROOT / "data" / "v4" / "report_manifest.csv"
PROTOCOL_LOCK_PATH = ROOT / "data" / "v4" / "protocol_lock.json"
CANDIDATE_PROMPT_PATH = ROOT / "prompts" / "v3" / "candidate_prompt.txt"
AUDIT_PROMPT_PATH = ROOT / "prompts" / "v3" / "audit_prompt.txt"
CANDIDATE_SCHEMA_PATH = ROOT / "schema" / "v3" / "candidate.schema.json"
AUDIT_SCHEMA_PATH = ROOT / "schema" / "v3" / "audit.schema.json"
CANONICAL_SCHEMA_PATH = ROOT / "schema" / "extraction.schema.json"
COMPILER_PATH = ROOT / "scripts" / "v3_pipeline.py"
SCRIPT_PATH = Path(__file__).resolve()
OCR_MARKER = "{{OCR_LINES}}"
REQUIRED_LOCKED_SOURCE_PATHS = (
    "data/v4/report_manifest.csv",
    "docs/protocol_v4.md",
    "prompts/v3/audit_prompt.txt",
    "prompts/v3/candidate_prompt.txt",
    "pyproject.toml",
    "schema/extraction.schema.json",
    "schema/v3/audit.schema.json",
    "schema/v3/candidate.schema.json",
    "scripts/evaluate_v4.py",
    "scripts/freeze_v4_protocol.py",
    "scripts/generate_v4_reports.py",
    "scripts/pilot_utils.py",
    "scripts/run_v4_inference.py",
    "scripts/v3_pipeline.py",
    "uv.lock",
)


class StrictGeneratedJSONError(ValueError):
    """Raised when a constrained generation is not strict JSON."""


def reject_duplicate_keys(pairs: List[Tuple[str, Any]]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise StrictGeneratedJSONError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def reject_non_finite(value: str) -> None:
    raise StrictGeneratedJSONError(f"Non-finite JSON number: {value}")


def parse_generated_json(raw_text: str, schema: Mapping[str, Any]) -> Dict[str, Any]:
    try:
        parsed = json.loads(
            raw_text.strip(),
            object_pairs_hook=reject_duplicate_keys,
            parse_constant=reject_non_finite,
        )
    except json.JSONDecodeError as exc:
        raise StrictGeneratedJSONError(
            f"{exc.msg} at line {exc.lineno}, column {exc.colno}"
        ) from exc
    if not isinstance(parsed, dict):
        raise StrictGeneratedJSONError("Top-level JSON value must be an object")
    errors = sorted(
        Draft202012Validator(schema).iter_errors(parsed),
        key=lambda error: list(error.absolute_path),
    )
    if errors:
        details = "; ".join(
            f"{'.'.join(str(part) for part in error.absolute_path) or '<root>'}: {error.message}"
            for error in errors
        )
        raise StrictGeneratedJSONError(f"Generated JSON violates its pass schema: {details}")
    return parsed


def relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def repository_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        text=True,
    ).strip()


def _is_canonical_repo_relative_path(relative_path: Any) -> bool:
    if (
        not isinstance(relative_path, str)
        or not relative_path
        or "\\" in relative_path
        or "\x00" in relative_path
    ):
        return False
    path = Path(relative_path)
    return (
        not path.is_absolute()
        and relative_path == path.as_posix()
        and all(part not in {"", ".", ".."} for part in path.parts)
    )


def _valid_sha256(value: Any) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _valid_rate(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and 0 <= float(value) <= 1
    )


def development_lock_errors(lock: Mapping[str, Any]) -> List[str]:
    """Validate and hash the frozen development evidence without formal truth access."""

    errors: List[str] = []
    expected_criteria = {
        "provenance_4_of_4": True,
        "candidate_strict_json_schema_4_of_4": True,
        "audit_strict_json_schema_4_of_4": True,
        "canonical_schema_100_percent": True,
        "evidence_acceptance_100_percent": True,
        "zero_candidate_cap_hits": True,
        "zero_audit_cap_hits": True,
        "zero_unsupported_fields": True,
        "zero_history_carryover": True,
    }
    gate = lock.get("development_gate")
    expected_gate_keys = {
        "status",
        "criteria",
        "provenance_valid_count",
        "canonical_schema_valid_rate",
        "pooled_field_exact_match",
        "condition_field_exact_match",
        "unsupported_field_rate",
        "unsupported_field_count",
        "generation_cap_hit_count",
        "accuracy_selection_note",
        "accepted_non_null_count",
        "accepted_with_valid_non_history_evidence_count",
        "accepted_non_null_evidence_rate",
        "history_carryover_count",
    }
    if not isinstance(gate, Mapping):
        errors.append("development_gate is not a mapping")
        gate = {}
    elif set(gate) != expected_gate_keys:
        errors.append("development_gate does not match the exact expected shape")
    if gate.get("status") != "PASS":
        errors.append("development_gate status is not PASS")
    if gate.get("criteria") != expected_criteria:
        errors.append("development_gate criteria are not the exact passing smoke criteria")
    if gate.get("provenance_valid_count") != 4:
        errors.append("development_gate provenance count is not 4")
    if gate.get("canonical_schema_valid_rate") != 1.0:
        errors.append("development_gate canonical schema rate is not 1.0")
    if not _valid_rate(gate.get("pooled_field_exact_match")):
        errors.append("development_gate descriptive pooled exact match is invalid")
    condition_exact = gate.get("condition_field_exact_match")
    if (
        not isinstance(condition_exact, Mapping)
        or set(condition_exact) != set(CONDITIONS)
        or any(not _valid_rate(condition_exact.get(condition)) for condition in CONDITIONS)
    ):
        errors.append("development_gate condition exact-match mapping is invalid")
    if gate.get("unsupported_field_rate") != 0.0:
        errors.append("development_gate unsupported-field rate is not zero")
    if gate.get("unsupported_field_count") != 0:
        errors.append("development_gate unsupported-field count is not zero")
    if gate.get("generation_cap_hit_count") != {"candidate": 0, "audit": 0}:
        errors.append("development_gate generation cap-hit counts are not zero")
    if gate.get("accuracy_selection_note") != (
        "Exact-match accuracy is descriptive only and is not a development selection threshold."
    ):
        errors.append("development_gate accuracy-selection note is invalid")
    accepted = gate.get("accepted_non_null_count")
    accepted_valid = gate.get("accepted_with_valid_non_history_evidence_count")
    if (
        not isinstance(accepted, int)
        or isinstance(accepted, bool)
        or accepted < 0
        or accepted_valid != accepted
    ):
        errors.append("development_gate accepted-evidence counts are invalid")
    if gate.get("accepted_non_null_evidence_rate") != 1.0:
        errors.append("development_gate accepted-evidence rate is not 1.0")
    if gate.get("history_carryover_count") != 0:
        errors.append("development_gate history carryover is not zero")

    artifacts = lock.get("development_artifacts")
    development_root = ROOT / "results" / "v4" / "development" / "iteration-1"
    if not isinstance(artifacts, Mapping) or not artifacts:
        errors.append("development_artifacts is missing or empty")
        artifacts = {}
    invalid_artifact_entries = [
        str(relative_path)
        for relative_path, expected_sha in artifacts.items()
        if (
            not _is_canonical_repo_relative_path(relative_path)
            or not relative_path.startswith("results/v4/development/iteration-1/")
            or not _valid_sha256(expected_sha)
        )
    ]
    if invalid_artifact_entries:
        errors.append(
            "development_artifacts contains unsafe or invalid entries: "
            f"{sorted(invalid_artifact_entries)}"
        )
    if not development_root.is_dir() or development_root.is_symlink():
        errors.append("development iteration directory is missing or unsafe")
        discovered: set[str] = set()
    else:
        discovered = set()
        for path in development_root.rglob("*"):
            if path.is_symlink():
                errors.append(f"development artifact is a symlink: {relative(path)}")
            elif path.is_file():
                discovered.add(relative(path))
    if set(artifacts) != discovered:
        errors.append("development_artifacts does not match the exact on-disk inventory")
    if not invalid_artifact_entries and set(artifacts) == discovered:
        for relative_path, expected_sha in artifacts.items():
            path = ROOT / relative_path
            if sha256_file(path) != expected_sha:
                errors.append(f"development artifact hash mismatch: {relative_path}")

    metrics_relative = "results/v4/development/iteration-1/metrics.json"
    iteration_relative = "results/v4/development/iteration-1/iteration_record.json"
    claim_relative = (
        "results/v4/development/iteration-1/execution_attempts/"
        f"{DEVELOPMENT_ATTEMPT_ID}.claimed.json"
    )
    scalar_hashes = {
        metrics_relative: lock.get("development_metrics_sha256"),
        iteration_relative: lock.get("development_iteration_record_sha256"),
    }
    for relative_path, scalar_sha in scalar_hashes.items():
        if artifacts.get(relative_path) != scalar_sha or not _valid_sha256(scalar_sha):
            errors.append(f"development scalar hash does not bind {relative_path}")
    if claim_relative not in artifacts:
        errors.append("development one-shot claim is absent from development_artifacts")

    summaries = lock.get("development_iterations")
    expected_summary = {
        "iteration": 1,
        "status": "selected_complete",
        "documentation_path": iteration_relative,
        "documentation_sha256": artifacts.get(iteration_relative),
        "artifact_count": len(artifacts),
    }
    if summaries != [expected_summary]:
        errors.append("development_iterations summary is invalid")

    development_commit = lock.get("development_repository_commit")
    if not isinstance(development_commit, str) or not re.fullmatch(
        r"[0-9a-f]{40}", development_commit
    ):
        errors.append("development_repository_commit is invalid")

    metrics_path = ROOT / metrics_relative
    if metrics_path.is_file() and not metrics_path.is_symlink():
        try:
            metrics = read_json(metrics_path)
        except (OSError, ValueError) as exc:
            errors.append(f"development metrics are unreadable: {exc}")
        else:
            expected_metrics = {
                "status": "DEVELOPMENT_PASS",
                "protocol_version": PROTOCOL_VERSION,
                "iteration": 1,
                "output_count": 4,
                "provenance_valid_count": 4,
                "pass": True,
                "model_id": MODEL_ID,
                "model_revision": MODEL_REVISION,
            }
            for key, value in expected_metrics.items():
                if metrics.get(key) != value:
                    errors.append(f"development metrics {key} is invalid")
            if metrics.get("smoke_criteria") != expected_criteria:
                errors.append("development metrics smoke criteria differ from the lock")
            telemetry = metrics.get("generation_telemetry")
            if not isinstance(telemetry, Mapping) or any(
                not isinstance(telemetry.get(pass_name), Mapping)
                or telemetry[pass_name].get("cap_hit_count") != 0
                for pass_name in ("candidate", "audit")
            ):
                errors.append("development metrics generation telemetry is invalid")
    elif metrics_relative in artifacts:
        errors.append("development metrics file is missing or unsafe")

    iteration_path = ROOT / iteration_relative
    if iteration_path.is_file() and not iteration_path.is_symlink():
        try:
            iteration_record = read_json(iteration_path)
        except (OSError, ValueError) as exc:
            errors.append(f"development iteration record is unreadable: {exc}")
        else:
            expected_iteration = {
                "status": "DEVELOPMENT_PASS",
                "protocol_version": PROTOCOL_VERSION,
                "iteration": 1,
                "repository_commit": development_commit,
                "metrics_path": metrics_relative,
                "metrics_sha256": artifacts.get(metrics_relative),
            }
            for key, value in expected_iteration.items():
                if iteration_record.get(key) != value:
                    errors.append(f"development iteration record {key} is invalid")
    elif iteration_relative in artifacts:
        errors.append("development iteration record is missing or unsafe")

    claim_path = ROOT / claim_relative
    if claim_path.is_file() and not claim_path.is_symlink():
        try:
            claim = read_json(claim_path)
        except (OSError, ValueError) as exc:
            errors.append(f"development one-shot claim is unreadable: {exc}")
        else:
            expected_claim = {
                "execution_attempt_id": DEVELOPMENT_ATTEMPT_ID,
                "status": "claimed",
                "protocol_version": PROTOCOL_VERSION,
                "repository_commit": development_commit,
                "manifest_path": relative(V4_MANIFEST),
                "manifest_sha256": sha256_file(V4_MANIFEST),
                "selected_row_count": 4,
                "condition": "all",
            }
            for key, value in expected_claim.items():
                if claim.get(key) != value:
                    errors.append(f"development one-shot claim {key} is invalid")
    elif claim_relative in artifacts:
        errors.append("development one-shot claim is missing or unsafe")
    return errors


def v3_provenance_lock_errors(
    lock: Mapping[str, Any],
    manifest_rows: Sequence[Mapping[str, str]],
) -> List[str]:
    """Validate preserved v3 evidence without reading v4 formal truth-side files."""

    errors: List[str] = []
    provenance = lock.get("v3_boundary_and_reuse_provenance")
    expected_keys = {
        "reuse_disposition",
        "byte_identical_reused_artifacts",
        "v3_protocol_lock",
        "v3_formal_attempt_events",
        "v3_formal_model_artifacts",
        "v3_formal_model_artifact_case_ids",
        "v3_unexecuted_reused_formal_case_ids",
    }
    if not isinstance(provenance, Mapping):
        return ["v3_boundary_and_reuse_provenance is not a mapping"]
    if set(provenance) != expected_keys:
        errors.append("v3 provenance does not match the exact expected shape")
    expected_disposition = {
        "MEL-104": "development_only_exposed_in_v3",
        "MEL-105": "formal_reuse_no_v3_model_call",
        "MEL-106": "formal_reuse_no_v3_model_call",
        "MEL-107": "formal_reuse_no_v3_model_call",
        "MEL-108": "formal_reuse_no_v3_model_call",
        "MEL-109": "new_v4_formal",
    }
    if provenance.get("reuse_disposition") != expected_disposition:
        errors.append("v3 reuse disposition is invalid")
    if provenance.get("v3_formal_model_artifact_case_ids") != ["MEL-104"]:
        errors.append("v3 exposed formal model-artifact case inventory is invalid")
    if provenance.get("v3_unexecuted_reused_formal_case_ids") != [
        "MEL-105",
        "MEL-106",
        "MEL-107",
        "MEL-108",
    ]:
        errors.append("v3 unexecuted reused-formal case inventory is invalid")

    v3_lock = provenance.get("v3_protocol_lock")
    if not isinstance(v3_lock, Mapping) or set(v3_lock) != {"path", "sha256"}:
        errors.append("v3 protocol-lock descriptor is invalid")
    else:
        path_value = v3_lock.get("path")
        if path_value != "data/v3/protocol_lock.json" or not _valid_sha256(v3_lock.get("sha256")):
            errors.append("v3 protocol-lock descriptor values are invalid")
        else:
            path = ROOT / path_value
            if not path.is_file() or path.is_symlink() or sha256_file(path) != v3_lock["sha256"]:
                errors.append("v3 protocol-lock hash mismatch")

    attempts = provenance.get("v3_formal_attempt_events")
    attempts_dir = ROOT / "results" / "v3" / "formal" / "execution_attempts"
    expected_attempt_paths: set[str] = set()
    if attempts_dir.is_dir() and not attempts_dir.is_symlink():
        for path in attempts_dir.iterdir():
            if (
                not path.is_file()
                or path.is_symlink()
                or not re.fullmatch(r".+\.(started|failed)\.json", path.name)
            ):
                errors.append(f"unsafe or unexpected v3 attempt-ledger entry: {relative(path)}")
            else:
                expected_attempt_paths.add(relative(path))
    else:
        errors.append("v3 formal attempt-ledger directory is missing or unsafe")
    if not isinstance(attempts, Mapping) or set(attempts) != expected_attempt_paths:
        errors.append("v3 formal attempt-event inventory is invalid")
        attempts = {}
    else:
        for relative_path, expected_sha in attempts.items():
            path = ROOT / relative_path
            if not _valid_sha256(expected_sha) or sha256_file(path) != expected_sha:
                errors.append(f"v3 formal attempt-event hash mismatch: {relative_path}")
    failed_paths = [
        ROOT / relative_path for relative_path in attempts if relative_path.endswith(".failed.json")
    ]
    if len(failed_paths) != 1:
        errors.append("v3 provenance must bind exactly one failed event")
    else:
        try:
            failed_event = read_json(failed_paths[0])
        except (OSError, ValueError) as exc:
            errors.append(f"v3 failed event is unreadable: {exc}")
        else:
            expected_failure = {
                "current_row": {
                    "document_id": "MEL-104-B",
                    "condition": "ocr_degraded",
                },
                "failure_stage": "audit_validation",
                "valid_model_response_count_for_current_row": 1,
                "resume_permitted_for_current_row": False,
                "completed_new_row_count": 3,
            }
            for key, value in expected_failure.items():
                if failed_event.get(key) != value:
                    errors.append(f"v3 failed-event {key} is invalid")

    model_artifacts = provenance.get("v3_formal_model_artifacts")
    v3_formal_root = ROOT / "results" / "v3" / "formal"
    discovered_model_artifacts: set[str] = set()
    if v3_formal_root.is_dir() and not v3_formal_root.is_symlink():
        for path in v3_formal_root.rglob("*"):
            if path.is_symlink():
                errors.append(f"v3 formal artifact is a symlink: {relative(path)}")
            elif path.is_file() and re.search(r"MEL-[0-9]{3}", path.name):
                discovered_model_artifacts.add(relative(path))
    else:
        errors.append("v3 formal result directory is missing or unsafe")
    if (
        not isinstance(model_artifacts, Mapping)
        or set(model_artifacts) != discovered_model_artifacts
    ):
        errors.append("v3 formal model-artifact inventory is invalid")
        model_artifacts = {}
    else:
        for relative_path, expected_sha in model_artifacts.items():
            path = ROOT / relative_path
            if (
                "MEL-104" not in path.name
                or not _valid_sha256(expected_sha)
                or sha256_file(path) != expected_sha
            ):
                errors.append(f"v3 formal model-artifact mismatch: {relative_path}")

    comparisons = provenance.get("byte_identical_reused_artifacts")
    expected_v4_paths: set[str] = set()
    for row in manifest_rows:
        if row.get("semantic_case_id") in {
            "MEL-104",
            "MEL-105",
            "MEL-106",
            "MEL-107",
            "MEL-108",
        }:
            expected_v4_paths.update(
                row[field]
                for field in (
                    "source_text_path",
                    "ground_truth_path",
                    "pdf_path",
                    "image_path",
                )
            )
    if not isinstance(comparisons, Mapping) or set(comparisons) != expected_v4_paths:
        errors.append("v3/v4 byte-identical reuse comparison inventory is invalid")
        comparisons = {}
    corpus_hashes: Dict[str, Any] = {}
    for section in ("corpus_artifacts", "evaluation_only_corpus_artifacts"):
        value = lock.get(section)
        if isinstance(value, Mapping):
            corpus_hashes.update(value)
    for v4_relative, descriptor in comparisons.items():
        expected_v3_relative = v4_relative.replace("/v4/", "/v3/", 1)
        if (
            not isinstance(descriptor, Mapping)
            or set(descriptor) != {"v3_path", "v3_sha256", "v4_sha256"}
            or descriptor.get("v3_path") != expected_v3_relative
            or not _valid_sha256(descriptor.get("v3_sha256"))
            or descriptor.get("v4_sha256") != descriptor.get("v3_sha256")
            or corpus_hashes.get(v4_relative) != descriptor.get("v4_sha256")
        ):
            errors.append(f"invalid v3/v4 reuse descriptor: {v4_relative}")
            continue
        if expected_v3_relative.startswith(("data/v3/source_text/", "data/v3/ground_truth/")):
            # These are byte-identical aliases of v4 formal truth-side artifacts.
            # Their descriptor hashes are lock-cross-checked above, but operational
            # inference must not open the underlying files before output sealing.
            continue
        v3_path = ROOT / expected_v3_relative
        if (
            not v3_path.is_file()
            or v3_path.is_symlink()
            or sha256_file(v3_path) != descriptor["v3_sha256"]
        ):
            errors.append(f"v3 reused artifact hash mismatch: {expected_v3_relative}")
    return errors


def post_development_immutability_lock_errors(lock: Mapping[str, Any]) -> List[str]:
    """Cross-check the freeze-time dev-commit equality inventory without formal GT reads."""

    errors: List[str] = []
    hardening = lock.get("post_development_source_hardening")
    development_commit = lock.get("development_repository_commit")
    if not isinstance(hardening, Mapping):
        return ["post_development_source_hardening is not a mapping"]
    if hardening.get("classification") != ("byte_identical_sources_corpus_and_v3_provenance"):
        errors.append("post-development immutability classification is invalid")
    if hardening.get("development_commit") != development_commit:
        errors.append("post-development immutability commit is invalid")
    inventory = hardening.get("immutable_path_sha256")
    provenance = lock.get("v3_boundary_and_reuse_provenance")
    expected_paths = set(REQUIRED_LOCKED_SOURCE_PATHS)
    section_hashes: Dict[str, Any] = {}
    for section in (
        "locked_source_files",
        "corpus_artifacts",
        "evaluation_only_corpus_artifacts",
    ):
        value = lock.get(section)
        if isinstance(value, Mapping):
            expected_paths.update(value)
            section_hashes.update(value)
    if isinstance(provenance, Mapping):
        comparisons = provenance.get("byte_identical_reused_artifacts")
        if isinstance(comparisons, Mapping):
            for v4_relative, descriptor in comparisons.items():
                expected_paths.add(v4_relative)
                if isinstance(descriptor, Mapping):
                    v3_relative = descriptor.get("v3_path")
                    if isinstance(v3_relative, str):
                        expected_paths.add(v3_relative)
                        section_hashes[v3_relative] = descriptor.get("v3_sha256")
        v3_lock = provenance.get("v3_protocol_lock")
        if isinstance(v3_lock, Mapping) and isinstance(v3_lock.get("path"), str):
            expected_paths.add(v3_lock["path"])
            section_hashes[v3_lock["path"]] = v3_lock.get("sha256")
        for key in ("v3_formal_attempt_events", "v3_formal_model_artifacts"):
            mapping = provenance.get(key)
            if isinstance(mapping, Mapping):
                expected_paths.update(mapping)
                section_hashes.update(mapping)
    if not isinstance(inventory, Mapping) or set(inventory) != expected_paths:
        errors.append("post-development immutable path inventory is incomplete")
        return errors
    for relative_path, expected_sha in inventory.items():
        if not _valid_sha256(expected_sha):
            errors.append(f"invalid post-development immutable hash: {relative_path}")
        elif relative_path in section_hashes and section_hashes[relative_path] != expected_sha:
            errors.append(
                f"post-development immutable hash disagrees with lock section: {relative_path}"
            )
    return errors


def verify_protocol_lock() -> Dict[str, Any]:
    def is_evaluation_only_corpus_path(relative_path: str) -> bool:
        return relative_path == "data/v4/cases.csv" or relative_path.startswith(
            ("data/v4/ground_truth/", "data/v4/source_text/")
        )

    def is_canonical_repo_relative_path(relative_path: Any) -> bool:
        if (
            not isinstance(relative_path, str)
            or not relative_path
            or "\\" in relative_path
            or "\x00" in relative_path
        ):
            return False
        path = Path(relative_path)
        return (
            not path.is_absolute()
            and relative_path == path.as_posix()
            and all(part not in {"", ".", ".."} for part in path.parts)
        )

    if PROTOCOL_LOCK_PATH.is_symlink() or not PROTOCOL_LOCK_PATH.is_file():
        raise SystemExit(
            "Formal v4 is locked out until a safe data/v4/protocol_lock.json "
            "is created and committed."
        )
    lock = read_json(PROTOCOL_LOCK_PATH)
    expected_scalars = {
        "status": "frozen",
        "protocol_version": PROTOCOL_VERSION,
        "protocol_identifier": PROTOCOL_IDENTIFIER,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "resolved_model_revision": MODEL_REVISION,
        "dtype": DTYPE_NAME,
        "random_seed": RANDOM_SEED,
        "candidate_max_new_tokens": CANDIDATE_MAX_NEW_TOKENS,
        "audit_max_new_tokens": AUDIT_MAX_NEW_TOKENS,
        "candidate_serializer_force_json_field_order": (
            CANDIDATE_SERIALIZER_FORCE_JSON_FIELD_ORDER
        ),
        "candidate_serializer_max_consecutive_whitespaces": (
            CANDIDATE_SERIALIZER_MAX_CONSECUTIVE_WHITESPACES
        ),
        "audit_serializer_force_json_field_order": AUDIT_SERIALIZER_FORCE_JSON_FIELD_ORDER,
        "audit_serializer_max_consecutive_whitespaces": (
            AUDIT_SERIALIZER_MAX_CONSECUTIVE_WHITESPACES
        ),
        "tesseract_language": TESSERACT_LANGUAGE,
        "tesseract_config": TESSERACT_CONFIG,
        "tesseract_timeout_seconds": OCR_TIMEOUT_SECONDS,
        "formal_output_count": 20,
        "development_iterations_used": 1,
        "selected_development_iteration": 1,
        "candidate_calls_per_input": 1,
        "audit_calls_per_input": 1,
        "formal_execution_attempt_id": FORMAL_EXECUTION_ATTEMPT_ID,
        "development_attempt_id": DEVELOPMENT_ATTEMPT_ID,
        "assistant_prefill": False,
        "do_sample": False,
        "num_beams": 1,
    }
    errors = [
        f"{key}: expected {value!r}, found {lock.get(key)!r}"
        for key, value in expected_scalars.items()
        if lock.get(key) != value
    ]
    lock_relative_path = relative(PROTOCOL_LOCK_PATH)
    tracked_lock = subprocess.run(
        ["git", "ls-files", "--error-unmatch", "--", lock_relative_path],
        cwd=ROOT,
        check=False,
        capture_output=True,
    )
    if tracked_lock.returncode != 0:
        errors.append("protocol_lock.json is not tracked in the execution repository")
    else:
        lock_status = subprocess.run(
            [
                "git",
                "status",
                "--porcelain",
                "--untracked-files=no",
                "--",
                lock_relative_path,
            ],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        if lock_status.returncode != 0:
            errors.append("could not verify protocol_lock.json worktree status")
        elif lock_status.stdout.strip():
            errors.append("protocol_lock.json differs from the execution commit")
        try:
            committed_lock = subprocess.check_output(
                ["git", "cat-file", "blob", f"HEAD:{lock_relative_path}"],
                cwd=ROOT,
            )
        except (OSError, subprocess.CalledProcessError):
            errors.append("protocol_lock.json is not committed in the execution HEAD")
        else:
            if sha256_bytes(committed_lock) != sha256_file(PROTOCOL_LOCK_PATH):
                errors.append("protocol_lock.json bytes differ from the execution HEAD")
    source_commit = lock.get("protocol_source_commit")
    if not isinstance(source_commit, str) or not re.fullmatch(r"[0-9a-f]{40}", source_commit):
        errors.append("protocol_source_commit is not a 40-character lowercase SHA")
    elif (
        subprocess.run(
            ["git", "merge-base", "--is-ancestor", source_commit, "HEAD"],
            cwd=ROOT,
            check=False,
            capture_output=True,
        ).returncode
        != 0
    ):
        errors.append("protocol_source_commit is not an ancestor of the execution commit")

    locked_files = lock.get("locked_source_files")
    if not isinstance(locked_files, Mapping):
        errors.append("locked_source_files is not a mapping")
        locked_files = {}
    elif set(locked_files) != set(REQUIRED_LOCKED_SOURCE_PATHS):
        errors.append("locked_source_files does not match the exact required source inventory")
    manifest_source_valid = False
    for relative_path in REQUIRED_LOCKED_SOURCE_PATHS:
        if is_evaluation_only_corpus_path(relative_path):
            errors.append(
                f"evaluation-only path is forbidden in REQUIRED_LOCKED_SOURCE_PATHS: "
                f"{relative_path}"
            )
            continue
        path = ROOT / relative_path
        expected_sha = locked_files.get(relative_path)
        if not path.is_file() or path.is_symlink():
            errors.append(f"missing or unsafe locked source file: {relative_path}")
            continue
        actual_sha = sha256_file(path)
        if expected_sha != actual_sha:
            errors.append(f"locked source hash mismatch: {relative_path}")
            continue
        if relative_path == "data/v4/report_manifest.csv":
            manifest_source_valid = True

    expected_operational_paths = {"data/v4/report_manifest.csv"}
    expected_evaluation_only_paths = {"data/v4/cases.csv"}
    if manifest_source_valid:
        try:
            with (ROOT / "data" / "v4" / "report_manifest.csv").open(
                newline="",
                encoding="utf-8",
            ) as handle:
                manifest_rows = list(csv.DictReader(handle))
        except OSError as exc:
            errors.append(f"could not read the locked v4 manifest: {exc}")
            manifest_rows = []
    else:
        manifest_rows = []
    manifest_path_fields = (
        ("pdf_path", "output/v4/pdf/", ".pdf", expected_operational_paths),
        ("image_path", "output/v4/rendered/", ".png", expected_operational_paths),
        ("source_text_path", "data/v4/source_text/", ".txt", expected_evaluation_only_paths),
        ("ground_truth_path", "data/v4/ground_truth/", ".json", expected_evaluation_only_paths),
    )
    for row_index, row in enumerate(manifest_rows, start=2):
        for field, prefix, suffix, inventory in manifest_path_fields:
            relative_path = row.get(field)
            if (
                not is_canonical_repo_relative_path(relative_path)
                or not relative_path.startswith(prefix)
                or not relative_path.endswith(suffix)
            ):
                errors.append(
                    f"manifest row {row_index} has unsafe or noncanonical {field}: "
                    f"{relative_path!r}"
                )
                continue
            inventory.add(relative_path)

    corpus_artifacts = lock.get("corpus_artifacts")
    if not isinstance(corpus_artifacts, Mapping) or not corpus_artifacts:
        errors.append("corpus_artifacts is missing or empty")
    else:
        invalid_operational_entries = [
            relative_path
            for relative_path, expected_sha in corpus_artifacts.items()
            if (
                not is_canonical_repo_relative_path(relative_path)
                or is_evaluation_only_corpus_path(relative_path)
                or not isinstance(expected_sha, str)
                or not re.fullmatch(r"[0-9a-f]{64}", expected_sha)
            )
        ]
        if invalid_operational_entries:
            errors.append(
                "corpus_artifacts contains unsafe or invalid entries: "
                f"{sorted(map(str, invalid_operational_entries))}"
            )
        if set(corpus_artifacts) != expected_operational_paths:
            errors.append(
                "corpus_artifacts does not match the exact manifest-derived operational inventory"
            )
        if not invalid_operational_entries and set(corpus_artifacts) == (
            expected_operational_paths
        ):
            for relative_path in sorted(expected_operational_paths):
                path = ROOT / relative_path
                if not path.is_file() or path.is_symlink():
                    errors.append(f"missing or unsafe locked corpus artifact: {relative_path}")
                elif sha256_file(path) != corpus_artifacts[relative_path]:
                    errors.append(f"locked corpus hash mismatch: {relative_path}")

    evaluation_only = lock.get("evaluation_only_corpus_artifacts")
    if not isinstance(evaluation_only, Mapping) or not evaluation_only:
        errors.append("evaluation_only_corpus_artifacts is missing or empty")
    else:
        invalid_evaluation_entries = [
            relative_path
            for relative_path, expected_sha in evaluation_only.items()
            if (
                not is_canonical_repo_relative_path(relative_path)
                or not is_evaluation_only_corpus_path(relative_path)
                or not isinstance(expected_sha, str)
                or not re.fullmatch(r"[0-9a-f]{64}", expected_sha)
            )
        ]
        if invalid_evaluation_entries:
            errors.append(
                "evaluation_only_corpus_artifacts contains unsafe or invalid entries: "
                f"{sorted(map(str, invalid_evaluation_entries))}"
            )
        if set(evaluation_only) != expected_evaluation_only_paths:
            errors.append(
                "evaluation_only_corpus_artifacts does not match the exact "
                "manifest-derived evaluation-only inventory"
            )
        operational_paths = (
            set(corpus_artifacts) if isinstance(corpus_artifacts, Mapping) else set()
        )
        if set(evaluation_only) & operational_paths:
            errors.append("operational and evaluation-only corpus mappings overlap")

    development_commit = lock.get("development_repository_commit")
    if not isinstance(development_commit, str) or not re.fullmatch(
        r"[0-9a-f]{40}", development_commit
    ):
        errors.append("development_repository_commit is not a 40-character lowercase SHA")
    elif isinstance(source_commit, str) and re.fullmatch(r"[0-9a-f]{40}", source_commit):
        if (
            subprocess.run(
                ["git", "merge-base", "--is-ancestor", development_commit, source_commit],
                cwd=ROOT,
                check=False,
                capture_output=True,
            ).returncode
            != 0
        ):
            errors.append(
                "development_repository_commit is not an ancestor of protocol_source_commit"
            )
    runtime = lock.get("runtime")
    required_runtime_keys = (
        "resolved_revision",
        "device",
        "text_vocab_size",
        "random_seed",
        "accelerate_version",
        "torch_version",
        "transformers_version",
        "lm_format_enforcer_version",
        "pytesseract_version",
        "tesseract_version",
        "cuda_version",
        "cuda_device_name",
        "dtype",
        "python_version",
        "platform",
    )
    if not isinstance(runtime, Mapping):
        errors.append("runtime is not a mapping")
    else:
        for key in required_runtime_keys:
            value = runtime.get(key)
            if key in {"text_vocab_size", "random_seed"}:
                if not isinstance(value, int) or isinstance(value, bool):
                    errors.append(f"runtime.{key} is missing or invalid")
            elif not isinstance(value, str) or not value:
                errors.append(f"runtime.{key} is missing or empty")
    if lock.get("schema_enforcement") != (
        "lm-format-enforcer JsonSchemaParser via Transformers prefix_allowed_tokens_fn; "
        "candidate retains v3-effective tokenizer-adjusted defaults "
        "(force_json_field_order=false, max_consecutive_whitespaces=12); "
        "audit mutates the tokenizer-adjusted root parser after prefix construction "
        "(force_json_field_order=true, max_consecutive_whitespaces=0)"
    ):
        errors.append("schema_enforcement does not match the implemented constrained decoder")
    if lock.get("formal_execution_order") != "manifest order; candidate then blind audit per input":
        errors.append("formal_execution_order does not match the implemented runner")
    if lock.get("infrastructure_retry_rule") != (
        "one formal attempt only; no retry or resume after the attempt starts"
    ):
        errors.append("infrastructure_retry_rule does not match the implemented runner")
    if lock.get("pass_criteria") != {
        "provenance_valid_count": 20,
        "canonical_schema_valid_rate": 1.0,
        "accepted_non_null_evidence_rate": 1.0,
        "pooled_field_exact_match_minimum": 0.9,
        "each_condition_field_exact_match_minimum": 0.85,
        "unsupported_field_rate_maximum": 0.02,
        "history_carryover_count_maximum": 0,
    }:
        errors.append("pass_criteria does not match the prespecified v4 gate")
    errors.extend(development_lock_errors(lock))
    errors.extend(v3_provenance_lock_errors(lock, manifest_rows))
    errors.extend(post_development_immutability_lock_errors(lock))
    if errors:
        detail = "\n".join(f"- {error}" for error in errors)
        raise SystemExit(f"Formal v4 protocol lock validation failed:\n{detail}")
    return dict(lock)


def load_manifest(split: str, condition: str) -> List[Dict[str, str]]:
    with V4_MANIFEST.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    selected = [
        row
        for row in rows
        if row["split"] == split and (condition == "all" or row["condition"] == condition)
    ]
    if not selected:
        raise SystemExit(f"No v4 manifest rows found for split={split!r}, condition={condition!r}")
    if any(row["dataset_version"] != "pilot-v4" for row in selected):
        raise SystemExit("The selected manifest contains a non-v4 dataset row")
    return selected


def output_paths(
    row: Mapping[str, str],
    development_iteration: Optional[int] = None,
) -> Dict[str, Path]:
    if row["split"] == "development":
        if development_iteration != 1:
            raise ValueError("V4 development output paths require iteration 1")
        base = ROOT / "results" / "v4" / "development" / f"iteration-{development_iteration}"
    else:
        if development_iteration is not None:
            raise ValueError("Formal output paths do not take a development iteration")
        base = ROOT / "results" / "v4" / "formal"
    condition = row["condition"]
    document_id = row["document_id"]
    return {
        "ocr": base / "ocr" / condition / f"{document_id}.json",
        "candidate_prompt": (
            base / "rendered_prompts" / "candidate" / condition / f"{document_id}.txt"
        ),
        "audit_prompt": (base / "rendered_prompts" / "audit" / condition / f"{document_id}.txt"),
        "candidate_raw": (
            base / "raw" / "candidate" / MODEL_SLUG / condition / f"{document_id}.txt"
        ),
        "audit_raw": base / "raw" / "audit" / MODEL_SLUG / condition / f"{document_id}.txt",
        "candidate_generation": (
            base
            / "generation_metadata"
            / "candidate"
            / MODEL_SLUG
            / condition
            / f"{document_id}.json"
        ),
        "audit_generation": (
            base / "generation_metadata" / "audit" / MODEL_SLUG / condition / f"{document_id}.json"
        ),
        "candidate_parsed": (
            base / "parsed" / "candidate" / MODEL_SLUG / condition / f"{document_id}.json"
        ),
        "audit_parsed": (
            base / "parsed" / "audit" / MODEL_SLUG / condition / f"{document_id}.json"
        ),
        "normalized": (base / "normalized" / MODEL_SLUG / condition / f"{document_id}.json"),
        "compiler_audit": (
            base / "compiler_audits" / MODEL_SLUG / condition / f"{document_id}.json"
        ),
        "record": base / "run_records" / MODEL_SLUG / condition / f"{document_id}.json",
    }


def _as_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _as_confidence(value: Any) -> Optional[float]:
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        return None
    return confidence if math.isfinite(confidence) and confidence >= 0 else None


def lines_from_tesseract_data(data: Mapping[str, Sequence[Any]]) -> List[Dict[str, Any]]:
    required = ("text", "conf", "page_num", "block_num", "par_num", "line_num")
    missing = [key for key in required if key not in data]
    if missing:
        raise ValueError(f"Tesseract output lacks required columns: {', '.join(missing)}")
    lengths = {len(data[key]) for key in required}
    if len(lengths) != 1:
        raise ValueError("Tesseract output columns have inconsistent lengths")

    grouped: "OrderedDict[Tuple[int, int, int, int], Dict[str, Any]]" = OrderedDict()
    for index in range(len(data["text"])):
        word = str(data["text"][index]).strip()
        if not word:
            continue
        key = (
            _as_int(data["page_num"][index]),
            _as_int(data["block_num"][index]),
            _as_int(data["par_num"][index]),
            _as_int(data["line_num"][index]),
        )
        group = grouped.setdefault(key, {"words": [], "confidences": []})
        group["words"].append(word)
        confidence = _as_confidence(data["conf"][index])
        if confidence is not None:
            group["confidences"].append(confidence)

    lines: List[Dict[str, Any]] = []
    for ordinal, (key, group) in enumerate(grouped.items(), start=1):
        page_num, block_num, paragraph_num, source_line_num = key
        confidences = group["confidences"]
        mean_confidence = round(sum(confidences) / len(confidences), 3) if confidences else None
        lines.append(
            {
                "line_id": f"L{ordinal:03d}",
                "text": " ".join(group["words"]),
                "mean_confidence": mean_confidence,
                "page_num": page_num,
                "block_num": block_num,
                "paragraph_num": paragraph_num,
                "source_line_num": source_line_num,
            }
        )
    if not lines:
        raise RuntimeError("Tesseract produced no non-empty OCR lines")
    return lines


def validate_sectionized_lines(lines: Any) -> List[Dict[str, Any]]:
    if not isinstance(lines, list) or not lines:
        raise TypeError("sectionize_ocr_lines() must return a non-empty list")
    validated: List[Dict[str, Any]] = []
    seen = set()
    for line in lines:
        if not isinstance(line, Mapping):
            raise TypeError("Each sectionized OCR line must be a mapping")
        line_id = line.get("line_id")
        text = line.get("text")
        if not isinstance(line_id, str) or not isinstance(text, str) or not text.strip():
            raise ValueError("Every sectionized OCR line needs a string line_id and text")
        if line_id in seen:
            raise ValueError(f"Duplicate OCR line ID after sectionization: {line_id}")
        seen.add(line_id)
        validated.append(dict(line))
    return validated


def run_ocr(
    image_path: Path,
) -> Tuple[List[Dict[str, Any]], Dict[str, List[Any]], Dict[str, Any]]:
    try:
        import pytesseract
        from pytesseract import Output
    except ImportError as exc:
        raise SystemExit("V4 OCR requires `uv sync --extra inference`.") from exc

    started = time.perf_counter()
    with Image.open(image_path) as source:
        image = source.convert("RGB")
    try:
        data = pytesseract.image_to_data(
            image,
            lang=TESSERACT_LANGUAGE,
            config=TESSERACT_CONFIG,
            output_type=Output.DICT,
            timeout=OCR_TIMEOUT_SECONDS,
        )
    finally:
        image.close()
    raw_word_data = {str(key): list(values) for key, values in data.items()}
    base_lines = lines_from_tesseract_data(raw_word_data)
    sectionized = validate_sectionized_lines(sectionize_ocr_lines(base_lines))
    elapsed = time.perf_counter() - started
    return (
        sectionized,
        raw_word_data,
        {
            "engine": "tesseract",
            "tesseract_version": str(pytesseract.get_tesseract_version()),
            "pytesseract_version": importlib.metadata.version("pytesseract"),
            "language": TESSERACT_LANGUAGE,
            "config": TESSERACT_CONFIG,
            "image_preprocessing": "none",
            "output_type": "pytesseract.Output.DICT",
            "line_grouping": "ordered page_num/block_num/par_num/line_num word groups",
            "line_id_assignment": "L001..Lnnn in first-seen grouped OCR order",
            "timeout_seconds": OCR_TIMEOUT_SECONDS,
            "elapsed_seconds": elapsed,
            "line_count": len(sectionized),
        },
    )


def format_ocr_lines(lines: Sequence[Mapping[str, Any]]) -> str:
    rendered = []
    for line in lines:
        section = line.get("section")
        section_label = f" [{section}]" if isinstance(section, str) and section else ""
        rendered.append(f"[{line['line_id']}]{section_label} {line['text']}")
    return "\n".join(rendered)


def render_prompt(template_path: Path, lines: Sequence[Mapping[str, Any]]) -> str:
    template = template_path.read_text(encoding="utf-8")
    if template.count(OCR_MARKER) != 1:
        raise ValueError(f"{relative(template_path)} must contain exactly one {OCR_MARKER} marker")
    return template.replace(OCR_MARKER, format_ocr_lines(lines))


def render_candidate_prompt(lines: Sequence[Mapping[str, Any]]) -> str:
    return render_prompt(CANDIDATE_PROMPT_PATH, lines)


def render_audit_prompt(lines: Sequence[Mapping[str, Any]]) -> str:
    # Deliberately has no candidate argument: the audit pass is blind by construction.
    return render_prompt(AUDIT_PROMPT_PATH, lines)


@dataclass
class Backend:
    model: Any
    processor: Any
    tokenizer_data: Any
    torch: Any
    dtype: Any
    metadata: Dict[str, Any]


@dataclass(frozen=True)
class GenerationResult:
    raw_text: str
    elapsed_seconds: float
    prompt_token_count: int
    generated_token_ids: Tuple[int, ...]
    generated_token_ids_sha256: str
    eos_token_ids: Tuple[int, ...]
    token_count: int
    eos_observed: bool
    cap_hit: bool
    max_new_tokens: int
    serializer: Dict[str, Any]


def build_backend() -> Backend:
    try:
        import torch
        from lmformatenforcer.integrations.transformers import (
            build_token_enforcer_tokenizer_data,
        )
        from transformers import AutoModelForImageTextToText, AutoProcessor
    except ImportError as exc:
        raise SystemExit("V4 inference requires `uv sync --extra inference`.") from exc

    torch.manual_seed(RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_SEED)

    token = os.getenv("HF_TOKEN") or None
    dtype = torch.bfloat16
    processor = AutoProcessor.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        token=token,
    )
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        token=token,
        dtype=dtype,
        device_map="auto",
    )
    text_config = getattr(model.config, "text_config", None)
    vocab_size = getattr(text_config, "vocab_size", None)
    if not isinstance(vocab_size, int):
        raise RuntimeError("MedGemma config does not expose text_config.vocab_size")
    tokenizer_data = build_token_enforcer_tokenizer_data(
        processor.tokenizer,
        vocab_size=vocab_size,
    )
    resolved_revision = getattr(model.config, "_commit_hash", None)
    if resolved_revision != MODEL_REVISION:
        raise RuntimeError(
            f"Resolved model revision {resolved_revision!r} does not match pin {MODEL_REVISION!r}"
        )
    return Backend(
        model=model,
        processor=processor,
        tokenizer_data=tokenizer_data,
        torch=torch,
        dtype=dtype,
        metadata={
            "resolved_revision": resolved_revision,
            "device": str(model.device),
            "dtype": str(getattr(model, "dtype", "unknown")),
            "text_vocab_size": vocab_size,
            "cuda_version": torch.version.cuda,
            "cuda_device_name": (
                torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
            ),
            "accelerate_version": importlib.metadata.version("accelerate"),
            "torch_version": torch.__version__,
            "transformers_version": importlib.metadata.version("transformers"),
            "lm_format_enforcer_version": importlib.metadata.version("lm-format-enforcer"),
            "random_seed": RANDOM_SEED,
        },
    )


def verify_runtime_against_lock(
    protocol_lock: Mapping[str, Any],
    backend_metadata: Mapping[str, Any],
    ocr_metadata: Optional[Mapping[str, Any]] = None,
) -> None:
    runtime = protocol_lock.get("runtime")
    if not isinstance(runtime, Mapping):
        raise RuntimeError("Protocol lock has no runtime mapping")
    expected_backend = {
        key: runtime.get(key)
        for key in (
            "resolved_revision",
            "device",
            "text_vocab_size",
            "random_seed",
            "accelerate_version",
            "torch_version",
            "transformers_version",
            "lm_format_enforcer_version",
            "cuda_version",
            "cuda_device_name",
            "dtype",
        )
    }
    mismatches = [
        f"{key}: lock {value!r}, runtime {backend_metadata.get(key)!r}"
        for key, value in expected_backend.items()
        if backend_metadata.get(key) != value
    ]
    if ocr_metadata is not None:
        expected_ocr = {
            "pytesseract_version": runtime.get("pytesseract_version"),
            "tesseract_version": runtime.get("tesseract_version"),
        }
        mismatches.extend(
            f"{key}: lock {value!r}, runtime {ocr_metadata.get(key)!r}"
            for key, value in expected_ocr.items()
            if ocr_metadata.get(key) != value
        )
    current_host = {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
    }
    mismatches.extend(
        f"{key}: lock {runtime.get(key)!r}, runtime {value!r}"
        for key, value in current_host.items()
        if runtime.get(key) != value
    )
    if mismatches:
        detail = "\n".join(f"- {item}" for item in mismatches)
        raise RuntimeError(f"Frozen v4 runtime does not match the protocol lock:\n{detail}")


def build_effective_prefix_allowed_tokens_fn(
    tokenizer_data: Any,
    schema: Mapping[str, Any],
    *,
    force_json_field_order: bool,
    max_consecutive_whitespaces: int,
) -> Tuple[Any, Dict[str, Any]]:
    from lmformatenforcer import JsonSchemaParser
    from lmformatenforcer.integrations.transformers import (
        build_transformers_prefix_allowed_tokens_fn,
    )

    parser = JsonSchemaParser(dict(schema))
    prefix_allowed_tokens_fn = build_transformers_prefix_allowed_tokens_fn(
        tokenizer_data,
        parser,
    )
    # LM Format Enforcer's TokenEnforcer replaces the supplied root parser
    # configuration while adapting it to the tokenizer alphabet. Configure the
    # effective parser only after prefix construction, and preserve that alphabet.
    tokenizer_alphabet = tokenizer_data.tokenizer_alphabet
    if parser.config.alphabet != tokenizer_alphabet:
        raise RuntimeError("LMFE root parser did not retain the tokenizer-adjusted alphabet")
    parser.config.force_json_field_order = force_json_field_order
    parser.config.max_consecutive_whitespaces = max_consecutive_whitespaces
    if (
        parser.config.alphabet != tokenizer_alphabet
        or parser.config.force_json_field_order is not force_json_field_order
        or parser.config.max_consecutive_whitespaces != max_consecutive_whitespaces
    ):
        raise RuntimeError("LMFE effective serializer settings could not be established")
    serializer = {
        "configuration_stage": "after_transformers_prefix_construction",
        "tokenizer_alphabet_preserved": True,
        "force_json_field_order": parser.config.force_json_field_order,
        "max_consecutive_whitespaces": parser.config.max_consecutive_whitespaces,
        "max_json_array_length": parser.config.max_json_array_length,
    }
    return prefix_allowed_tokens_fn, serializer


def constrained_generate(
    backend: Backend,
    image_path: Path,
    prompt: str,
    schema: Mapping[str, Any],
    max_new_tokens: int,
    *,
    force_json_field_order: bool,
    max_consecutive_whitespaces: int,
) -> GenerationResult:
    prefix_allowed_tokens_fn, serializer = build_effective_prefix_allowed_tokens_fn(
        backend.tokenizer_data,
        schema,
        force_json_field_order=force_json_field_order,
        max_consecutive_whitespaces=max_consecutive_whitespaces,
    )
    with Image.open(image_path) as source:
        image = source.convert("RGB")
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    try:
        inputs = backend.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(backend.model.device, dtype=backend.dtype)
        input_length = inputs["input_ids"].shape[-1]
        started = time.perf_counter()
        with backend.torch.inference_mode():
            output = backend.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                num_beams=1,
                prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
            )
        elapsed = time.perf_counter() - started
    finally:
        image.close()
    generated = output[0][input_length:]
    generated_ids = [int(value) for value in generated.detach().cpu().tolist()]
    configured_eos = getattr(backend.model.generation_config, "eos_token_id", None)
    if configured_eos is None:
        eos_token_ids: set[int] = set()
    elif isinstance(configured_eos, int):
        eos_token_ids = {configured_eos}
    else:
        eos_token_ids = {int(value) for value in configured_eos}
    tokenizer_eos = getattr(backend.processor.tokenizer, "eos_token_id", None)
    if isinstance(tokenizer_eos, int):
        eos_token_ids.add(tokenizer_eos)
    eos_observed = bool(generated_ids and generated_ids[-1] in eos_token_ids)
    token_count = len(generated_ids)
    serialized_token_ids = json.dumps(
        generated_ids,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")
    raw_text = backend.processor.decode(generated, skip_special_tokens=True)
    return GenerationResult(
        raw_text=raw_text,
        elapsed_seconds=elapsed,
        prompt_token_count=int(input_length),
        generated_token_ids=tuple(generated_ids),
        generated_token_ids_sha256=sha256_bytes(serialized_token_ids),
        eos_token_ids=tuple(sorted(eos_token_ids)),
        token_count=token_count,
        eos_observed=eos_observed,
        cap_hit=token_count >= max_new_tokens,
        max_new_tokens=max_new_tokens,
        serializer=serializer,
    )


def write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")


def write_text_atomic(path: Path, value: str) -> None:
    """Durably replace one not-yet-published raw generation artifact."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    serialized = json.dumps(dict(value), indent=2, sort_keys=True, ensure_ascii=True) + "\n"
    write_text_atomic(path, serialized)


def generation_metadata(
    result: GenerationResult,
    *,
    pass_name: str,
) -> Dict[str, Any]:
    return {
        "pass": pass_name,
        "raw_text": text_metadata(result.raw_text),
        "elapsed_seconds": result.elapsed_seconds,
        "prompt_token_count": result.prompt_token_count,
        "generated_token_ids": list(result.generated_token_ids),
        "generated_token_ids_sha256": result.generated_token_ids_sha256,
        "eos_token_ids": list(result.eos_token_ids),
        "token_count": result.token_count,
        "eos_observed": result.eos_observed,
        "cap_hit": result.cap_hit,
        "max_new_tokens": result.max_new_tokens,
        "serializer": dict(result.serializer),
        "decoding": {
            "do_sample": False,
            "num_beams": 1,
            "skip_special_tokens": True,
        },
    }


def formal_attempt_id() -> str:
    return FORMAL_EXECUTION_ATTEMPT_ID


def write_exclusive_json(path: Path, payload: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(dict(payload), indent=2, sort_keys=True, ensure_ascii=True) + "\n"
    with path.open("x", encoding="utf-8") as handle:
        handle.write(serialized)
        handle.flush()
        os.fsync(handle.fileno())
    return path


def write_formal_attempt_event(
    attempt_id: str,
    event: str,
    payload: Mapping[str, Any],
) -> Path:
    if event not in {"started", "failed", "completed"}:
        raise ValueError(f"Unsupported formal attempt event: {event}")
    path = ROOT / "results" / "v4" / "formal" / "execution_attempts" / f"{attempt_id}.{event}.json"
    return write_exclusive_json(path, payload)


def claim_formal_attempt(payload: Mapping[str, Any]) -> Path:
    """Atomically consume the sole formal-attempt budget."""

    return write_formal_attempt_event(
        FORMAL_EXECUTION_ATTEMPT_ID,
        "started",
        payload,
    )


def claim_development_attempt(payload: Mapping[str, Any]) -> Path:
    """Atomically consume the sole v4 development-smoke budget."""

    path = (
        ROOT
        / "results"
        / "v4"
        / "development"
        / "iteration-1"
        / "execution_attempts"
        / f"{DEVELOPMENT_ATTEMPT_ID}.claimed.json"
    )
    return write_exclusive_json(path, payload)


def redact_error_message(error: BaseException) -> str:
    message = str(error)
    token = os.getenv("HF_TOKEN")
    if token:
        message = message.replace(token, "[REDACTED]")
    message = re.sub(r"(?i)(bearer\s+)[^\s,;]+", r"\1[REDACTED]", message)
    return re.sub(
        r"(?i)([?&](?:access_)?token=)[^&\s]+",
        r"\1[REDACTED]",
        message,
    )


def verify_formal_one_shot_safety(
    rows: Sequence[Mapping[str, str]],
    protocol_lock_sha256: str,
) -> None:
    """Require a pristine formal namespace before the sole permitted attempt."""

    if not re.fullmatch(r"[0-9a-f]{64}", protocol_lock_sha256):
        raise SystemExit("Formal v4 protocol-lock digest is invalid")
    formal_dir = ROOT / "results" / "v4" / "formal"
    existing_files = sorted(path for path in formal_dir.rglob("*") if path.is_file())
    row_artifacts = sorted(
        {path for row in rows for path in output_paths(row).values() if path.exists()}
    )
    if existing_files or row_artifacts:
        files = sorted({*existing_files, *row_artifacts})
        detail = "\n".join(f"- {relative(path)}" for path in files[:20])
        if len(files) > 20:
            detail += f"\n- ... and {len(files) - 20} more"
        raise SystemExit(
            "Formal v4 is one-shot and its namespace is not pristine; "
            "no model calls were made:\n"
            f"{detail}"
        )


def verify_development_one_shot_safety() -> None:
    development_dir = ROOT / "results" / "v4" / "development" / "iteration-1"
    existing_files = sorted(path for path in development_dir.rglob("*") if path.is_file())
    if existing_files:
        detail = "\n".join(f"- {relative(path)}" for path in existing_files[:20])
        if len(existing_files) > 20:
            detail += f"\n- ... and {len(existing_files) - 20} more"
        raise SystemExit(
            "The sole v4 development-smoke namespace is not pristine; "
            "no model calls were made:\n"
            f"{detail}"
        )


def artifact_metadata(path: Path) -> Dict[str, Any]:
    return {
        "path": relative(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def text_metadata(value: str) -> Dict[str, Any]:
    encoded = value.encode("utf-8")
    return {
        "bytes": len(encoded),
        "sha256": sha256_bytes(encoded),
    }


def verify_existing(
    paths: Mapping[str, Path],
    row: Mapping[str, str],
    development_iteration: Optional[int],
    protocol_lock: Optional[Mapping[str, Any]],
) -> None:
    if not paths["record"].is_file():
        raise SystemExit(
            f"Existing v4 artifacts lack a run record: {relative(paths['record'])}. "
            "The immutable partial row cannot be overwritten or resumed in place."
        )
    record = read_json(paths["record"])
    expected = {
        "attempt_status": "completed",
        "protocol_version": PROTOCOL_VERSION,
        "protocol_lock_path": (relative(PROTOCOL_LOCK_PATH) if protocol_lock is not None else None),
        "dataset_version": row["dataset_version"],
        "split": row["split"],
        "development_iteration": development_iteration,
        "document_id": row["document_id"],
        "condition": row["condition"],
        "model_id": MODEL_ID,
        "requested_revision": MODEL_REVISION,
        "image_sha256": row["image_sha256"],
        "candidate_prompt_template_sha256": sha256_file(CANDIDATE_PROMPT_PATH),
        "audit_prompt_template_sha256": sha256_file(AUDIT_PROMPT_PATH),
        "candidate_schema_sha256": sha256_file(CANDIDATE_SCHEMA_PATH),
        "audit_schema_sha256": sha256_file(AUDIT_SCHEMA_PATH),
        "canonical_schema_sha256": sha256_file(CANONICAL_SCHEMA_PATH),
        "compiler_sha256": sha256_file(COMPILER_PATH),
        "protocol_lock_sha256": (
            sha256_file(PROTOCOL_LOCK_PATH) if protocol_lock is not None else None
        ),
        "protocol_source_commit": (
            protocol_lock.get("protocol_source_commit") if protocol_lock is not None else None
        ),
    }
    mismatches = [
        f"{key}: expected {value!r}, found {record.get(key)!r}"
        for key, value in expected.items()
        if record.get(key) != value
    ]
    if protocol_lock is not None and not isinstance(record.get("execution_attempt_id"), str):
        mismatches.append("execution_attempt_id: missing from completed formal record")
    artifacts = record.get("artifacts", {})
    for name, path in paths.items():
        if name == "record":
            continue
        descriptor = artifacts.get(name)
        if not path.is_file():
            mismatches.append(f"{name}: missing {relative(path)}")
        elif not isinstance(descriptor, Mapping):
            mismatches.append(f"{name}: missing recorded artifact descriptor")
        elif descriptor.get("sha256") != sha256_file(path):
            mismatches.append(f"{name}: artifact hash differs from run record")
    if mismatches:
        detail = "\n".join(f"- {item}" for item in mismatches)
        raise SystemExit(
            f"Existing v4 output provenance does not match this protocol:\n{detail}\n"
            "The immutable row cannot be overwritten or resumed in place."
        )


def validate_manifest_image(row: Mapping[str, str]) -> Path:
    image_path = ROOT / row["image_path"]
    if not image_path.is_file():
        raise SystemExit(f"Missing manifest image: {relative(image_path)}")
    actual_sha = sha256_file(image_path)
    if actual_sha != row["image_sha256"]:
        raise SystemExit(
            f"Image hash mismatch for {row['document_id']} {row['condition']}: "
            f"manifest {row['image_sha256']}, actual {actual_sha}"
        )
    return image_path


def run_row(
    row: Mapping[str, str],
    backend: Backend,
    repository_sha: str,
    candidate_schema: Mapping[str, Any],
    audit_schema: Mapping[str, Any],
    canonical_schema: Mapping[str, Any],
    development_iteration: Optional[int],
    protocol_lock: Optional[Mapping[str, Any]],
    execution_attempt_id: Optional[str] = None,
    progress: Optional[Dict[str, Any]] = None,
) -> bool:
    if progress is None:
        progress = {}
    progress.update(
        {
            "stage": "preflight",
            "model_call_count": 0,
            "valid_model_response_count": 0,
            "resume_permitted": False,
        }
    )
    paths = output_paths(row, development_iteration)
    existing = [path for path in paths.values() if path.exists()]
    if existing:
        verify_existing(paths, row, development_iteration, protocol_lock)
        progress["stage"] = "verified_existing"
        print(f"SKIP existing {row['document_id']} {row['condition']}")
        return False

    progress["stage"] = "image_validation"
    image_path = validate_manifest_image(row)
    progress["stage"] = "ocr"
    progress["resume_permitted"] = True
    ocr_lines, raw_word_data, ocr_metadata = run_ocr(image_path)
    progress["stage"] = "ocr_runtime_validation"
    progress["resume_permitted"] = False
    if protocol_lock is not None:
        verify_runtime_against_lock(protocol_lock, backend.metadata, ocr_metadata)
    progress["stage"] = "prompt_rendering"
    progress["resume_permitted"] = False
    candidate_prompt = render_candidate_prompt(ocr_lines)
    audit_prompt = render_audit_prompt(ocr_lines)
    write_json(
        paths["ocr"],
        {
            "document_id": row["document_id"],
            "condition": row["condition"],
            **ocr_metadata,
            "raw_word_data": raw_word_data,
            "lines": ocr_lines,
        },
    )
    write_text(paths["candidate_prompt"], candidate_prompt)
    write_text(paths["audit_prompt"], audit_prompt)

    progress["stage"] = "candidate_generation"
    # Crossing the model-call boundary permanently taints this row, even if
    # generation itself raises before returning decodable text.
    progress["model_call_count"] = 1
    progress["resume_permitted"] = False
    candidate_result = constrained_generate(
        backend,
        image_path,
        candidate_prompt,
        candidate_schema,
        CANDIDATE_MAX_NEW_TOKENS,
        force_json_field_order=CANDIDATE_SERIALIZER_FORCE_JSON_FIELD_ORDER,
        max_consecutive_whitespaces=(CANDIDATE_SERIALIZER_MAX_CONSECUTIVE_WHITESPACES),
    )
    # Raw model text and its mechanical termination telemetry are durably
    # persisted before any JSON parser or schema validator can reject it.
    write_text_atomic(paths["candidate_raw"], candidate_result.raw_text)
    write_json_atomic(
        paths["candidate_generation"],
        generation_metadata(candidate_result, pass_name="candidate"),
    )
    progress["stage"] = "candidate_cap_validation"
    if candidate_result.cap_hit:
        raise RuntimeError("Candidate generation consumed its configured token maximum")
    progress["stage"] = "candidate_validation"
    candidate = parse_generated_json(candidate_result.raw_text, candidate_schema)
    progress["valid_model_response_count"] = 1

    write_json(paths["candidate_parsed"], candidate)

    # The audit prompt was finalized before candidate generation and receives no candidate data.
    progress["stage"] = "audit_generation"
    progress["model_call_count"] = 2
    progress["resume_permitted"] = False
    audit_result = constrained_generate(
        backend,
        image_path,
        audit_prompt,
        audit_schema,
        AUDIT_MAX_NEW_TOKENS,
        force_json_field_order=AUDIT_SERIALIZER_FORCE_JSON_FIELD_ORDER,
        max_consecutive_whitespaces=AUDIT_SERIALIZER_MAX_CONSECUTIVE_WHITESPACES,
    )
    write_text_atomic(paths["audit_raw"], audit_result.raw_text)
    write_json_atomic(
        paths["audit_generation"],
        generation_metadata(audit_result, pass_name="audit"),
    )
    progress["stage"] = "audit_cap_validation"
    if audit_result.cap_hit:
        raise RuntimeError("Audit generation consumed its configured token maximum")
    progress["stage"] = "audit_validation"
    audit = parse_generated_json(audit_result.raw_text, audit_schema)
    progress["valid_model_response_count"] = 2
    write_json(paths["audit_parsed"], audit)

    progress["stage"] = "compilation"
    prediction, compiler_audit = compile_prediction(candidate, audit, ocr_lines)
    if not isinstance(prediction, dict):
        raise TypeError("compile_prediction() must return a canonical prediction dict first")
    if not isinstance(compiler_audit, dict):
        raise TypeError("compile_prediction() must return a compiler audit dict second")

    canonical_errors = sorted(
        Draft202012Validator(canonical_schema).iter_errors(prediction),
        key=lambda error: list(error.absolute_path),
    )
    if canonical_errors:
        details = "; ".join(
            f"{'.'.join(str(part) for part in error.absolute_path) or '<root>'}: {error.message}"
            for error in canonical_errors
        )
        raise ValueError(f"Compiled prediction violates the frozen canonical schema: {details}")

    progress["stage"] = "final_artifact_write"
    write_json(paths["normalized"], prediction)
    write_json(paths["compiler_audit"], compiler_audit)

    artifacts = {name: artifact_metadata(path) for name, path in paths.items() if name != "record"}
    safe_manifest_row = {key: value for key, value in row.items() if key != "ground_truth_path"}
    write_json(
        paths["record"],
        {
            "attempt_status": "completed",
            "execution_attempt_id": execution_attempt_id,
            "run_timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "protocol_version": PROTOCOL_VERSION,
            "protocol_lock_path": (
                relative(PROTOCOL_LOCK_PATH) if protocol_lock is not None else None
            ),
            "protocol_lock_sha256": (
                sha256_file(PROTOCOL_LOCK_PATH) if protocol_lock is not None else None
            ),
            "protocol_source_commit": (
                protocol_lock.get("protocol_source_commit") if protocol_lock is not None else None
            ),
            "repository_commit": repository_sha,
            "dataset_version": row["dataset_version"],
            "generator_version": row["generator_version"],
            "split": row["split"],
            "development_iteration": development_iteration,
            "semantic_case_id": row["semantic_case_id"],
            "document_id": row["document_id"],
            "template_id": row["template_id"],
            "condition": row["condition"],
            "manifest_path": relative(V4_MANIFEST),
            "manifest_sha256": sha256_file(V4_MANIFEST),
            "manifest_row_without_ground_truth": safe_manifest_row,
            "model_id": MODEL_ID,
            "requested_revision": MODEL_REVISION,
            **backend.metadata,
            "backend": "transformers-direct",
            "requested_dtype": DTYPE_NAME,
            "do_sample": False,
            "num_beams": 1,
            "random_seed": RANDOM_SEED,
            "assistant_prefill": False,
            "candidate_max_new_tokens": CANDIDATE_MAX_NEW_TOKENS,
            "audit_max_new_tokens": AUDIT_MAX_NEW_TOKENS,
            "candidate_elapsed_seconds": candidate_result.elapsed_seconds,
            "audit_elapsed_seconds": audit_result.elapsed_seconds,
            "candidate_generation": generation_metadata(
                candidate_result,
                pass_name="candidate",
            ),
            "audit_generation": generation_metadata(
                audit_result,
                pass_name="audit",
            ),
            "image_path": row["image_path"],
            "image_sha256": row["image_sha256"],
            "candidate_prompt_template_path": relative(CANDIDATE_PROMPT_PATH),
            "candidate_prompt_template_sha256": sha256_file(CANDIDATE_PROMPT_PATH),
            "candidate_rendered_prompt": text_metadata(candidate_prompt),
            "audit_prompt_template_path": relative(AUDIT_PROMPT_PATH),
            "audit_prompt_template_sha256": sha256_file(AUDIT_PROMPT_PATH),
            "audit_rendered_prompt": text_metadata(audit_prompt),
            "audit_is_blind": True,
            "audit_prompt_constructed_before_candidate_generation": True,
            "candidate_schema_path": relative(CANDIDATE_SCHEMA_PATH),
            "candidate_schema_sha256": sha256_file(CANDIDATE_SCHEMA_PATH),
            "audit_schema_path": relative(AUDIT_SCHEMA_PATH),
            "audit_schema_sha256": sha256_file(AUDIT_SCHEMA_PATH),
            "canonical_schema_path": relative(CANONICAL_SCHEMA_PATH),
            "canonical_schema_sha256": sha256_file(CANONICAL_SCHEMA_PATH),
            "compiler_path": relative(COMPILER_PATH),
            "compiler_sha256": sha256_file(COMPILER_PATH),
            "inference_script_path": relative(SCRIPT_PATH),
            "inference_script_sha256": sha256_file(SCRIPT_PATH),
            "ocr": ocr_metadata,
            "compiler_summary": {
                "compiler_version": compiler_audit.get("compiler_version"),
                "accepted_non_null_count": compiler_audit.get("accepted_non_null_count"),
                "rejected_non_null_count": compiler_audit.get("rejected_non_null_count"),
                "history_evidence_rejection_count": compiler_audit.get(
                    "history_evidence_rejection_count"
                ),
                "schema_valid": compiler_audit.get("schema_valid"),
            },
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "artifacts": artifacts,
        },
    )
    print(
        f"WROTE {row['document_id']} {row['condition']} "
        f"(OCR {ocr_metadata['elapsed_seconds']:.1f}s, "
        f"candidate {candidate_result.elapsed_seconds:.1f}s, "
        f"audit {audit_result.elapsed_seconds:.1f}s)"
    )
    progress["stage"] = "completed"
    return True


def run(args: argparse.Namespace) -> None:
    if args.split == "development" and args.development_iteration != 1:
        raise SystemExit(
            "V4 permits one prespecified development smoke only: --development-iteration 1."
        )
    if args.split == "formal":
        if args.development_iteration is not None:
            raise SystemExit("--development-iteration is not valid for the formal split.")
        if args.limit is not None:
            raise SystemExit("Formal v4 forbids --limit; all 20 frozen inputs must run.")
        if args.condition != "all":
            raise SystemExit("Formal v4 requires --condition all in manifest order.")
        if args.overwrite:
            raise SystemExit("Formal v4 forbids overwriting any prior artifact.")
    elif args.overwrite:
        raise SystemExit(
            "V4 outputs are immutable and no second development iteration is permitted."
        )
    if args.split == "development":
        if args.limit is not None:
            raise SystemExit("V4 development smoke forbids --limit; all 4 inputs must run.")
        if args.condition != "all":
            raise SystemExit("V4 development smoke requires --condition all.")

    protocol_lock = verify_protocol_lock() if args.split == "formal" else None
    candidate_schema = read_json(CANDIDATE_SCHEMA_PATH)
    audit_schema = read_json(AUDIT_SCHEMA_PATH)
    canonical_schema = read_json(CANONICAL_SCHEMA_PATH)
    rows = load_manifest(args.split, args.condition)
    if args.split == "formal" and len(rows) != 20:
        raise SystemExit(f"Formal v4 requires exactly 20 manifest rows; found {len(rows)}")
    if args.split == "development" and args.condition == "all" and len(rows) != 4:
        raise SystemExit(
            f"V4 development smoke requires exactly 4 manifest rows; found {len(rows)}"
        )
    if args.limit is not None:
        if args.limit <= 0:
            raise SystemExit("--limit must be positive")
        rows = rows[: args.limit]
    if args.split == "development":
        verify_development_one_shot_safety()

    repository_sha = repository_commit()
    protocol_lock_sha256 = sha256_file(PROTOCOL_LOCK_PATH) if protocol_lock is not None else None
    execution_attempt_id = FORMAL_EXECUTION_ATTEMPT_ID if protocol_lock is not None else None
    attempt_context: Dict[str, Any] = {
        "execution_attempt_id": execution_attempt_id,
        "protocol_version": PROTOCOL_VERSION,
        "protocol_lock_path": (relative(PROTOCOL_LOCK_PATH) if protocol_lock is not None else None),
        "protocol_lock_sha256": protocol_lock_sha256,
        "protocol_source_commit": (
            protocol_lock.get("protocol_source_commit") if protocol_lock is not None else None
        ),
        "repository_commit": repository_sha,
        "manifest_path": relative(V4_MANIFEST),
        "manifest_sha256": sha256_file(V4_MANIFEST),
        "selected_row_count": len(rows),
        "condition": args.condition,
    }
    if protocol_lock is not None:
        verify_formal_one_shot_safety(rows, protocol_lock_sha256)
        claim_formal_attempt(
            {
                **attempt_context,
                "status": "started",
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            },
        )
    else:
        claim_development_attempt(
            {
                "execution_attempt_id": DEVELOPMENT_ATTEMPT_ID,
                "status": "claimed",
                "protocol_version": PROTOCOL_VERSION,
                "repository_commit": repository_sha,
                "manifest_path": relative(V4_MANIFEST),
                "manifest_sha256": sha256_file(V4_MANIFEST),
                "selected_row_count": len(rows),
                "condition": args.condition,
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            }
        )

    backend: Optional[Backend] = None
    completed = 0
    skipped = 0
    current_row: Optional[Mapping[str, str]] = None
    row_progress: Dict[str, Any] = {
        "stage": "backend_initialization",
        "model_call_count": 0,
        "valid_model_response_count": 0,
        "resume_permitted": False,
    }
    try:
        backend = build_backend()
        if protocol_lock is not None:
            verify_runtime_against_lock(protocol_lock, backend.metadata)
        for row in rows:
            current_row = row
            row_progress = {}
            wrote = run_row(
                row,
                backend,
                repository_sha,
                candidate_schema,
                audit_schema,
                canonical_schema,
                args.development_iteration,
                protocol_lock,
                execution_attempt_id,
                row_progress,
            )
            completed += int(wrote)
            skipped += int(not wrote)
        if protocol_lock is not None and (completed != 20 or skipped != 0):
            raise RuntimeError(
                "Formal v4 attempt did not produce exactly 20 new rows and zero existing rows"
            )
    except BaseException as exc:
        if protocol_lock is not None:
            write_formal_attempt_event(
                execution_attempt_id,
                "failed",
                {
                    **attempt_context,
                    "status": "failed",
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    "completed_new_row_count": completed,
                    "verified_existing_row_count": skipped,
                    "current_row": (
                        {
                            "document_id": current_row["document_id"],
                            "condition": current_row["condition"],
                        }
                        if current_row is not None
                        else None
                    ),
                    "failure_stage": row_progress.get("stage"),
                    "valid_model_response_count_for_current_row": row_progress.get(
                        "valid_model_response_count", 0
                    ),
                    "model_call_count_for_current_row": row_progress.get("model_call_count", 0),
                    "resume_permitted_for_current_row": False,
                    "error_type": type(exc).__name__,
                    "error_message": redact_error_message(exc),
                },
            )
        raise
    finally:
        if backend is not None:
            del backend
    if protocol_lock is not None:
        write_formal_attempt_event(
            execution_attempt_id,
            "completed",
            {
                **attempt_context,
                "status": "completed",
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "completed_new_row_count": completed,
                "verified_existing_row_count": skipped,
            },
        )
    print(f"Completed {completed} new v4 two-pass inference runs.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=SPLITS, required=True)
    parser.add_argument("--condition", choices=(*CONDITIONS, "all"), default="all")
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--development-iteration",
        type=int,
        choices=(1,),
        help="Required for the single immutable v4 development smoke.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
