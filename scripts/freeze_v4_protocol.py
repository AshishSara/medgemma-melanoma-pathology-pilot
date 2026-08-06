#!/usr/bin/env python3
"""Freeze the v4 formal protocol after a passing, provenance-valid development run."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Dict, List, Mapping, Sequence

from jsonschema import Draft202012Validator
from pilot_utils import ROOT, read_json, sha256_bytes, sha256_file
from run_v4_inference import (
    AUDIT_MAX_NEW_TOKENS,
    AUDIT_SERIALIZER_FORCE_JSON_FIELD_ORDER,
    AUDIT_SERIALIZER_MAX_CONSECUTIVE_WHITESPACES,
    CANDIDATE_MAX_NEW_TOKENS,
    CANDIDATE_SERIALIZER_FORCE_JSON_FIELD_ORDER,
    CANDIDATE_SERIALIZER_MAX_CONSECUTIVE_WHITESPACES,
    CONDITIONS,
    DEVELOPMENT_ATTEMPT_ID,
    DTYPE_NAME,
    FORMAL_EXECUTION_ATTEMPT_ID,
    MODEL_ID,
    MODEL_REVISION,
    MODEL_SLUG,
    OCR_MARKER,
    OCR_TIMEOUT_SECONDS,
    PROTOCOL_IDENTIFIER,
    PROTOCOL_LOCK_PATH,
    PROTOCOL_VERSION,
    RANDOM_SEED,
    REQUIRED_LOCKED_SOURCE_PATHS,
    TESSERACT_CONFIG,
    TESSERACT_LANGUAGE,
    format_ocr_lines,
    parse_generated_json,
)
from v3_pipeline import CLINICAL_FIELD_PATHS, compile_prediction

DEVELOPMENT_OUTPUT_COUNT = 4
FORMAL_OUTPUT_COUNT = 20
EXPECTED_MANIFEST_OUTPUT_COUNT = DEVELOPMENT_OUTPUT_COUNT + FORMAL_OUTPUT_COUNT
EXPECTED_DEVELOPMENT_CONDITION_COUNT = DEVELOPMENT_OUTPUT_COUNT // len(CONDITIONS)
DEVELOPMENT_CASE_IDS = ("MEL-104",)
FORMAL_CASE_IDS = ("MEL-105", "MEL-106", "MEL-107", "MEL-108", "MEL-109")
TEMPLATE_IDS = ("A", "B")
CASE_ORIGINS = {
    "MEL-104": ("v3_exposed_development", "v3_byte_identical_regeneration"),
    "MEL-105": ("v3_unexecuted_formal", "v3_byte_identical_regeneration"),
    "MEL-106": ("v3_unexecuted_formal", "v3_byte_identical_regeneration"),
    "MEL-107": ("v3_unexecuted_formal", "v3_byte_identical_regeneration"),
    "MEL-108": ("v3_unexecuted_formal", "v3_byte_identical_regeneration"),
    "MEL-109": ("new_v4_formal", "v4_new"),
}

PASS_CRITERIA = {
    "provenance_valid_count": 20,
    "canonical_schema_valid_rate": 1.0,
    "accepted_non_null_evidence_rate": 1.0,
    "pooled_field_exact_match_minimum": 0.9,
    "each_condition_field_exact_match_minimum": 0.85,
    "unsupported_field_rate_maximum": 0.02,
    "history_carryover_count_maximum": 0,
}

SCHEMA_ENFORCEMENT = (
    "lm-format-enforcer JsonSchemaParser via Transformers prefix_allowed_tokens_fn; "
    "candidate retains v3-effective tokenizer-adjusted defaults "
    "(force_json_field_order=false, max_consecutive_whitespaces=12); "
    "audit mutates the tokenizer-adjusted root parser after prefix construction "
    "(force_json_field_order=true, max_consecutive_whitespaces=0)"
)
FORMAL_EXECUTION_ORDER = "manifest order; candidate then blind audit per input"
INFRASTRUCTURE_RETRY_RULE = "one formal attempt only; no retry or resume after the attempt starts"

STABLE_BACKEND_RUNTIME_KEYS = (
    "resolved_revision",
    "device",
    "dtype",
    "text_vocab_size",
    "cuda_version",
    "cuda_device_name",
    "accelerate_version",
    "torch_version",
    "transformers_version",
    "lm_format_enforcer_version",
    "random_seed",
)

STABLE_OCR_RUNTIME_KEYS = (
    "engine",
    "tesseract_version",
    "pytesseract_version",
    "language",
    "config",
    "image_preprocessing",
    "output_type",
    "line_grouping",
    "line_id_assignment",
    "timeout_seconds",
)

REQUIRED_RUNTIME_KEYS = (
    "accelerate_version",
    "torch_version",
    "transformers_version",
    "lm_format_enforcer_version",
    "pytesseract_version",
    "tesseract_version",
    "cuda_version",
    "cuda_device_name",
    "dtype",
)


class ProtocolFreezeError(RuntimeError):
    """Raised before a protocol lock can be safely created."""


def _fail(message: str) -> None:
    raise ProtocolFreezeError(message)


def _relative(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError as exc:
        raise ProtocolFreezeError(f"Path escapes repository root: {path}") from exc


def _safe_repo_path(root: Path, raw_path: str) -> Path:
    if not isinstance(raw_path, str) or not raw_path:
        _fail("Manifest artifact path is missing")
    pure = PurePosixPath(raw_path)
    if pure.is_absolute() or ".." in pure.parts or str(pure) != raw_path:
        _fail(f"Manifest artifact path is not a canonical repository-relative path: {raw_path!r}")
    path = root.joinpath(*pure.parts)
    if path.is_symlink():
        _fail(f"Lock-bound artifacts may not be symlinks: {raw_path}")
    return path


def _read_object(path: Path, label: str) -> Dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ProtocolFreezeError(f"{label} is missing or invalid JSON: {path}") from exc
    if not isinstance(value, dict):
        _fail(f"{label} must be a JSON object: {path}")
    return value


def _development_paths(
    root: Path,
    row: Mapping[str, str],
    iteration: int,
) -> Dict[str, Path]:
    base = root / "results" / "v4" / "development" / f"iteration-{iteration}"
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
        "audit_raw": (base / "raw" / "audit" / MODEL_SLUG / condition / f"{document_id}.txt"),
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
        "record": (base / "run_records" / MODEL_SLUG / condition / f"{document_id}.json"),
    }


def load_and_validate_manifest(root: Path) -> List[Dict[str, str]]:
    manifest_path = root / "data" / "v4" / "report_manifest.csv"
    try:
        with manifest_path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
    except OSError as exc:
        raise ProtocolFreezeError(f"Cannot read v4 manifest: {manifest_path}") from exc
    return validate_manifest_rows(rows)


def validate_manifest_rows(rows: Sequence[Mapping[str, str]]) -> List[Dict[str, str]]:
    rows = [dict(row) for row in rows]
    if len(rows) != EXPECTED_MANIFEST_OUTPUT_COUNT:
        _fail(f"Expected {EXPECTED_MANIFEST_OUTPUT_COUNT} v4 manifest rows, found {len(rows)}")
    if any(None in row for row in rows):
        _fail("V4 manifest has malformed or extra columns")
    if any(row.get("dataset_version") != "pilot-v4" for row in rows):
        _fail("V4 manifest contains a non-v4 dataset row")

    keys = [(row.get("document_id"), row.get("condition")) for row in rows]
    if len(set(keys)) != len(keys):
        _fail("V4 manifest contains duplicate document/condition keys")
    development = [row for row in rows if row.get("split") == "development"]
    formal = [row for row in rows if row.get("split") == "formal"]
    if len(development) != DEVELOPMENT_OUTPUT_COUNT or len(formal) != FORMAL_OUTPUT_COUNT:
        _fail(
            "V4 manifest must contain exactly "
            f"{DEVELOPMENT_OUTPUT_COUNT} development and {FORMAL_OUTPUT_COUNT} formal rows"
        )
    if len(development) + len(formal) != len(rows):
        _fail("V4 manifest contains an unknown split")
    for split_name, split_rows, expected_per_condition in (
        ("development", development, EXPECTED_DEVELOPMENT_CONDITION_COUNT),
        ("formal", formal, FORMAL_OUTPUT_COUNT // len(CONDITIONS)),
    ):
        counts = {
            condition: sum(row.get("condition") == condition for row in split_rows)
            for condition in CONDITIONS
        }
        if counts != {condition: expected_per_condition for condition in CONDITIONS}:
            _fail(f"{split_name} manifest condition matrix is incomplete: {counts}")

    expected_matrix = {
        (split, case_id, f"{case_id}-{template_id}", template_id, condition)
        for split, case_ids in (
            ("development", DEVELOPMENT_CASE_IDS),
            ("formal", FORMAL_CASE_IDS),
        )
        for case_id in case_ids
        for template_id in TEMPLATE_IDS
        for condition in CONDITIONS
    }
    observed_matrix = {
        (
            row.get("split"),
            row.get("semantic_case_id"),
            row.get("document_id"),
            row.get("template_id"),
            row.get("condition"),
        )
        for row in rows
    }
    if observed_matrix != expected_matrix:
        _fail("V4 manifest does not match the frozen case/template/condition matrix")
    for row in rows:
        document_id = row["document_id"]
        condition = row["condition"]
        expected_case_origin, expected_artifact_origin = CASE_ORIGINS[row["semantic_case_id"]]
        if row.get("case_origin") != expected_case_origin:
            _fail(f"Manifest case_origin is invalid for {document_id}")
        if row.get("artifact_origin") != expected_artifact_origin:
            _fail(f"Manifest artifact_origin is invalid for {document_id}")
        expected_paths = {
            "source_text_path": f"data/v4/source_text/{document_id}.txt",
            "ground_truth_path": f"data/v4/ground_truth/{document_id}.json",
            "pdf_path": f"output/v4/pdf/{condition}/{document_id}.pdf",
            "image_path": f"output/v4/rendered/{condition}/{document_id}.png",
        }
        for key, expected_path in expected_paths.items():
            if row.get(key) != expected_path:
                _fail(f"Manifest {key} is not canonically bound for {document_id} {condition}")
    return rows


def collect_corpus_artifacts(
    root: Path,
    rows: Sequence[Mapping[str, str]],
) -> Dict[str, str]:
    """Hash the closed corpus inventory without deserializing any ground truth."""

    expected = {
        "data/v4/cases.csv",
        "data/v4/report_manifest.csv",
    }
    for row in rows:
        for key in (
            "source_text_path",
            "ground_truth_path",
            "pdf_path",
            "image_path",
        ):
            raw_path = row.get(key)
            if not isinstance(raw_path, str):
                _fail(f"Manifest row {row.get('document_id')} lacks {key}")
            expected.add(raw_path)

    discovered = {
        "data/v4/cases.csv",
        "data/v4/report_manifest.csv",
    }
    corpus_roots = (
        root / "data" / "v4" / "source_text",
        root / "data" / "v4" / "ground_truth",
        root / "output" / "v4" / "pdf",
        root / "output" / "v4" / "rendered",
    )
    for directory in corpus_roots:
        if not directory.is_dir() or directory.is_symlink():
            _fail(f"Missing or unsafe corpus directory: {_relative(directory, root)}")
        for path in directory.rglob("*"):
            if path.is_symlink():
                _fail(f"Corpus inventory contains a symlink: {_relative(path, root)}")
            if path.is_file():
                discovered.add(_relative(path, root))
    if discovered != expected:
        missing = sorted(expected - discovered)
        extra = sorted(discovered - expected)
        _fail(f"Corpus inventory differs from manifest; missing={missing}, extra={extra}")

    hashes: Dict[str, str] = {}
    for relative_path in sorted(expected):
        path = _safe_repo_path(root, relative_path)
        if not path.is_file():
            _fail(f"Missing corpus artifact: {relative_path}")
        hashes[relative_path] = sha256_file(path)

    for row in rows:
        for path_key, hash_key in (("pdf_path", "pdf_sha256"), ("image_path", "image_sha256")):
            relative_path = row[path_key]
            recorded_hash = row.get(hash_key)
            if not isinstance(recorded_hash, str) or not re.fullmatch(
                r"[0-9a-f]{64}", recorded_hash
            ):
                _fail(f"Manifest {hash_key} is invalid for {row['document_id']}")
            if hashes[relative_path] != recorded_hash:
                _fail(
                    f"Manifest {hash_key} does not match {relative_path} "
                    f"for {row['document_id']} {row['condition']}"
                )
    return hashes


def partition_corpus_artifacts(
    rows: Sequence[Mapping[str, str]],
    all_artifacts: Mapping[str, str],
) -> Dict[str, Dict[str, str]]:
    operational_paths = {"data/v4/report_manifest.csv"}
    evaluation_only_paths = {"data/v4/cases.csv"}
    for row in rows:
        operational_paths.update((row["pdf_path"], row["image_path"]))
        evaluation_only_paths.update((row["source_text_path"], row["ground_truth_path"]))
    if operational_paths & evaluation_only_paths:
        _fail("Operational and evaluation-only corpus inventories overlap")
    if operational_paths | evaluation_only_paths != set(all_artifacts):
        _fail("Corpus partition does not cover the complete frozen inventory")
    return {
        "operational": {path: all_artifacts[path] for path in sorted(operational_paths)},
        "evaluation_only": {path: all_artifacts[path] for path in sorted(evaluation_only_paths)},
    }


def collect_v3_reuse_provenance(
    root: Path,
    rows: Sequence[Mapping[str, str]],
) -> tuple[Dict[str, Any], List[str]]:
    """Bind v4 reuse to byte-identical v3 artifacts and the preserved v3 boundary."""

    comparisons: Dict[str, Dict[str, str]] = {}
    bound_paths: set[str] = {"data/v3/protocol_lock.json"}
    seen: set[tuple[str, str]] = set()
    for row in rows:
        case_id = row["semantic_case_id"]
        if case_id not in {"MEL-104", "MEL-105", "MEL-106", "MEL-107", "MEL-108"}:
            continue
        key = (row["document_id"], row["condition"])
        if key in seen:
            continue
        seen.add(key)
        for field in ("source_text_path", "ground_truth_path", "pdf_path", "image_path"):
            v4_relative = row[field]
            v3_relative = v4_relative.replace("/v4/", "/v3/", 1)
            v4_path = _safe_repo_path(root, v4_relative)
            v3_path = _safe_repo_path(root, v3_relative)
            if not v3_path.is_file() or not v4_path.is_file():
                _fail(f"Missing reused artifact comparison: {v3_relative} -> {v4_relative}")
            v3_sha = sha256_file(v3_path)
            v4_sha = sha256_file(v4_path)
            if v3_sha != v4_sha or v3_path.read_bytes() != v4_path.read_bytes():
                _fail(f"V4 reused artifact is not byte-identical to v3: {v4_relative}")
            comparisons[v4_relative] = {
                "v3_path": v3_relative,
                "v3_sha256": v3_sha,
                "v4_sha256": v4_sha,
            }
            bound_paths.update((v3_relative, v4_relative))

    v3_formal = root / "results" / "v3" / "formal"
    if not v3_formal.is_dir() or v3_formal.is_symlink():
        _fail("Preserved v3 formal result directory is missing or unsafe")
    case_artifacts: Dict[str, str] = {}
    observed_case_ids: set[str] = set()
    for path in sorted(v3_formal.rglob("*")):
        if path.is_symlink():
            _fail(f"V3 formal result inventory contains a symlink: {_relative(path, root)}")
        if not path.is_file():
            continue
        match = re.search(r"(MEL-[0-9]{3})", path.name)
        if match is None:
            continue
        case_id = match.group(1)
        observed_case_ids.add(case_id)
        relative_path = _relative(path, root)
        case_artifacts[relative_path] = sha256_file(path)
        bound_paths.add(relative_path)
    if observed_case_ids != {"MEL-104"}:
        _fail(
            "V3 formal model-artifact inventory must contain only exposed MEL-104; "
            f"found={sorted(observed_case_ids)}"
        )

    attempts_dir = v3_formal / "execution_attempts"
    started = sorted(attempts_dir.glob("*.started.json"))
    failed = sorted(attempts_dir.glob("*.failed.json"))
    completed = sorted(attempts_dir.glob("*.completed.json"))
    if len(started) != 1 or len(failed) != 1 or completed:
        _fail("Preserved v3 formal attempt ledger does not contain one failed attempt")
    failed_event = _read_object(failed[0], "v3 failed formal attempt")
    if (
        failed_event.get("current_row") != {"document_id": "MEL-104-B", "condition": "ocr_degraded"}
        or failed_event.get("failure_stage") != "audit_validation"
        or failed_event.get("valid_model_response_count_for_current_row") != 1
        or failed_event.get("resume_permitted_for_current_row") is not False
        or failed_event.get("completed_new_row_count") != 3
    ):
        _fail("Preserved v3 formal failure event does not match the disclosed boundary")
    attempt_hashes = {}
    for path in (*started, *failed):
        relative_path = _relative(path, root)
        attempt_hashes[relative_path] = sha256_file(path)
        bound_paths.add(relative_path)

    v3_lock_path = root / "data" / "v3" / "protocol_lock.json"
    if not v3_lock_path.is_file() or v3_lock_path.is_symlink():
        _fail("Preserved v3 protocol lock is missing or unsafe")
    return (
        {
            "reuse_disposition": {
                "MEL-104": "development_only_exposed_in_v3",
                "MEL-105": "formal_reuse_no_v3_model_call",
                "MEL-106": "formal_reuse_no_v3_model_call",
                "MEL-107": "formal_reuse_no_v3_model_call",
                "MEL-108": "formal_reuse_no_v3_model_call",
                "MEL-109": "new_v4_formal",
            },
            "byte_identical_reused_artifacts": dict(sorted(comparisons.items())),
            "v3_protocol_lock": {
                "path": "data/v3/protocol_lock.json",
                "sha256": sha256_file(v3_lock_path),
            },
            "v3_formal_attempt_events": dict(sorted(attempt_hashes.items())),
            "v3_formal_model_artifacts": dict(sorted(case_artifacts.items())),
            "v3_formal_model_artifact_case_ids": sorted(observed_case_ids),
            "v3_unexecuted_reused_formal_case_ids": [
                "MEL-105",
                "MEL-106",
                "MEL-107",
                "MEL-108",
            ],
        },
        sorted(bound_paths),
    )


def collect_locked_source_hashes(root: Path) -> Dict[str, str]:
    hashes: Dict[str, str] = {}
    for relative_path in REQUIRED_LOCKED_SOURCE_PATHS:
        path = _safe_repo_path(root, relative_path)
        if not path.is_file():
            _fail(f"Missing required locked source: {relative_path}")
        hashes[relative_path] = sha256_file(path)
    return hashes


def _git(root: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
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
            f"Could not read {relative_path} from execution commit {commit}"
        ) from exc


def validate_protocol_source_commit(
    root: Path,
    protocol_source_commit: str,
    bound_paths: Sequence[str],
) -> None:
    if not re.fullmatch(r"[0-9a-f]{40}", protocol_source_commit):
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
        _fail(
            "--protocol-source-commit must equal current HEAD before the lock is created; "
            "the subsequent lock commit will make it an ancestor of formal execution"
        )

    tracked_status = _git(
        root,
        "status",
        "--porcelain",
        "--untracked-files=no",
    ).stdout.strip()
    if tracked_status:
        _fail("Tracked worktree/index changes must be committed before freezing")

    for relative_path in sorted(set(bound_paths)):
        result = _git(
            root,
            "cat-file",
            "-e",
            f"{protocol_source_commit}:{relative_path}",
            check=False,
        )
        if result.returncode != 0:
            _fail(f"Lock-bound artifact is not present in protocol source commit: {relative_path}")
        committed_bytes = _committed_blob(root, protocol_source_commit, relative_path)
        if sha256_bytes(committed_bytes) != sha256_file(root / relative_path):
            _fail(f"Working artifact bytes differ from protocol source commit: {relative_path}")


def validate_post_development_immutability(
    root: Path,
    development_commit: str,
    immutable_paths: Sequence[str],
) -> Dict[str, Any]:
    """Require every pre-smoke source, corpus, and reuse artifact to remain identical."""

    if not immutable_paths:
        _fail("Post-development immutable path inventory is empty")
    hashes: Dict[str, str] = {}
    for relative_path in sorted(set(immutable_paths)):
        current_path = _safe_repo_path(root, relative_path)
        if not current_path.is_file():
            _fail(f"Post-development immutable artifact is missing: {relative_path}")
        development_bytes = _committed_blob(root, development_commit, relative_path)
        current_bytes = current_path.read_bytes()
        if development_bytes != current_bytes:
            _fail(f"Locked source changed after development execution: {relative_path}")
        hashes[relative_path] = sha256_bytes(current_bytes)
    return {
        "classification": "byte_identical_sources_corpus_and_v3_provenance",
        "development_commit": development_commit,
        "immutable_path_sha256": dict(sorted(hashes.items())),
    }


def validate_post_development_source_hardening(
    root: Path,
    development_commit: str,
) -> Dict[str, Any]:
    """Compatibility wrapper for source-only callers and regression tests."""

    result = validate_post_development_immutability(
        root,
        development_commit,
        REQUIRED_LOCKED_SOURCE_PATHS,
    )
    return {
        "classification": "byte_identical_locked_sources",
        "development_commit": development_commit,
        "locked_source_sha256": result["immutable_path_sha256"],
    }


def _render_prompt(template_path: Path, lines: Sequence[Mapping[str, Any]]) -> str:
    template = template_path.read_text(encoding="utf-8")
    if template.count(OCR_MARKER) != 1:
        _fail(f"{template_path} must contain exactly one {OCR_MARKER}")
    return template.replace(OCR_MARKER, format_ocr_lines(lines))


def _expected_run_record_values(
    root: Path,
    row: Mapping[str, str],
    iteration: int,
    execution_inference_sha256: str,
) -> Dict[str, Any]:
    safe_manifest_row = {key: value for key, value in row.items() if key != "ground_truth_path"}
    return {
        "attempt_status": "completed",
        "execution_attempt_id": None,
        "protocol_version": PROTOCOL_VERSION,
        "protocol_lock_path": None,
        "protocol_lock_sha256": None,
        "protocol_source_commit": None,
        "dataset_version": "pilot-v4",
        "generator_version": row["generator_version"],
        "split": "development",
        "development_iteration": iteration,
        "semantic_case_id": row["semantic_case_id"],
        "document_id": row["document_id"],
        "template_id": row["template_id"],
        "condition": row["condition"],
        "manifest_path": "data/v4/report_manifest.csv",
        "manifest_sha256": sha256_file(root / "data" / "v4" / "report_manifest.csv"),
        "manifest_row_without_ground_truth": safe_manifest_row,
        "model_id": MODEL_ID,
        "requested_revision": MODEL_REVISION,
        "resolved_revision": MODEL_REVISION,
        "backend": "transformers-direct",
        "requested_dtype": DTYPE_NAME,
        "do_sample": False,
        "num_beams": 1,
        "random_seed": RANDOM_SEED,
        "assistant_prefill": False,
        "candidate_max_new_tokens": CANDIDATE_MAX_NEW_TOKENS,
        "audit_max_new_tokens": AUDIT_MAX_NEW_TOKENS,
        "image_path": row["image_path"],
        "image_sha256": row["image_sha256"],
        "candidate_prompt_template_path": "prompts/v3/candidate_prompt.txt",
        "candidate_prompt_template_sha256": sha256_file(
            root / "prompts" / "v3" / "candidate_prompt.txt"
        ),
        "audit_prompt_template_path": "prompts/v3/audit_prompt.txt",
        "audit_prompt_template_sha256": sha256_file(root / "prompts" / "v3" / "audit_prompt.txt"),
        "audit_is_blind": True,
        "audit_prompt_constructed_before_candidate_generation": True,
        "candidate_schema_path": "schema/v3/candidate.schema.json",
        "candidate_schema_sha256": sha256_file(root / "schema" / "v3" / "candidate.schema.json"),
        "audit_schema_path": "schema/v3/audit.schema.json",
        "audit_schema_sha256": sha256_file(root / "schema" / "v3" / "audit.schema.json"),
        "canonical_schema_path": "schema/extraction.schema.json",
        "canonical_schema_sha256": sha256_file(root / "schema" / "extraction.schema.json"),
        "compiler_path": "scripts/v3_pipeline.py",
        "compiler_sha256": sha256_file(root / "scripts" / "v3_pipeline.py"),
        "inference_script_path": "scripts/run_v4_inference.py",
        "inference_script_sha256": execution_inference_sha256,
    }


def _validate_artifact_descriptor(
    root: Path,
    name: str,
    path: Path,
    descriptor: Any,
) -> None:
    if not path.is_file() or path.is_symlink():
        _fail(f"Missing or unsafe development artifact: {_relative(path, root)}")
    if not isinstance(descriptor, Mapping):
        _fail(f"Run record lacks descriptor for {name}: {_relative(path, root)}")
    expected = {
        "path": _relative(path, root),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }
    mismatches = [
        f"{key}: expected {value!r}, found {descriptor.get(key)!r}"
        for key, value in expected.items()
        if descriptor.get(key) != value
    ]
    if mismatches:
        _fail(f"Invalid {name} descriptor in {_relative(path, root)}: {mismatches}")


def _runtime_fingerprint(record: Mapping[str, Any]) -> Dict[str, Any]:
    ocr = record.get("ocr")
    if not isinstance(ocr, Mapping):
        _fail("Development run record has no OCR runtime mapping")
    fingerprint = {key: record.get(key) for key in STABLE_BACKEND_RUNTIME_KEYS}
    fingerprint.update({f"ocr.{key}": ocr.get(key) for key in STABLE_OCR_RUNTIME_KEYS})
    fingerprint.update(
        {
            "backend": record.get("backend"),
            "requested_dtype": record.get("requested_dtype"),
            "platform": record.get("platform"),
            "python_version": record.get("python_version"),
        }
    )
    if fingerprint["resolved_revision"] != MODEL_REVISION:
        _fail("Development runtime did not resolve the frozen model revision")
    if fingerprint["random_seed"] != RANDOM_SEED:
        _fail("Development runtime random seed differs from the protocol")
    expected_ocr = {
        "ocr.engine": "tesseract",
        "ocr.language": TESSERACT_LANGUAGE,
        "ocr.config": TESSERACT_CONFIG,
        "ocr.image_preprocessing": "none",
        "ocr.output_type": "pytesseract.Output.DICT",
        "ocr.line_grouping": "ordered page_num/block_num/par_num/line_num word groups",
        "ocr.line_id_assignment": "L001..Lnnn in first-seen grouped OCR order",
        "ocr.timeout_seconds": OCR_TIMEOUT_SECONDS,
    }
    for key, value in expected_ocr.items():
        if fingerprint.get(key) != value:
            _fail(f"Development runtime {key} differs from the protocol")
    if fingerprint["dtype"] != "torch.bfloat16":
        _fail("Development backend dtype is not torch.bfloat16")
    if not isinstance(fingerprint["device"], str) or not fingerprint["device"]:
        _fail("Development backend device is missing")
    if (
        not isinstance(fingerprint["text_vocab_size"], int)
        or isinstance(fingerprint["text_vocab_size"], bool)
        or fingerprint["text_vocab_size"] <= 0
    ):
        _fail("Development text vocabulary size is invalid")
    for key in ("platform", "python_version"):
        if not isinstance(fingerprint[key], str) or not fingerprint[key]:
            _fail(f"Development runtime {key} is missing")
    return fingerprint


def common_runtime(records: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    if len(records) != DEVELOPMENT_OUTPUT_COUNT:
        _fail(f"Expected {DEVELOPMENT_OUTPUT_COUNT} development run records, found {len(records)}")
    fingerprints = [_runtime_fingerprint(record) for record in records]
    first = fingerprints[0]
    for index, fingerprint in enumerate(fingerprints[1:], start=2):
        if fingerprint != first:
            differing = sorted(
                key
                for key in set(first) | set(fingerprint)
                if first.get(key) != fingerprint.get(key)
            )
            _fail(f"Development runtime metadata differs in row {index}: {differing}")

    ocr = records[0]["ocr"]
    runtime: Dict[str, Any] = {
        "resolved_revision": records[0].get("resolved_revision"),
        "device": records[0].get("device"),
        "text_vocab_size": records[0].get("text_vocab_size"),
        "random_seed": records[0].get("random_seed"),
        "accelerate_version": records[0].get("accelerate_version"),
        "torch_version": records[0].get("torch_version"),
        "transformers_version": records[0].get("transformers_version"),
        "lm_format_enforcer_version": records[0].get("lm_format_enforcer_version"),
        "pytesseract_version": ocr.get("pytesseract_version"),
        "tesseract_version": ocr.get("tesseract_version"),
        "cuda_version": records[0].get("cuda_version"),
        "cuda_device_name": records[0].get("cuda_device_name"),
        "dtype": records[0].get("dtype"),
        "python_version": records[0].get("python_version"),
        "platform": records[0].get("platform"),
    }
    for key in REQUIRED_RUNTIME_KEYS:
        value = runtime.get(key)
        if not isinstance(value, str) or not value:
            _fail(f"Development runtime field {key} is missing or empty")
    if runtime["resolved_revision"] != MODEL_REVISION:
        _fail("Development runtime resolved revision differs from the model pin")
    if (
        not isinstance(runtime["text_vocab_size"], int)
        or isinstance(runtime["text_vocab_size"], bool)
        or runtime["text_vocab_size"] <= 0
    ):
        _fail("Development runtime text vocabulary size is invalid")
    if runtime["random_seed"] != RANDOM_SEED:
        _fail("Development runtime random seed differs from the protocol")
    return dict(runtime)


def _evidence_gate_summary(compiler_logs: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    from evaluate_v4 import _accepted_decision_has_valid_evidence

    accepted = valid = history_carryover = 0
    for log in compiler_logs:
        if log.get("schema_valid") is not True or log.get("schema_errors") != []:
            _fail("Development compiler audit is not schema-valid")
        fields = log.get("fields")
        if not isinstance(fields, Mapping):
            _fail("Development compiler audit has no field decision mapping")
        for path, raw_decision in fields.items():
            if not isinstance(raw_decision, Mapping):
                _fail(f"Compiler decision is not a mapping: {path}")
            is_accepted_non_null = (
                raw_decision.get("accepted") is True and raw_decision.get("final_value") is not None
            )
            if (
                is_accepted_non_null
                and isinstance(raw_decision.get("evidence_sections"), list)
                and "history" in raw_decision["evidence_sections"]
            ):
                history_carryover += 1
            if path not in CLINICAL_FIELD_PATHS or not is_accepted_non_null:
                continue
            accepted += 1
            if _accepted_decision_has_valid_evidence(path, raw_decision):
                valid += 1
    return {
        "accepted_non_null_count": accepted,
        "accepted_with_valid_non_history_evidence_count": valid,
        "accepted_non_null_evidence_rate": 1.0 if accepted == 0 else valid / accepted,
        "history_carryover_count": history_carryover,
    }


def require_development_gate(
    final_pipeline: Mapping[str, Any],
    evidence: Mapping[str, Any],
    generation_telemetry: Mapping[str, Any],
) -> Dict[str, Any]:
    overall = final_pipeline.get("overall")
    conditions = final_pipeline.get("by_condition")
    if not isinstance(overall, Mapping) or not isinstance(conditions, list):
        _fail("Development metrics lack final overall or condition results")
    by_condition = {item.get("condition"): item for item in conditions if isinstance(item, Mapping)}
    if set(by_condition) != set(CONDITIONS):
        _fail("Development metrics do not contain exactly the two frozen conditions")

    numeric_fields = (
        "schema_valid_rate",
        "field_exact_match",
        "unsupported_field_rate",
    )
    for label, aggregate in [
        ("overall", overall),
        *[(condition, by_condition[condition]) for condition in CONDITIONS],
    ]:
        for key in numeric_fields:
            value = aggregate.get(key)
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                _fail(f"Development {label} metric {key} is not numeric")
            if not math.isfinite(float(value)):
                _fail(f"Development {label} metric {key} is non-finite")

    if overall.get("output_count") != DEVELOPMENT_OUTPUT_COUNT:
        _fail("Development overall output count is not 4")
    if overall.get("schema_valid_count") != DEVELOPMENT_OUTPUT_COUNT:
        _fail("Development canonical schema count is not 4/4")
    for condition in CONDITIONS:
        if by_condition[condition].get("output_count") != EXPECTED_DEVELOPMENT_CONDITION_COUNT:
            _fail(f"Development condition {condition} does not contain two outputs")
    if set(generation_telemetry) != {"candidate", "audit"}:
        _fail("Development generation telemetry lacks candidate or audit pass")
    cap_hits = {}
    for pass_name in ("candidate", "audit"):
        pass_telemetry = generation_telemetry.get(pass_name)
        if not isinstance(pass_telemetry, Mapping):
            _fail(f"Development {pass_name} generation telemetry is invalid")
        cap_hit_count = pass_telemetry.get("cap_hit_count")
        if (
            not isinstance(cap_hit_count, int)
            or isinstance(cap_hit_count, bool)
            or cap_hit_count < 0
        ):
            _fail(f"Development {pass_name} cap-hit count is invalid")
        cap_hits[pass_name] = cap_hit_count

    criteria = {
        "provenance_4_of_4": True,
        "candidate_strict_json_schema_4_of_4": True,
        "audit_strict_json_schema_4_of_4": True,
        "canonical_schema_100_percent": overall["schema_valid_rate"] == 1.0,
        "evidence_acceptance_100_percent": (evidence.get("accepted_non_null_evidence_rate") == 1.0),
        "zero_candidate_cap_hits": cap_hits["candidate"] == 0,
        "zero_audit_cap_hits": cap_hits["audit"] == 0,
        "zero_unsupported_fields": overall.get("unsupported_count") == 0,
        "zero_history_carryover": (
            evidence.get("history_carryover_count")
            <= PASS_CRITERIA["history_carryover_count_maximum"]
        ),
    }
    if not all(criteria.values()):
        failed = sorted(key for key, passed in criteria.items() if not passed)
        _fail(f"Development iteration does not meet the prespecified formal gate: {failed}")
    return {
        "status": "PASS",
        "criteria": criteria,
        "provenance_valid_count": DEVELOPMENT_OUTPUT_COUNT,
        "canonical_schema_valid_rate": overall["schema_valid_rate"],
        "pooled_field_exact_match": overall["field_exact_match"],
        "condition_field_exact_match": {
            condition: by_condition[condition]["field_exact_match"] for condition in CONDITIONS
        },
        "unsupported_field_rate": overall["unsupported_field_rate"],
        "unsupported_field_count": overall.get("unsupported_count"),
        "generation_cap_hit_count": cap_hits,
        "accuracy_selection_note": (
            "Exact-match accuracy is descriptive only and is not a development selection threshold."
        ),
        **dict(evidence),
    }


def _recomputed_metrics(
    root: Path,
    preflight: Sequence[Mapping[str, Any]],
    development_rows: Sequence[Mapping[str, str]],
) -> Dict[str, Any]:
    # This is the only semantic ground-truth access in the freezer. The rows
    # have already been restricted to the declared development split.
    if any(row.get("split") != "development" for row in development_rows):
        _fail("Refusing metric recomputation outside the development split")
    from evaluate_v4 import (
        _condition_metrics,
        _paired_clean_to_degraded,
        _per_field_metrics,
        _score_layer,
        _template_metrics,
    )

    truth_by_document = {
        row["document_id"]: read_json(root / row["ground_truth_path"]) for row in development_rows
    }
    canonical_schema = read_json(root / "schema" / "extraction.schema.json")
    candidate_overall, candidate_rows = _score_layer(
        preflight,
        truth_by_document,
        "candidate",
        canonical_schema,
    )
    final_overall, final_rows = _score_layer(
        preflight,
        truth_by_document,
        "normalized",
        canonical_schema,
    )

    def layer(overall: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
        return {
            "overall": dict(overall),
            "by_condition": _condition_metrics(rows),
            "by_template": _template_metrics(rows),
            "per_field": _per_field_metrics(rows),
            "paired_clean_to_degraded": _paired_clean_to_degraded(rows),
        }

    return {
        "candidate": layer(candidate_overall, candidate_rows),
        "final_pipeline": layer(final_overall, final_rows),
    }


def validate_complete_development_iteration(
    root: Path,
    rows: Sequence[Mapping[str, str]],
    iteration: int,
    post_development_immutable_paths: Sequence[str],
) -> Dict[str, Any]:
    development_rows = [dict(row) for row in rows if row.get("split") == "development"]
    if len(development_rows) != DEVELOPMENT_OUTPUT_COUNT:
        _fail(f"Expected {DEVELOPMENT_OUTPUT_COUNT} development rows")

    candidate_schema = read_json(root / "schema" / "v3" / "candidate.schema.json")
    audit_schema = read_json(root / "schema" / "v3" / "audit.schema.json")
    canonical_schema = read_json(root / "schema" / "extraction.schema.json")
    canonical_validator = Draft202012Validator(canonical_schema)
    candidate_template = root / "prompts" / "v3" / "candidate_prompt.txt"
    audit_template = root / "prompts" / "v3" / "audit_prompt.txt"
    records: List[Dict[str, Any]] = []
    compiler_logs: List[Dict[str, Any]] = []
    preflight: List[Dict[str, Any]] = []

    for row in development_rows:
        paths = _development_paths(root, row, iteration)
        if any(not path.is_file() or path.is_symlink() for path in paths.values()):
            missing = [
                _relative(path, root)
                for path in paths.values()
                if not path.is_file() or path.is_symlink()
            ]
            _fail(f"Development row is incomplete or unsafe: {missing}")
        record = _read_object(paths["record"], "development run record")
        repository_commit = record.get("repository_commit")
        if not isinstance(repository_commit, str) or not re.fullmatch(
            r"[0-9a-f]{40}", repository_commit
        ):
            _fail("Development repository_commit is not a full lowercase SHA")
        execution_inference_sha256 = sha256_bytes(
            _committed_blob(
                root,
                repository_commit,
                "scripts/run_v4_inference.py",
            )
        )
        expected_record = _expected_run_record_values(
            root,
            row,
            iteration,
            execution_inference_sha256,
        )
        mismatches = [
            f"{key}: expected {value!r}, found {record.get(key)!r}"
            for key, value in expected_record.items()
            if record.get(key) != value
        ]
        if mismatches:
            _fail(
                f"Development run record does not match {row['document_id']} "
                f"{row['condition']}: {mismatches}"
            )

        descriptors = record.get("artifacts")
        if not isinstance(descriptors, Mapping):
            _fail("Development run record has no artifact mapping")
        for name, path in paths.items():
            if name != "record":
                _validate_artifact_descriptor(root, name, path, descriptors.get(name))

        candidate_raw = paths["candidate_raw"].read_text(encoding="utf-8")
        audit_raw = paths["audit_raw"].read_text(encoding="utf-8")
        from evaluate_v4 import _generation_telemetry_issues

        for pass_name, raw_text in (
            ("candidate", candidate_raw),
            ("audit", audit_raw),
        ):
            telemetry = _read_object(
                paths[f"{pass_name}_generation"],
                f"{pass_name} generation telemetry",
            )
            telemetry_issues = _generation_telemetry_issues(
                pass_name=pass_name,
                raw_text=raw_text,
                artifact=telemetry,
                record_value=record.get(f"{pass_name}_generation"),
            )
            if telemetry_issues:
                _fail(
                    f"Development {pass_name} generation telemetry is invalid: {telemetry_issues}"
                )
        candidate = parse_generated_json(candidate_raw, candidate_schema)
        audit = parse_generated_json(audit_raw, audit_schema)
        if candidate != _read_object(paths["candidate_parsed"], "parsed candidate"):
            _fail("Raw candidate differs from its parsed artifact")
        if audit != _read_object(paths["audit_parsed"], "parsed audit"):
            _fail("Raw audit differs from its parsed artifact")

        ocr_payload = _read_object(paths["ocr"], "OCR artifact")
        lines = ocr_payload.get("lines")
        if not isinstance(lines, list) or not lines:
            _fail("OCR artifact lacks numbered lines")
        if not isinstance(ocr_payload.get("raw_word_data"), Mapping):
            _fail("OCR artifact lacks retained word-level data")
        if (
            ocr_payload.get("document_id") != row["document_id"]
            or ocr_payload.get("condition") != row["condition"]
        ):
            _fail("OCR artifact identity differs from the manifest row")
        recorded_ocr = record.get("ocr")
        expected_ocr = {
            key: ocr_payload.get(key)
            for key in (*STABLE_OCR_RUNTIME_KEYS, "elapsed_seconds", "line_count")
        }
        if recorded_ocr != expected_ocr:
            _fail("Run-record OCR metadata differs from the OCR artifact")
        rendered_candidate = _render_prompt(candidate_template, lines)
        rendered_audit = _render_prompt(audit_template, lines)
        if paths["candidate_prompt"].read_text(encoding="utf-8") != rendered_candidate:
            _fail("Candidate prompt differs from deterministic rendering")
        if paths["audit_prompt"].read_text(encoding="utf-8") != rendered_audit:
            _fail("Audit prompt differs from deterministic rendering")
        prompt_metadata = {
            "candidate_rendered_prompt": {
                "bytes": len(rendered_candidate.encode("utf-8")),
                "sha256": sha256_bytes(rendered_candidate.encode("utf-8")),
            },
            "audit_rendered_prompt": {
                "bytes": len(rendered_audit.encode("utf-8")),
                "sha256": sha256_bytes(rendered_audit.encode("utf-8")),
            },
        }
        for key, value in prompt_metadata.items():
            if record.get(key) != value:
                _fail(f"Run record {key} differs from reconstructed prompt")

        recompiled, compiler_log = compile_prediction(candidate, audit, lines)
        normalized = _read_object(paths["normalized"], "normalized prediction")
        if recompiled != normalized:
            _fail("Normalized prediction differs from deterministic recompilation")
        schema_errors = list(canonical_validator.iter_errors(normalized))
        if schema_errors:
            _fail("Normalized development prediction violates the canonical schema")
        recorded_compiler_log = _read_object(paths["compiler_audit"], "compiler audit")
        if compiler_log != recorded_compiler_log:
            _fail("Compiler audit differs from deterministic recompilation")
        expected_compiler_summary = {
            "compiler_version": compiler_log.get("compiler_version"),
            "accepted_non_null_count": compiler_log.get("accepted_non_null_count"),
            "rejected_non_null_count": compiler_log.get("rejected_non_null_count"),
            "history_evidence_rejection_count": compiler_log.get(
                "history_evidence_rejection_count"
            ),
            "schema_valid": compiler_log.get("schema_valid"),
        }
        if record.get("compiler_summary") != expected_compiler_summary:
            _fail("Run-record compiler summary differs from deterministic recompilation")

        records.append(record)
        compiler_logs.append(compiler_log)
        preflight.append(
            {
                "row": row,
                "paths": paths,
                "candidate": candidate,
                "normalized": normalized,
            }
        )

    repository_commits = {record["repository_commit"] for record in records}
    if len(repository_commits) != 1:
        _fail("Development run records were produced from more than one repository commit")
    repository_commit = next(iter(repository_commits))
    claim_path = (
        root
        / "results"
        / "v4"
        / "development"
        / "iteration-1"
        / "execution_attempts"
        / f"{DEVELOPMENT_ATTEMPT_ID}.claimed.json"
    )
    claim = _read_object(claim_path, "development one-shot claim")
    expected_claim = {
        "execution_attempt_id": DEVELOPMENT_ATTEMPT_ID,
        "status": "claimed",
        "protocol_version": PROTOCOL_VERSION,
        "repository_commit": repository_commit,
        "manifest_path": "data/v4/report_manifest.csv",
        "manifest_sha256": sha256_file(root / "data" / "v4" / "report_manifest.csv"),
        "selected_row_count": DEVELOPMENT_OUTPUT_COUNT,
        "condition": "all",
    }
    for key, value in expected_claim.items():
        if claim.get(key) != value:
            _fail(f"Development one-shot claim {key} is invalid")
    if not isinstance(claim.get("timestamp_utc"), str) or not claim["timestamp_utc"].strip():
        _fail("Development one-shot claim timestamp is missing")
    source_hardening = validate_post_development_immutability(
        root,
        repository_commit,
        post_development_immutable_paths,
    )
    runtime = common_runtime(records)
    evidence = _evidence_gate_summary(compiler_logs)
    recomputed = _recomputed_metrics(root, preflight, development_rows)
    generation_telemetry = {
        pass_name: {
            "cap_hit_count": sum(
                bool(
                    _read_object(
                        item["paths"][f"{pass_name}_generation"],
                        f"{pass_name} generation telemetry",
                    )["cap_hit"]
                )
                for item in preflight
            ),
            "eos_observed_count": sum(
                bool(
                    _read_object(
                        item["paths"][f"{pass_name}_generation"],
                        f"{pass_name} generation telemetry",
                    )["eos_observed"]
                )
                for item in preflight
            ),
            "token_counts": [
                _read_object(
                    item["paths"][f"{pass_name}_generation"],
                    f"{pass_name} generation telemetry",
                )["token_count"]
                for item in preflight
            ],
        }
        for pass_name in ("candidate", "audit")
    }
    gate = require_development_gate(
        recomputed["final_pipeline"],
        evidence,
        generation_telemetry,
    )

    base = root / "results" / "v4" / "development" / f"iteration-{iteration}"
    metrics_path = base / "metrics.json"
    metrics = _read_object(metrics_path, "development metrics")
    expected_headers = {
        "status": "DEVELOPMENT_PASS",
        "protocol_version": PROTOCOL_VERSION,
        "iteration": iteration,
        "output_count": DEVELOPMENT_OUTPUT_COUNT,
        "provenance_valid_count": DEVELOPMENT_OUTPUT_COUNT,
        "pass": True,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
    }
    for key, value in expected_headers.items():
        if metrics.get(key) != value:
            _fail(f"Development metrics {key} does not match the frozen protocol")
    for layer in ("candidate", "final_pipeline"):
        if metrics.get(layer) != recomputed[layer]:
            _fail(f"Development {layer} metrics differ from independent recomputation")
    if metrics.get("generation_telemetry") != generation_telemetry:
        _fail("Development generation telemetry differs from independent recomputation")
    if metrics.get("smoke_criteria") != gate["criteria"]:
        _fail("Development smoke criteria differ from independent recomputation")

    iteration_record_path = base / "iteration_record.json"
    iteration_record = _read_object(iteration_record_path, "development iteration record")
    expected_iteration_record = {
        "status": "DEVELOPMENT_PASS",
        "protocol_version": PROTOCOL_VERSION,
        "iteration": iteration,
        "manifest_sha256": sha256_file(root / "data" / "v4" / "report_manifest.csv"),
        "candidate_schema_sha256": sha256_file(root / "schema" / "v3" / "candidate.schema.json"),
        "audit_schema_sha256": sha256_file(root / "schema" / "v3" / "audit.schema.json"),
        "canonical_schema_sha256": sha256_file(root / "schema" / "extraction.schema.json"),
        "repository_commit": repository_commit,
        "metrics_path": _relative(metrics_path, root),
        "metrics_sha256": sha256_file(metrics_path),
        "run_record_sha256": {
            f"{row['document_id']}:{row['condition']}": sha256_file(
                _development_paths(root, row, iteration)["record"]
            )
            for row in development_rows
        },
    }
    for key, value in expected_iteration_record.items():
        if iteration_record.get(key) != value:
            _fail(f"Development iteration record {key} is invalid")
    for key in ("selection_rule",):
        if not isinstance(iteration_record.get(key), str) or not iteration_record[key].strip():
            _fail(f"Development iteration record {key} is missing")

    return {
        "runtime": runtime,
        "repository_commit": repository_commit,
        "post_development_source_hardening": source_hardening,
        "gate": gate,
        "metrics_sha256": sha256_file(metrics_path),
        "iteration_record_sha256": sha256_file(iteration_record_path),
    }


def collect_documented_development_iterations(
    root: Path,
    selected_iteration: int,
) -> Dict[str, Any]:
    if selected_iteration != 1:
        _fail("V4 permits exactly one development iteration")
    development_root = root / "results" / "v4" / "development"
    if not development_root.is_dir() or development_root.is_symlink():
        _fail("Development results directory is missing or unsafe")
    discovered_iterations: Dict[int, Path] = {}
    for path in development_root.iterdir():
        if not path.is_dir() or path.is_symlink():
            _fail(f"Unsafe development iteration path: {_relative(path, root)}")
        match = re.fullmatch(r"iteration-([0-9]+)", path.name)
        if not match:
            _fail(f"Malformed development iteration directory: {path.name}")
        iteration = int(match.group(1))
        if iteration != 1:
            _fail("V4 development results contain a forbidden additional iteration")
        discovered_iterations[iteration] = path
    expected_iterations = {1}
    if set(discovered_iterations) != expected_iterations:
        _fail(
            "Development iterations must be sequential and complete through the selected "
            f"iteration; found={sorted(discovered_iterations)}"
        )

    artifacts: Dict[str, str] = {}
    summaries: List[Dict[str, Any]] = []
    for iteration in sorted(discovered_iterations):
        base = discovered_iterations[iteration]
        files = sorted(path for path in base.rglob("*") if path.is_file())
        if any(path.is_symlink() for path in base.rglob("*")):
            _fail(f"Development iteration {iteration} contains a symlink")
        if not files:
            _fail(f"Development iteration {iteration} is empty")
        for path in files:
            artifacts[_relative(path, root)] = sha256_file(path)

        completed_path = base / "iteration_record.json"
        failed_path = base / "iteration_attempt.json"
        if completed_path.is_file() and failed_path.is_file():
            _fail(
                f"Development iteration {iteration} has both failed and completed primary records"
            )
        if iteration == selected_iteration:
            if not completed_path.is_file():
                _fail("Selected development iteration has no completed iteration record")
            status = "selected_complete"
            documentation_path = completed_path
        elif completed_path.is_file():
            status = "completed_not_selected"
            documentation_path = completed_path
        elif failed_path.is_file():
            attempt = _read_object(failed_path, "failed development iteration record")
            if (
                not isinstance(attempt.get("status"), str)
                or not attempt["status"].startswith("failed")
                or attempt.get("ground_truth_used_during_inference") is not False
            ):
                _fail(f"Development iteration {iteration} failure record is invalid")
            status = "failed_preserved"
            documentation_path = failed_path
        else:
            _fail(f"Development iteration {iteration} has no primary documentation record")
        summaries.append(
            {
                "iteration": iteration,
                "status": status,
                "documentation_path": _relative(documentation_path, root),
                "documentation_sha256": sha256_file(documentation_path),
                "artifact_count": len(files),
            }
        )
    return {
        "summaries": summaries,
        "artifacts": dict(sorted(artifacts.items())),
    }


def _ensure_commit_ancestry(
    root: Path,
    ancestor_commit: str,
    descendant_commit: str,
) -> None:
    result = _git(
        root,
        "merge-base",
        "--is-ancestor",
        ancestor_commit,
        descendant_commit,
        check=False,
    )
    if result.returncode != 0:
        _fail("Development execution commit is not an ancestor of the protocol source commit")


def build_lock_payload(
    *,
    protocol_source_commit: str,
    selected_iteration: int,
    development: Mapping[str, Any],
    documented_iterations: Mapping[str, Any],
    locked_source_files: Mapping[str, str],
    corpus_artifacts: Mapping[str, str],
    evaluation_only_corpus_artifacts: Mapping[str, str],
    v3_reuse_provenance: Mapping[str, Any],
) -> Dict[str, Any]:
    return {
        "status": "frozen",
        "protocol_version": PROTOCOL_VERSION,
        "protocol_identifier": PROTOCOL_IDENTIFIER,
        "protocol_source_commit": protocol_source_commit,
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
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
        "formal_output_count": FORMAL_OUTPUT_COUNT,
        "candidate_calls_per_input": 1,
        "audit_calls_per_input": 1,
        "formal_execution_attempt_id": FORMAL_EXECUTION_ATTEMPT_ID,
        "development_attempt_id": DEVELOPMENT_ATTEMPT_ID,
        "assistant_prefill": False,
        "do_sample": False,
        "num_beams": 1,
        "schema_enforcement": SCHEMA_ENFORCEMENT,
        "formal_execution_order": FORMAL_EXECUTION_ORDER,
        "infrastructure_retry_rule": INFRASTRUCTURE_RETRY_RULE,
        "pass_criteria": dict(PASS_CRITERIA),
        "development_iterations_used": selected_iteration,
        "selected_development_iteration": selected_iteration,
        "development_repository_commit": development["repository_commit"],
        "post_development_source_hardening": development["post_development_source_hardening"],
        "development_gate": development["gate"],
        "development_metrics_sha256": development["metrics_sha256"],
        "development_iteration_record_sha256": development["iteration_record_sha256"],
        "development_iterations": documented_iterations["summaries"],
        "runtime": development["runtime"],
        "locked_source_files": dict(sorted(locked_source_files.items())),
        "corpus_artifacts": dict(sorted(corpus_artifacts.items())),
        "evaluation_only_corpus_artifacts": dict(sorted(evaluation_only_corpus_artifacts.items())),
        "development_artifacts": documented_iterations["artifacts"],
        "v3_boundary_and_reuse_provenance": dict(v3_reuse_provenance),
    }


def _write_new_json(path: Path, payload: Mapping[str, Any]) -> None:
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError as exc:
        raise ProtocolFreezeError(f"Refusing to overwrite existing protocol lock: {path}") from exc


def freeze_protocol(
    selected_iteration: int,
    protocol_source_commit: str,
    *,
    root: Path | None = None,
    lock_path: Path | None = None,
) -> Dict[str, Any]:
    root = ROOT if root is None else root.resolve()
    lock_path = PROTOCOL_LOCK_PATH if lock_path is None else lock_path
    if selected_iteration != 1:
        _fail("Selected development iteration must be 1")
    if lock_path.exists() or lock_path.is_symlink():
        _fail(f"Refusing to overwrite existing protocol lock: {lock_path}")

    rows = load_and_validate_manifest(root)
    locked_source_files = collect_locked_source_hashes(root)
    all_corpus_artifacts = collect_corpus_artifacts(root, rows)
    corpus_partition = partition_corpus_artifacts(rows, all_corpus_artifacts)
    corpus_artifacts = corpus_partition["operational"]
    evaluation_only_corpus_artifacts = corpus_partition["evaluation_only"]
    v3_reuse_provenance, v3_bound_paths = collect_v3_reuse_provenance(root, rows)
    documented_iterations = collect_documented_development_iterations(
        root,
        selected_iteration,
    )
    development = validate_complete_development_iteration(
        root,
        rows,
        selected_iteration,
        [
            *REQUIRED_LOCKED_SOURCE_PATHS,
            *all_corpus_artifacts,
            *v3_bound_paths,
        ],
    )
    _ensure_commit_ancestry(
        root,
        development["repository_commit"],
        protocol_source_commit,
    )
    bound_paths = [
        *locked_source_files,
        *corpus_artifacts,
        *evaluation_only_corpus_artifacts,
        *documented_iterations["artifacts"],
        *v3_bound_paths,
    ]
    validate_protocol_source_commit(root, protocol_source_commit, bound_paths)
    payload = build_lock_payload(
        protocol_source_commit=protocol_source_commit,
        selected_iteration=selected_iteration,
        development=development,
        documented_iterations=documented_iterations,
        locked_source_files=locked_source_files,
        corpus_artifacts=corpus_artifacts,
        evaluation_only_corpus_artifacts=evaluation_only_corpus_artifacts,
        v3_reuse_provenance=v3_reuse_provenance,
    )
    _write_new_json(lock_path, payload)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--development-iteration",
        type=int,
        choices=(1,),
        required=True,
        help="Complete, passing development iteration to freeze.",
    )
    parser.add_argument(
        "--protocol-source-commit",
        required=True,
        help="Exact 40-character current HEAD containing all lock-bound inputs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        payload = freeze_protocol(
            args.development_iteration,
            args.protocol_source_commit,
        )
    except ProtocolFreezeError as exc:
        raise SystemExit(f"V4 protocol freeze refused: {exc}") from exc
    print(
        f"WROTE {PROTOCOL_LOCK_PATH.relative_to(ROOT)} for development iteration "
        f"{payload['selected_development_iteration']}."
    )


if __name__ == "__main__":
    main()
