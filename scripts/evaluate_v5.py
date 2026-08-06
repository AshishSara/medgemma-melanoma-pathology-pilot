#!/usr/bin/env python3
"""Evaluate the locked v5 qualification or formal split without test-set peeking."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

from jsonschema import Draft202012Validator
from pilot_utils import (
    PRIMARY_FIELD_PATHS,
    ROOT,
    compare_prediction,
    flatten_json,
    read_json,
    safe_divide,
    sha256_file,
    values_exactly_equal,
    write_json,
)
from run_v5_inference import (
    AUDIT_MAX_NEW_TOKENS,
    AUDIT_PROMPT_PATH,
    AUDIT_SCHEMA_PATH,
    AUDIT_SERIALIZER_FORCE_JSON_FIELD_ORDER,
    AUDIT_SERIALIZER_MAX_CONSECUTIVE_WHITESPACES,
    CANDIDATE_MAX_NEW_TOKENS,
    CANDIDATE_PROMPT_PATH,
    CANDIDATE_SCHEMA_PATH,
    CANDIDATE_SERIALIZER_FORCE_JSON_FIELD_ORDER,
    CANDIDATE_SERIALIZER_MAX_CONSECUTIVE_WHITESPACES,
    CANONICAL_SCHEMA_PATH,
    COMPILER_PATH,
    DTYPE_NAME,
    MODEL_ID,
    MODEL_REVISION,
    PROTOCOL_LOCK_PATH,
    PROTOCOL_VERSION,
    RANDOM_SEED,
    V5_MANIFEST,
    output_paths,
    parse_generated_json,
    render_audit_prompt,
    render_candidate_prompt,
    verify_protocol_lock,
)
from run_v5_inference import SCRIPT_PATH as INFERENCE_SCRIPT_PATH
from v3_pipeline import CLINICAL_FIELD_PATHS, compile_prediction

CONDITIONS = ("clean", "ocr_degraded")
SPLIT_OUTPUT_COUNTS = {"development": 4, "formal": 20}
GATE_THRESHOLDS: Dict[str, Dict[str, Any]] = {
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

METRICS_PATHS = {
    split: ROOT / "results" / "v5" / split / "metrics.json" for split in SPLIT_OUTPUT_COUNTS
}
OUTPUT_SEAL_PATHS = {
    split: ROOT / "results" / "v5" / split / "output_seal.json" for split in SPLIT_OUTPUT_COUNTS
}
MANUAL_REVIEW_PATH = ROOT / "results" / "v5" / "formal" / "manual_review_blinded.csv"
HUMAN_ATTESTATION_PATH = ROOT / "results" / "v5" / "formal" / "human_review_attestation.json"
HUMAN_ATTESTATION_TEXT = (
    "I visually reviewed every assigned v5 formal artifact bundle without correcting "
    "predictions and completed the blinded checklist before viewing the automated result."
)
HISTORY_SENTINEL_FIELDS = {
    "MEL-203": ("breslow_thickness_mm", "breslow_qualifier", "staging.pT"),
}
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_COMMIT_RE = re.compile(r"[0-9a-f]{40}")


def _relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _descriptor(path: Path) -> Dict[str, Any]:
    return {
        "path": _relative(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _expected_text_descriptor(value: str) -> Dict[str, Any]:
    return {"bytes": len(value.encode("utf-8")), "sha256": _sha256_text(value)}


def _locked_sha(value: Any) -> str | None:
    if isinstance(value, str) and _SHA256_RE.fullmatch(value):
        return value
    if isinstance(value, Mapping):
        sha = value.get("sha256")
        if isinstance(sha, str) and _SHA256_RE.fullmatch(sha):
            return sha
    return None


def _load_manifest(split: str) -> List[Dict[str, str]]:
    with V5_MANIFEST.open(newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle) if row.get("split") == split]
    required = SPLIT_OUTPUT_COUNTS[split]
    if len(rows) != required:
        raise ValueError(f"Expected {required} v5 {split} rows, found {len(rows)}")
    if any(row.get("dataset_version") != "pilot-v5" for row in rows):
        raise ValueError(f"The v5 {split} manifest contains a non-v5 row")
    keys = [(row["document_id"], row["condition"]) for row in rows]
    if len(set(keys)) != required:
        raise ValueError(f"The v5 {split} manifest contains duplicate assigned inputs")
    if {row["condition"] for row in rows} != set(CONDITIONS):
        raise ValueError(f"The v5 {split} manifest does not contain both fixed conditions")
    condition_counts = Counter(row["condition"] for row in rows)
    if any(
        condition_counts[condition] * 16 != GATE_THRESHOLDS[split]["condition_field_opportunities"]
        for condition in CONDITIONS
    ):
        raise ValueError(f"The v5 {split} condition denominators do not match the protocol")
    return rows


def _verify_operational_lock(
    protocol_lock: Mapping[str, Any],
    manifest: Sequence[Mapping[str, str]],
) -> None:
    """Verify manifest and inference-visible corpus bytes without opening truth."""

    frozen = protocol_lock.get("corpus_artifacts")
    if not isinstance(frozen, Mapping):
        raise ValueError("Protocol lock has no operational corpus mapping")
    required_paths = {_relative(V5_MANIFEST)}
    for row in manifest:
        required_paths.update((row["pdf_path"], row["image_path"]))
    for relative_path in sorted(required_paths):
        expected_sha = _locked_sha(frozen.get(relative_path))
        if expected_sha is None:
            raise ValueError(f"Protocol lock does not bind operational artifact {relative_path}")
        path = ROOT / relative_path
        if (
            ".." in Path(relative_path).parts
            or not path.is_file()
            or path.is_symlink()
            or sha256_file(path) != expected_sha
        ):
            raise ValueError(f"Operational corpus hash mismatch: {relative_path}")


def _verify_evaluation_only_corpus(
    protocol_lock: Mapping[str, Any],
    manifest: Sequence[Mapping[str, str]],
) -> Dict[str, str]:
    """Verify truth-side bytes only after the split output seal exists."""

    frozen = protocol_lock.get("evaluation_only_corpus_artifacts")
    if not isinstance(frozen, Mapping) or not frozen:
        raise ValueError("Protocol lock has no evaluation-only corpus mapping")
    with V5_MANIFEST.open(newline="", encoding="utf-8") as handle:
        all_rows = list(csv.DictReader(handle))
    expected_paths = {"data/v5/cases.csv"}
    for row in all_rows:
        expected_paths.update((row["source_text_path"], row["ground_truth_path"]))
    if set(frozen) != expected_paths:
        raise ValueError("Evaluation-only corpus mapping does not match the full v5 manifest")
    verified: Dict[str, str] = {}
    for relative_path in sorted(expected_paths):
        expected_sha = _locked_sha(frozen.get(relative_path))
        path = ROOT / relative_path
        if (
            expected_sha is None
            or ".." in Path(relative_path).parts
            or not relative_path.startswith(
                ("data/v5/cases.csv", "data/v5/source_text/", "data/v5/ground_truth/")
            )
            or not path.is_file()
            or path.is_symlink()
            or sha256_file(path) != expected_sha
        ):
            raise ValueError(f"Evaluation-only corpus hash mismatch: {relative_path}")
        verified[relative_path] = expected_sha
    # `manifest` is deliberately accepted to make the caller's sealed split explicit.
    if not manifest:
        raise ValueError("Cannot verify truth-side corpus for an empty split")
    return verified


def _artifact_descriptor_errors(name: str, path: Path, descriptor: Any) -> List[str]:
    if not path.is_file() or path.is_symlink():
        return [f"{name}: missing or unsafe {_relative(path)}"]
    if not isinstance(descriptor, Mapping):
        return [f"{name}: missing run-record descriptor"]
    expected = _descriptor(path)
    return [
        f"{name}.{key}: expected {value!r}, found {descriptor.get(key)!r}"
        for key, value in expected.items()
        if descriptor.get(key) != value
    ]


def _event_reference_issues(name: str, path: Path, reference: Mapping[str, Any]) -> List[str]:
    if not path.is_file() or path.is_symlink():
        return [f"{name} is missing or unsafe"]
    if reference.get("sha256") != sha256_file(path):
        return [f"{name} hash is invalid"]
    return []


def _generation_issues(pass_name: str, raw_text: str, generation: Any) -> List[str]:
    issues: List[str] = []
    if not isinstance(generation, Mapping):
        return [f"{pass_name} generation metadata is not an object"]
    maximum = CANDIDATE_MAX_NEW_TOKENS if pass_name == "candidate" else AUDIT_MAX_NEW_TOKENS
    expected_serializer = {
        "configuration_stage": "after_transformers_prefix_construction",
        "tokenizer_alphabet_preserved": True,
        "force_json_field_order": (
            CANDIDATE_SERIALIZER_FORCE_JSON_FIELD_ORDER
            if pass_name == "candidate"
            else AUDIT_SERIALIZER_FORCE_JSON_FIELD_ORDER
        ),
        "max_consecutive_whitespaces": (
            CANDIDATE_SERIALIZER_MAX_CONSECUTIVE_WHITESPACES
            if pass_name == "candidate"
            else AUDIT_SERIALIZER_MAX_CONSECUTIVE_WHITESPACES
        ),
        "max_json_array_length": 20,
    }
    if generation.get("pass") != pass_name:
        issues.append(f"{pass_name} generation pass identity is invalid")
    if generation.get("max_new_tokens") != maximum:
        issues.append(f"{pass_name} generation maximum is invalid")
    token_count = generation.get("token_count")
    token_count_valid = (
        isinstance(token_count, int)
        and not isinstance(token_count, bool)
        and 0 < token_count <= maximum
    )
    if not token_count_valid:
        issues.append(f"{pass_name} token_count is invalid")
    cap_hit = generation.get("cap_hit")
    if not isinstance(cap_hit, bool):
        issues.append(f"{pass_name} cap_hit is not boolean")
    elif token_count_valid:
        mechanically_at_cap = token_count >= maximum
        if cap_hit != mechanically_at_cap:
            issues.append(f"{pass_name} cap_hit contradicts its token count")
        if mechanically_at_cap:
            issues.append(f"{pass_name} generation reached its token cap")
    generated_ids = generation.get("generated_token_ids")
    if (
        not isinstance(generated_ids, list)
        or any(
            not isinstance(value, int) or isinstance(value, bool) or value < 0
            for value in generated_ids
        )
        or (token_count_valid and len(generated_ids) != token_count)
    ):
        issues.append(f"{pass_name} generated token IDs are invalid")
    else:
        encoded = json.dumps(generated_ids, separators=(",", ":"), ensure_ascii=True).encode(
            "ascii"
        )
        if generation.get("generated_token_ids_sha256") != hashlib.sha256(encoded).hexdigest():
            issues.append(f"{pass_name} generated token-ID digest is invalid")
    prompt_count = generation.get("prompt_token_count")
    if not isinstance(prompt_count, int) or isinstance(prompt_count, bool) or prompt_count <= 0:
        issues.append(f"{pass_name} prompt_token_count is invalid")
    elapsed = generation.get("elapsed_seconds")
    if (
        not isinstance(elapsed, (int, float))
        or isinstance(elapsed, bool)
        or not math.isfinite(float(elapsed))
        or float(elapsed) < 0
    ):
        issues.append(f"{pass_name} elapsed_seconds is invalid")
    if generation.get("raw_text") != _expected_text_descriptor(raw_text):
        issues.append(f"{pass_name} raw-text descriptor is invalid")
    if generation.get("decoding") != {
        "do_sample": False,
        "num_beams": 1,
        "skip_special_tokens": True,
    }:
        issues.append(f"{pass_name} decoding settings are invalid")
    serializer = generation.get("serializer")
    if not isinstance(serializer, Mapping):
        issues.append(f"{pass_name} serializer metadata is not an object")
    else:
        for key, value in expected_serializer.items():
            if serializer.get(key) != value:
                issues.append(
                    f"{pass_name} serializer {key}: expected {value!r}, "
                    f"found {serializer.get(key)!r}"
                )
    return issues


def _bundle_issues(
    *,
    pass_name: str,
    bundle: Any,
    row: Mapping[str, str],
    protocol_lock: Mapping[str, Any],
    rendered_prompt: str,
    schema_path: Path,
) -> List[str]:
    if not isinstance(bundle, Mapping):
        return [f"{pass_name} response bundle is not an object"]
    issues: List[str] = []
    repository_commit = bundle.get("repository_commit")
    expected = {
        "protocol_version": PROTOCOL_VERSION,
        "protocol_lock_path": _relative(PROTOCOL_LOCK_PATH),
        "protocol_lock_sha256": sha256_file(PROTOCOL_LOCK_PATH),
        "protocol_source_commit": protocol_lock.get("protocol_source_commit"),
        "dataset_version": "pilot-v5",
        "split": row["split"],
        "semantic_case_id": row["semantic_case_id"],
        "document_id": row["document_id"],
        "condition": row["condition"],
        "image_path": row["image_path"],
        "image_sha256": row["image_sha256"],
        "pass": pass_name,
        "model_id": MODEL_ID,
        "requested_revision": MODEL_REVISION,
        "resolved_revision": MODEL_REVISION,
        "prompt_template_path": _relative(
            CANDIDATE_PROMPT_PATH if pass_name == "candidate" else AUDIT_PROMPT_PATH
        ),
        "prompt_template_sha256": sha256_file(
            CANDIDATE_PROMPT_PATH if pass_name == "candidate" else AUDIT_PROMPT_PATH
        ),
        "rendered_prompt": _expected_text_descriptor(rendered_prompt),
        "response_schema_path": _relative(schema_path),
        "response_schema_sha256": sha256_file(schema_path),
    }
    for key, value in expected.items():
        if bundle.get(key) != value:
            issues.append(
                f"{pass_name} bundle {key}: expected {value!r}, found {bundle.get(key)!r}"
            )
    if not isinstance(repository_commit, str) or not _COMMIT_RE.fullmatch(repository_commit):
        issues.append(f"{pass_name} bundle repository_commit is invalid")
    call_ordinal = bundle.get("call_ordinal")
    if not isinstance(call_ordinal, int) or isinstance(call_ordinal, bool) or call_ordinal < 1:
        issues.append(f"{pass_name} bundle call_ordinal is invalid")
    attempt_id = bundle.get("execution_attempt_id")
    if not isinstance(attempt_id, str) or not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9._-]*", attempt_id
    ):
        issues.append(f"{pass_name} bundle execution_attempt_id is invalid")
    raw_text = bundle.get("raw_text")
    if not isinstance(raw_text, str):
        issues.append(f"{pass_name} bundle raw_text is not a string")
        raw_text = ""
    raw_descriptor = bundle.get("raw_text_descriptor")
    if raw_descriptor != _expected_text_descriptor(raw_text):
        issues.append(f"{pass_name} bundle raw_text_descriptor is invalid")
    issues.extend(_generation_issues(pass_name, raw_text, bundle.get("generation")))
    persisted_event = bundle.get("response_persisted_event")
    if not isinstance(persisted_event, Mapping):
        issues.append(f"{pass_name} bundle has no response-persisted event binding")
    else:
        path_value = persisted_event.get("path")
        expected_prefix = (
            f"results/v5/{row['split']}/call_events/{pass_name}/"
            f"{row['condition']}/{row['document_id']}/"
        )
        if (
            not isinstance(path_value, str)
            or not path_value.startswith(expected_prefix)
            or not path_value.endswith(".response_persisted.json")
            or ".." in Path(path_value).parts
        ):
            issues.append(f"{pass_name} response-persisted event path is unsafe")
        else:
            event_path = ROOT / path_value
            issues.extend(
                _artifact_descriptor_errors(
                    f"{pass_name}_response_persisted_event",
                    event_path,
                    persisted_event,
                )
            )
    return issues


def _expected_record_values(
    row: Mapping[str, str],
    protocol_lock: Mapping[str, Any],
) -> Dict[str, Any]:
    safe_row = {key: value for key, value in row.items() if key != "ground_truth_path"}
    return {
        "attempt_status": "completed",
        "protocol_version": PROTOCOL_VERSION,
        "protocol_lock_path": _relative(PROTOCOL_LOCK_PATH),
        "protocol_lock_sha256": sha256_file(PROTOCOL_LOCK_PATH),
        "protocol_source_commit": protocol_lock.get("protocol_source_commit"),
        "dataset_version": "pilot-v5",
        "generator_version": row["generator_version"],
        "split": row["split"],
        "semantic_case_id": row["semantic_case_id"],
        "document_id": row["document_id"],
        "template_id": row["template_id"],
        "condition": row["condition"],
        "manifest_path": _relative(V5_MANIFEST),
        "manifest_sha256": sha256_file(V5_MANIFEST),
        "manifest_row_without_ground_truth": safe_row,
        "model_id": MODEL_ID,
        "requested_revision": MODEL_REVISION,
        "resolved_revision": MODEL_REVISION,
        "requested_dtype": DTYPE_NAME,
        "do_sample": False,
        "num_beams": 1,
        "random_seed": RANDOM_SEED,
        "assistant_prefill": False,
        "candidate_max_new_tokens": CANDIDATE_MAX_NEW_TOKENS,
        "audit_max_new_tokens": AUDIT_MAX_NEW_TOKENS,
        "image_path": row["image_path"],
        "image_sha256": row["image_sha256"],
        "candidate_prompt_template_path": _relative(CANDIDATE_PROMPT_PATH),
        "candidate_prompt_template_sha256": sha256_file(CANDIDATE_PROMPT_PATH),
        "audit_prompt_template_path": _relative(AUDIT_PROMPT_PATH),
        "audit_prompt_template_sha256": sha256_file(AUDIT_PROMPT_PATH),
        "audit_is_blind": True,
        "audit_prompt_constructed_before_candidate_generation": True,
        "candidate_schema_path": _relative(CANDIDATE_SCHEMA_PATH),
        "candidate_schema_sha256": sha256_file(CANDIDATE_SCHEMA_PATH),
        "audit_schema_path": _relative(AUDIT_SCHEMA_PATH),
        "audit_schema_sha256": sha256_file(AUDIT_SCHEMA_PATH),
        "canonical_schema_path": _relative(CANONICAL_SCHEMA_PATH),
        "canonical_schema_sha256": sha256_file(CANONICAL_SCHEMA_PATH),
        "compiler_path": _relative(COMPILER_PATH),
        "compiler_sha256": sha256_file(COMPILER_PATH),
        "inference_script_path": _relative(INFERENCE_SCRIPT_PATH),
        "inference_script_sha256": sha256_file(INFERENCE_SCRIPT_PATH),
    }


def _preflight_row(
    row: Mapping[str, str],
    protocol_lock: Mapping[str, Any],
) -> Dict[str, Any]:
    """Validate a complete row, including deterministic replay, without truth."""

    paths = output_paths(row)
    issues: List[str] = []
    for path in paths.values():
        if not path.is_file() or path.is_symlink():
            issues.append(f"missing or unsafe {_relative(path)}")
    if issues:
        return {"row": dict(row), "paths": paths, "record": {}, "issues": issues}
    try:
        record = read_json(paths["record"])
    except (OSError, ValueError) as exc:
        return {
            "row": dict(row),
            "paths": paths,
            "record": {},
            "issues": [f"unreadable run record: {exc}"],
        }
    if not isinstance(record, Mapping):
        return {
            "row": dict(row),
            "paths": paths,
            "record": {},
            "issues": ["run record is not an object"],
        }
    for key, value in _expected_record_values(row, protocol_lock).items():
        if record.get(key) != value:
            issues.append(f"record {key}: expected {value!r}, found {record.get(key)!r}")
    repository_commit = record.get("repository_commit")
    if not isinstance(repository_commit, str) or not _COMMIT_RE.fullmatch(repository_commit):
        issues.append("record repository_commit is invalid")
    execution_attempt_id = record.get("execution_attempt_id")
    if not isinstance(execution_attempt_id, str) or not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9._-]*", execution_attempt_id
    ):
        issues.append("record execution_attempt_id is invalid")
    descriptors = record.get("artifacts")
    if not isinstance(descriptors, Mapping):
        issues.append("record artifacts is not an object")
        descriptors = {}
    for name, path in paths.items():
        if name != "record":
            issues.extend(_artifact_descriptor_errors(name, path, descriptors.get(name)))

    candidate = audit = normalized = ocr_payload = compiler_log = None
    candidate_bundle = audit_bundle = None
    try:
        ocr_payload = read_json(paths["ocr"])
        if (
            not isinstance(ocr_payload, Mapping)
            or not isinstance(ocr_payload.get("lines"), list)
            or not isinstance(ocr_payload.get("raw_word_data"), Mapping)
        ):
            raise ValueError("OCR artifact lacks lines or retained raw word data")
        lines = ocr_payload["lines"]
        candidate_prompt = render_candidate_prompt(lines)
        audit_prompt = render_audit_prompt(lines)
        if paths["candidate_prompt"].read_text(encoding="utf-8") != candidate_prompt:
            issues.append("retained candidate prompt differs from deterministic rendering")
        if paths["audit_prompt"].read_text(encoding="utf-8") != audit_prompt:
            issues.append("retained audit prompt differs from deterministic rendering")

        candidate_bundle = read_json(paths["candidate_bundle"])
        audit_bundle = read_json(paths["audit_bundle"])
        issues.extend(
            _bundle_issues(
                pass_name="candidate",
                bundle=candidate_bundle,
                row=row,
                protocol_lock=protocol_lock,
                rendered_prompt=candidate_prompt,
                schema_path=CANDIDATE_SCHEMA_PATH,
            )
        )
        issues.extend(
            _bundle_issues(
                pass_name="audit",
                bundle=audit_bundle,
                row=row,
                protocol_lock=protocol_lock,
                rendered_prompt=audit_prompt,
                schema_path=AUDIT_SCHEMA_PATH,
            )
        )
        if (
            isinstance(candidate_bundle, Mapping)
            and candidate_bundle.get("repository_commit") != repository_commit
        ):
            issues.append("candidate bundle and run-record commits differ")
        if (
            isinstance(audit_bundle, Mapping)
            and audit_bundle.get("repository_commit") != repository_commit
        ):
            issues.append("audit bundle and run-record commits differ")
        if (
            isinstance(candidate_bundle, Mapping)
            and candidate_bundle.get("execution_attempt_id") != execution_attempt_id
        ):
            issues.append("candidate bundle and run-record attempt IDs differ")
        if (
            isinstance(audit_bundle, Mapping)
            and audit_bundle.get("execution_attempt_id") != execution_attempt_id
        ):
            issues.append("audit bundle and run-record attempt IDs differ")

        candidate = parse_generated_json(
            candidate_bundle["raw_text"], read_json(CANDIDATE_SCHEMA_PATH)
        )
        audit = parse_generated_json(audit_bundle["raw_text"], read_json(AUDIT_SCHEMA_PATH))
        if candidate != read_json(paths["candidate_parsed"]):
            issues.append("candidate bundle differs from parsed candidate artifact")
        if audit != read_json(paths["audit_parsed"]):
            issues.append("audit bundle differs from parsed audit artifact")

        normalized, compiler_log = compile_prediction(candidate, audit, lines)
        if normalized != read_json(paths["normalized"]):
            issues.append("normalized output differs from deterministic recompilation")
        if compiler_log != read_json(paths["compiler_audit"]):
            issues.append("compiler audit differs from deterministic recompilation")
        canonical_errors = list(
            Draft202012Validator(read_json(CANONICAL_SCHEMA_PATH)).iter_errors(normalized)
        )
        if canonical_errors:
            issues.append("deterministic compiled output violates the canonical schema")
    except Exception as exc:
        issues.append(f"artifact validation failed: {type(exc).__name__}: {exc}")

    return {
        "row": dict(row),
        "paths": paths,
        "record": dict(record),
        "candidate_bundle": candidate_bundle,
        "audit_bundle": audit_bundle,
        "candidate": candidate,
        "audit": audit,
        "normalized": normalized,
        "ocr": ocr_payload,
        "compiler_log": compiler_log,
        "issues": issues,
    }


def _validate_event_ledgers(
    split: str,
    preflight: Sequence[Mapping[str, Any]],
) -> tuple[Dict[str, Any], List[str]]:
    """Hash all append-only attempt/call events and enforce bundle references."""

    issues: List[str] = []
    base = ROOT / "results" / "v5" / split
    descriptors: List[Dict[str, Any]] = []
    persisted_references: set[str] = set()
    attempts: set[str] = set()
    record_commits = {
        (item["row"]["condition"], item["row"]["document_id"]): item.get("record", {}).get(
            "repository_commit"
        )
        for item in preflight
    }
    call_chains: Dict[tuple[str, str, str], Dict[int, Dict[str, Mapping[str, Any]]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    for item in preflight:
        record = item.get("record")
        if isinstance(record, Mapping) and isinstance(record.get("execution_attempt_id"), str):
            attempts.add(record["execution_attempt_id"])
        for pass_name in ("candidate", "audit"):
            bundle = item.get(f"{pass_name}_bundle")
            if not isinstance(bundle, Mapping):
                continue
            binding = bundle.get("response_persisted_event")
            if isinstance(binding, Mapping) and isinstance(binding.get("path"), str):
                persisted_references.add(binding["path"])

    for namespace in ("call_events", "execution_attempts"):
        directory = base / namespace
        if not directory.is_dir() or directory.is_symlink():
            issues.append(f"{split} {namespace} ledger is missing or unsafe")
            continue
        files = sorted(directory.rglob("*.json"))
        if not files:
            issues.append(f"{split} {namespace} ledger is empty")
        for path in files:
            if not path.is_file() or path.is_symlink():
                issues.append(f"unsafe ledger entry: {_relative(path)}")
                continue
            try:
                event = read_json(path)
            except (OSError, ValueError) as exc:
                issues.append(f"unreadable ledger entry {_relative(path)}: {exc}")
                continue
            if not isinstance(event, Mapping):
                issues.append(f"ledger event is not an object: {_relative(path)}")
                continue
            if event.get("protocol_version") != PROTOCOL_VERSION:
                issues.append(f"ledger event has wrong protocol: {_relative(path)}")
            if (
                not isinstance(event.get("timestamp_utc"), str)
                or not event["timestamp_utc"].strip()
            ):
                issues.append(f"ledger event has invalid timestamp: {_relative(path)}")
            if event.get("protocol_lock_sha256") != sha256_file(PROTOCOL_LOCK_PATH):
                issues.append(f"ledger event has wrong lock: {_relative(path)}")
            if event.get("split") != split:
                issues.append(f"ledger event has wrong split: {_relative(path)}")
            attempt_id = event.get("execution_attempt_id")
            if not isinstance(attempt_id, str) or not re.fullmatch(
                r"[A-Za-z0-9][A-Za-z0-9._-]*", attempt_id
            ):
                issues.append(f"ledger event has invalid attempt ID: {_relative(path)}")
            elif namespace == "call_events":
                attempts.add(attempt_id)
            if namespace == "call_events":
                relative_event = path.relative_to(directory).as_posix()
                match = re.fullmatch(
                    r"(candidate|audit)/(clean|ocr_degraded)/"
                    r"(MEL-[0-9]{3}-[AB])/([0-9]{3})\."
                    r"(started|no_response|response_persisted)\.json",
                    relative_event,
                )
                if match is None:
                    issues.append(f"invalid call-event path: {_relative(path)}")
                else:
                    pass_name, condition, document_id, ordinal_text, event_name = match.groups()
                    ordinal = int(ordinal_text)
                    chain = call_chains[(pass_name, condition, document_id)][ordinal]
                    if event_name in chain:
                        issues.append(f"duplicate call event: {_relative(path)}")
                    chain[event_name] = event
                    expected_event_identity = {
                        "event": event_name,
                        "pass": pass_name,
                        "condition": condition,
                        "document_id": document_id,
                        "call_ordinal": ordinal,
                        "repository_commit": record_commits.get((condition, document_id)),
                    }
                    for key, value in expected_event_identity.items():
                        if event.get(key) != value:
                            issues.append(
                                f"{_relative(path)} {key}: expected {value!r}, "
                                f"found {event.get(key)!r}"
                            )
            descriptors.append(_descriptor(path))

    persisted_files = {
        item["path"]
        for item in descriptors
        if item["path"].startswith(f"results/v5/{split}/call_events/")
        and item["path"].endswith(".response_persisted.json")
    }
    if len(persisted_files) != 2 * SPLIT_OUTPUT_COUNTS[split]:
        issues.append(
            f"{split} has {len(persisted_files)} response-persisted events; "
            f"expected {2 * SPLIT_OUTPUT_COUNTS[split]}"
        )
    if persisted_references != persisted_files:
        issues.append("response bundles do not hash-bind exactly the persisted-response events")
    expected_call_keys = {
        (pass_name, row["condition"], row["document_id"])
        for row in (item["row"] for item in preflight)
        for pass_name in ("candidate", "audit")
    }
    if set(call_chains) != expected_call_keys:
        issues.append("call-event ledger keys do not exactly match the assigned split")
    for key in sorted(expected_call_keys):
        ordinals = call_chains.get(key, {})
        if not ordinals:
            issues.append(f"call-event chain is missing for {key}")
            continue
        sorted_ordinals = sorted(ordinals)
        if sorted_ordinals != list(range(1, len(sorted_ordinals) + 1)):
            issues.append(f"call-event ordinals are not contiguous for {key}")
        if len(sorted_ordinals) > 2:
            issues.append(f"more than one infrastructure retry occurred for {key}")
        no_response_count = 0
        persisted_count = 0
        for ordinal in sorted_ordinals:
            events = ordinals[ordinal]
            if "started" not in events:
                issues.append(f"call {key} ordinal {ordinal} has no started event")
            terminals = {name for name in ("no_response", "response_persisted") if name in events}
            if len(terminals) != 1:
                issues.append(f"call {key} ordinal {ordinal} must have exactly one terminal event")
            no_response_count += int("no_response" in terminals)
            persisted_count += int("response_persisted" in terminals)
        if no_response_count > 1:
            issues.append(f"more than one no-response failure occurred for {key}")
        if persisted_count != 1:
            issues.append(f"call chain {key} must contain exactly one persisted response")
        if len(sorted_ordinals) == 2 and (
            "no_response" not in ordinals[1] or "response_persisted" not in ordinals[2]
        ):
            issues.append(f"call chain {key} has an invalid infrastructure retry order")
    if not attempts:
        issues.append(f"{split} run records have no execution attempt")
    return {
        "event_count": len(descriptors),
        "execution_attempt_ids": sorted(attempts),
        "events": sorted(descriptors, key=lambda value: value["path"]),
    }, issues


def _incomplete_payload(
    split: str,
    preflight: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    required = SPLIT_OUTPUT_COUNTS[split]
    valid = sum(not item["issues"] for item in preflight)
    criteria = {
        name: False
        for name in (
            "complete_and_provenance_valid",
            "candidate_strict_schema_valid",
            "audit_strict_schema_valid",
            "canonical_schema_valid",
            "zero_candidate_cap_hits",
            "zero_audit_cap_hits",
            "pooled_field_exact_minimum",
            "each_condition_field_exact_minimum",
            "nonnull_recall_minimum",
            "unsupported_maximum",
            "accepted_non_null_evidence_100_percent",
            "zero_history_carryover",
        )
    }
    return {
        "status": f"{split.upper()}_INCOMPLETE",
        "protocol_version": PROTOCOL_VERSION,
        "split": split,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "ground_truth_read": False,
        "intended_output_count": required,
        "output_count": required,
        "provenance_valid_count": valid,
        "field_exact_total_denominator": GATE_THRESHOLDS[split]["field_opportunities"],
        "nonnull_total_denominator": (
            GATE_THRESHOLDS[split]["ground_truth_non_null_opportunities"]
        ),
        "null_total_denominator": (GATE_THRESHOLDS[split]["ground_truth_null_opportunities"]),
        "artifact_issues": [
            {
                "document_id": item["row"]["document_id"],
                "condition": item["row"]["condition"],
                "errors": item["issues"],
            }
            for item in preflight
            if item["issues"]
        ],
        "gate_thresholds": GATE_THRESHOLDS[split],
        "gate_criteria": criteria,
        "automated_gate_pass": False,
        "pass": False,
        "human_review_complete": False if split == "formal" else None,
        "outreach_ready": False,
    }


def _write_output_seal(
    split: str,
    preflight: Sequence[Mapping[str, Any]],
    ledger: Mapping[str, Any],
    development_gate: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    commits = {item["record"]["repository_commit"] for item in preflight}
    if len(commits) != 1:
        raise ValueError(f"{split} run records do not share one repository commit")
    rows = []
    for item in preflight:
        paths = item["paths"]
        rows.append(
            {
                "document_id": item["row"]["document_id"],
                "condition": item["row"]["condition"],
                "run_record": _descriptor(paths["record"]),
                "artifacts": item["record"]["artifacts"],
            }
        )
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "split": split,
        "repository_commit": commits.pop(),
        "manifest_sha256": sha256_file(V5_MANIFEST),
        "protocol_lock_sha256": sha256_file(PROTOCOL_LOCK_PATH),
        "output_count": len(rows),
        "ground_truth_read": False,
        "rows": rows,
        "event_ledger": dict(ledger),
        "development_gate": (dict(development_gate) if development_gate is not None else None),
    }
    write_json(OUTPUT_SEAL_PATHS[split], payload)
    return payload


def _verify_development_gate_for_formal() -> Dict[str, Any]:
    """Fail closed unless the locked four-input qualification gate passed."""

    metrics_path = METRICS_PATHS["development"]
    seal_path = OUTPUT_SEAL_PATHS["development"]
    for name, path in (("development metrics", metrics_path), ("development seal", seal_path)):
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"Formal evaluation requires safe {name}: {_relative(path)}")
    metrics = read_json(metrics_path)
    seal = read_json(seal_path)
    if not isinstance(metrics, Mapping) or not isinstance(seal, Mapping):
        raise ValueError("Development gate artifacts must be JSON objects")
    expected_metrics = {
        "status": "DEVELOPMENT_PASS",
        "protocol_version": PROTOCOL_VERSION,
        "split": "development",
        "ground_truth_read": True,
        "intended_output_count": SPLIT_OUTPUT_COUNTS["development"],
        "output_count": SPLIT_OUTPUT_COUNTS["development"],
        "provenance_valid_count": SPLIT_OUTPUT_COUNTS["development"],
        "gate_thresholds": GATE_THRESHOLDS["development"],
        "automated_gate_pass": True,
        "pass": True,
        "outreach_ready": False,
    }
    for key, value in expected_metrics.items():
        if metrics.get(key) != value:
            raise ValueError(
                f"Development gate {key}: expected {value!r}, found {metrics.get(key)!r}"
            )
    criteria = metrics.get("gate_criteria")
    if (
        not isinstance(criteria, Mapping)
        or not criteria
        or any(value is not True for value in criteria.values())
    ):
        raise ValueError("Development gate criteria are incomplete or did not all pass")
    provenance = metrics.get("provenance")
    if (
        not isinstance(provenance, Mapping)
        or provenance.get("output_seal_path") != _relative(seal_path)
        or provenance.get("output_seal_sha256") != sha256_file(seal_path)
    ):
        raise ValueError("Development metrics do not hash-bind the development output seal")
    if (
        seal.get("protocol_version") != PROTOCOL_VERSION
        or seal.get("split") != "development"
        or seal.get("output_count") != SPLIT_OUTPUT_COUNTS["development"]
        or seal.get("protocol_lock_sha256") != sha256_file(PROTOCOL_LOCK_PATH)
        or seal.get("manifest_sha256") != sha256_file(V5_MANIFEST)
    ):
        raise ValueError("Development output seal is not bound to the current v5 protocol")
    return {
        "metrics": _descriptor(metrics_path),
        "output_seal": _descriptor(seal_path),
    }


def _aggregate(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    metrics = [row["metrics"] for row in rows]
    exact = sum(item["field_exact_correct"] for item in metrics)
    total = sum(item["field_exact_total"] for item in metrics)
    true_positive = sum(item["true_positive"] for item in metrics)
    false_positive = sum(item["false_positive"] for item in metrics)
    false_negative = sum(item["false_negative"] for item in metrics)
    unsupported = sum(item["unsupported_count"] for item in metrics)
    null_total = sum(item["unsupported_opportunities"] for item in metrics)
    precision = safe_divide(true_positive, true_positive + false_positive)
    recall = safe_divide(true_positive, true_positive + false_negative)
    return {
        "output_count": len(rows),
        "schema_valid_count": sum(bool(row["schema_valid"]) for row in rows),
        "schema_valid_rate": safe_divide(sum(bool(row["schema_valid"]) for row in rows), len(rows)),
        "field_exact_correct": exact,
        "field_exact_total": total,
        "field_exact_match": safe_divide(exact, total),
        "nonnull_recalled": true_positive,
        "nonnull_total": true_positive + false_negative,
        "micro_precision": precision,
        "micro_recall": recall,
        "micro_f1": safe_divide(2 * precision * recall, precision + recall),
        "unsupported_count": unsupported,
        "unsupported_opportunities": null_total,
        "unsupported_field_rate": safe_divide(unsupported, null_total),
        "document_id_correct_count": sum(item["document_id_correct"] for item in metrics),
        "complete_report_count": sum(item["complete_report"] for item in metrics),
        "complete_report_accuracy": safe_divide(
            sum(item["complete_report"] for item in metrics), len(metrics)
        ),
    }


def _score_layer(
    preflight: Sequence[Mapping[str, Any]],
    truth_by_document: Mapping[str, Mapping[str, Any]],
    prediction_key: str,
) -> tuple[Dict[str, Any], List[Dict[str, Any]]]:
    validator = Draft202012Validator(read_json(CANONICAL_SCHEMA_PATH))
    rows: List[Dict[str, Any]] = []
    for item in preflight:
        manifest_row = item["row"]
        prediction = item[prediction_key]
        truth = truth_by_document[manifest_row["document_id"]]
        rows.append(
            {
                "document_id": manifest_row["document_id"],
                "semantic_case_id": manifest_row["semantic_case_id"],
                "template_id": manifest_row["template_id"],
                "condition": manifest_row["condition"],
                "schema_valid": not list(validator.iter_errors(prediction)),
                "metrics": compare_prediction(truth, prediction),
                "_truth": truth,
                "_prediction": prediction,
            }
        )
    return _aggregate(rows), rows


def _group_metrics(rows: Sequence[Mapping[str, Any]], key: str) -> List[Dict[str, Any]]:
    groups: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[row[key]].append(row)
    return [{key: value, **_aggregate(groups[value])} for value in sorted(groups)]


def _per_field_metrics(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Dict[str, Any]]:
    output: Dict[str, Dict[str, Any]] = {}
    for path in PRIMARY_FIELD_PATHS:
        exact = true_positive = false_positive = false_negative = 0
        unsupported = null_total = 0
        for row in rows:
            expected = flatten_json(row["_truth"])[path]
            observed = flatten_json(row["_prediction"])[path]
            is_exact = values_exactly_equal(expected, observed)
            exact += int(is_exact)
            if expected is not None and is_exact:
                true_positive += 1
            elif observed is not None:
                false_positive += 1
                false_negative += int(expected is not None)
            elif expected is not None:
                false_negative += 1
            if expected is None:
                null_total += 1
                unsupported += int(observed is not None)
        precision = safe_divide(true_positive, true_positive + false_positive)
        recall = safe_divide(true_positive, true_positive + false_negative)
        output[path] = {
            "exact_correct": exact,
            "total": len(rows),
            "exact_match": safe_divide(exact, len(rows)),
            "nonnull_recalled": true_positive,
            "nonnull_total": true_positive + false_negative,
            "precision": precision,
            "recall": recall,
            "f1": safe_divide(2 * precision * recall, precision + recall),
            "unsupported_count": unsupported,
            "unsupported_opportunities": null_total,
        }
    return output


def _accepted_decision_has_valid_evidence(path: str, decision: Mapping[str, Any]) -> bool:
    if decision.get("accepted") is not True or decision.get("final_value") is None:
        return False
    line_ids = decision.get("evidence_line_ids")
    sections = decision.get("evidence_sections")
    parsed = decision.get("parsed_evidence_values")
    allowed_section = "header" if path == "document_id" else "current"
    return (
        isinstance(line_ids, list)
        and 1 <= len(line_ids) <= 2
        and all(isinstance(value, str) and value for value in line_ids)
        and len(set(line_ids)) == len(line_ids)
        and isinstance(sections, list)
        and len(sections) == len(line_ids)
        and all(section == allowed_section for section in sections)
        and isinstance(parsed, list)
        and bool(parsed)
    )


def _evidence_summary(preflight: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    accepted = valid = history_carryover = history_rejections = 0
    reasons: Counter[str] = Counter()
    for item in preflight:
        log = item["compiler_log"]
        history_rejections += int(log.get("history_evidence_rejection_count", 0))
        for path, decision in log["fields"].items():
            reasons[str(decision.get("reason"))] += 1
            accepted_non_null = (
                path in CLINICAL_FIELD_PATHS
                and decision.get("accepted") is True
                and decision.get("final_value") is not None
            )
            if not accepted_non_null:
                continue
            accepted += 1
            valid += int(_accepted_decision_has_valid_evidence(path, decision))
            sections = decision.get("evidence_sections")
            history_carryover += int(isinstance(sections, list) and "history" in sections)
    return {
        "accepted_non_null_count": accepted,
        "accepted_with_valid_current_report_evidence_count": valid,
        "accepted_with_valid_current_report_evidence_rate": safe_divide(valid, accepted),
        "history_evidence_rejection_count": history_rejections,
        "history_carryover_count": history_carryover,
        "decision_reason_counts": dict(sorted(reasons.items())),
    }


def _history_sentinel_carryover(rows: Sequence[Mapping[str, Any]]) -> int:
    count = 0
    for row in rows:
        prediction = flatten_json(row["_prediction"])
        count += sum(
            prediction[path] is not None
            for path in HISTORY_SENTINEL_FIELDS.get(row["semantic_case_id"], ())
        )
    return count


def _telemetry_summary(
    preflight: Sequence[Mapping[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    summary: Dict[str, Dict[str, Any]] = {}
    for pass_name in ("candidate", "audit"):
        generations = [item[f"{pass_name}_bundle"]["generation"] for item in preflight]
        summary[pass_name] = {
            "output_count": len(generations),
            "token_counts": [value["token_count"] for value in generations],
            "cap_hit_count": sum(bool(value["cap_hit"]) for value in generations),
            "eos_observed_count": sum(bool(value.get("eos_observed")) for value in generations),
        }
    return summary


def _blinded_review_rows(
    manifest: Sequence[Mapping[str, str]],
) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for row in manifest:
        paths = output_paths(row)
        rows.append(
            {
                "document_id": row["document_id"],
                "condition": row["condition"],
                "image_path": row["image_path"],
                "ocr_path": _relative(paths["ocr"]),
                "candidate_bundle_path": _relative(paths["candidate_bundle"]),
                "audit_bundle_path": _relative(paths["audit_bundle"]),
                "normalized_path": _relative(paths["normalized"]),
                "compiler_audit_path": _relative(paths["compiler_audit"]),
                "reviewed": "",
                "artifact_bundle_complete": "",
                "evidence_checked": "",
                "prediction_not_corrected": "",
                "notes": "",
            }
        )
    return rows


def _ensure_blinded_review_checklist(
    manifest: Sequence[Mapping[str, str]],
) -> None:
    if MANUAL_REVIEW_PATH.exists():
        return
    MANUAL_REVIEW_PATH.parent.mkdir(parents=True, exist_ok=True)
    rows = _blinded_review_rows(manifest)
    with MANUAL_REVIEW_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _human_review_status(
    manifest: Sequence[Mapping[str, str]],
    output_seal: Mapping[str, Any],
) -> Dict[str, Any]:
    result = {
        "checklist_path": _relative(MANUAL_REVIEW_PATH),
        "attestation_path": _relative(HUMAN_ATTESTATION_PATH),
        "checklist_complete": False,
        "attestation_valid": False,
        "complete": False,
        "issues": [],
    }
    if not MANUAL_REVIEW_PATH.is_file() or MANUAL_REVIEW_PATH.is_symlink():
        result["issues"].append("Blinded review checklist is missing or unsafe")
        return result
    with MANUAL_REVIEW_PATH.open(newline="", encoding="utf-8") as handle:
        actual = list(csv.DictReader(handle))
    expected = _blinded_review_rows(manifest)
    if len(actual) != len(expected):
        result["issues"].append("Blinded checklist row count is invalid")
        return result
    identity_fields = tuple(
        key
        for key in expected[0]
        if key
        not in {
            "reviewed",
            "artifact_bundle_complete",
            "evidence_checked",
            "prediction_not_corrected",
            "notes",
        }
    )
    for index, (actual_row, expected_row) in enumerate(zip(actual, expected), start=1):
        if any(actual_row.get(key) != expected_row[key] for key in identity_fields):
            result["issues"].append(f"Blinded checklist identity mismatch in row {index}")
        for key in (
            "reviewed",
            "artifact_bundle_complete",
            "evidence_checked",
            "prediction_not_corrected",
        ):
            if actual_row.get(key, "").strip().lower() != "yes":
                result["issues"].append(f"Blinded checklist row {index} has incomplete {key}")
    result["checklist_complete"] = not result["issues"]
    if not result["checklist_complete"]:
        return result
    if not HUMAN_ATTESTATION_PATH.is_file() or HUMAN_ATTESTATION_PATH.is_symlink():
        result["issues"].append("Human attestation is missing or unsafe")
        return result
    try:
        attestation = read_json(HUMAN_ATTESTATION_PATH)
    except (OSError, ValueError) as exc:
        result["issues"].append(f"Human attestation is unreadable: {exc}")
        return result
    expected_attestation = {
        "protocol_version": PROTOCOL_VERSION,
        "split": "formal",
        "reviewed_output_seal_sha256": sha256_file(OUTPUT_SEAL_PATHS["formal"]),
        "reviewed_row_count": SPLIT_OUTPUT_COUNTS["formal"],
        "checklist_path": _relative(MANUAL_REVIEW_PATH),
        "checklist_sha256": sha256_file(MANUAL_REVIEW_PATH),
        "attestation": HUMAN_ATTESTATION_TEXT,
    }
    for key, value in expected_attestation.items():
        if attestation.get(key) != value:
            result["issues"].append(f"Human attestation {key} is invalid")
    reviewer = attestation.get("reviewer_name")
    completed = attestation.get("review_completed_at_utc")
    if not isinstance(reviewer, str) or not reviewer.strip():
        result["issues"].append("Human attestation reviewer_name is missing")
    if not isinstance(completed, str) or not completed.strip():
        result["issues"].append("Human attestation completion timestamp is missing")
    result["attestation_valid"] = not result["issues"]
    result["complete"] = result["checklist_complete"] and result["attestation_valid"]
    return result


def evaluate_split(split: str) -> Dict[str, Any]:
    protocol_lock = verify_protocol_lock()
    manifest = _load_manifest(split)
    _verify_operational_lock(protocol_lock, manifest)
    development_gate = _verify_development_gate_for_formal() if split == "formal" else None
    preflight = [_preflight_row(row, protocol_lock) for row in manifest]
    ledger, ledger_issues = _validate_event_ledgers(split, preflight)
    if ledger_issues and preflight:
        preflight[0]["issues"].extend(f"event ledger: {issue}" for issue in ledger_issues)
    commits = {
        item.get("record", {}).get("repository_commit") for item in preflight if not item["issues"]
    }
    if len(commits) > 1 and preflight:
        preflight[0]["issues"].append("run records were produced from multiple commits")
    if any(item["issues"] for item in preflight):
        payload = _incomplete_payload(split, preflight)
        write_json(METRICS_PATHS[split], payload)
        return payload

    output_seal = _write_output_seal(split, preflight, ledger, development_gate)
    _verify_evaluation_only_corpus(protocol_lock, manifest)

    # This is intentionally the first semantic ground-truth deserialization.
    truth_by_document = {
        row["document_id"]: read_json(ROOT / row["ground_truth_path"]) for row in manifest
    }
    candidate_overall, candidate_rows = _score_layer(preflight, truth_by_document, "candidate")
    final_overall, final_rows = _score_layer(preflight, truth_by_document, "normalized")
    condition_rows = _group_metrics(final_rows, "condition")
    template_rows = _group_metrics(final_rows, "template_id")
    evidence = _evidence_summary(preflight)
    evidence["history_sentinel_carryover_count"] = _history_sentinel_carryover(final_rows)
    evidence["history_carryover_count"] += evidence["history_sentinel_carryover_count"]
    telemetry = _telemetry_summary(preflight)
    threshold = GATE_THRESHOLDS[split]
    condition_exact = {row["condition"]: row["field_exact_correct"] for row in condition_rows}
    criteria = {
        "complete_and_provenance_valid": (len(preflight) == threshold["assigned_input_count"]),
        "candidate_strict_schema_valid": True,
        "audit_strict_schema_valid": True,
        "canonical_schema_valid": (
            final_overall["schema_valid_count"] == threshold["assigned_input_count"]
        ),
        "zero_candidate_cap_hits": (
            telemetry["candidate"]["cap_hit_count"] <= threshold["candidate_cap_hits_maximum"]
        ),
        "zero_audit_cap_hits": (
            telemetry["audit"]["cap_hit_count"] <= threshold["audit_cap_hits_maximum"]
        ),
        "pooled_field_exact_minimum": (
            final_overall["field_exact_total"] == threshold["field_opportunities"]
            and final_overall["field_exact_correct"] >= threshold["pooled_exact_minimum"]
        ),
        "each_condition_field_exact_minimum": all(
            condition_exact.get(condition, -1) >= threshold["per_condition_exact_minimum"]
            for condition in CONDITIONS
        ),
        "nonnull_recall_minimum": (
            final_overall["nonnull_total"] == threshold["ground_truth_non_null_opportunities"]
            and final_overall["nonnull_recalled"] >= threshold["non_null_recall_minimum"]
        ),
        "unsupported_maximum": (
            final_overall["unsupported_opportunities"]
            == threshold["ground_truth_null_opportunities"]
            and final_overall["unsupported_count"] <= threshold["unsupported_maximum"]
        ),
        "accepted_non_null_evidence_100_percent": (
            evidence["accepted_with_valid_current_report_evidence_rate"]
            >= threshold["accepted_non_null_evidence_rate_minimum"]
        ),
        "zero_history_carryover": (
            evidence["history_carryover_count"] <= threshold["history_carryover_maximum"]
        ),
    }
    automated_pass = all(criteria.values())

    human_review: Dict[str, Any] | None = None
    outreach_ready = False
    if split == "formal":
        _ensure_blinded_review_checklist(manifest)
        human_review = _human_review_status(manifest, output_seal)
        outreach_ready = automated_pass and bool(human_review["complete"])

    if split == "development":
        status = "DEVELOPMENT_PASS" if automated_pass else "DEVELOPMENT_FAIL"
    elif outreach_ready:
        status = "PASS"
    elif automated_pass:
        status = "AUTOMATED_PASS_HUMAN_REVIEW_PENDING"
    else:
        status = "FAIL"
    payload = {
        "status": status,
        "protocol_version": PROTOCOL_VERSION,
        "split": split,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "ground_truth_read": True,
        "intended_output_count": threshold["assigned_input_count"],
        "output_count": threshold["assigned_input_count"],
        "provenance_valid_count": threshold["assigned_input_count"],
        "provenance": {
            "output_seal_path": _relative(OUTPUT_SEAL_PATHS[split]),
            "output_seal_sha256": sha256_file(OUTPUT_SEAL_PATHS[split]),
            "repository_commit": output_seal["repository_commit"],
            "development_gate": development_gate,
        },
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "candidate": {
            "strict_schema_valid_count": threshold["assigned_input_count"],
            "overall": candidate_overall,
            "by_condition": _group_metrics(candidate_rows, "condition"),
            "by_template": _group_metrics(candidate_rows, "template_id"),
            "per_field": _per_field_metrics(candidate_rows),
        },
        "blind_audit": {
            "strict_schema_valid_count": threshold["assigned_input_count"],
        },
        "final_pipeline": {
            "overall": final_overall,
            "by_condition": condition_rows,
            "by_template": template_rows,
            "per_field": _per_field_metrics(final_rows),
        },
        "generation_telemetry": telemetry,
        "evidence": evidence,
        "gate_thresholds": threshold,
        "gate_criteria": criteria,
        "automated_gate_pass": automated_pass,
        "pass": automated_pass,
        "human_review": human_review,
        "human_review_complete": (
            bool(human_review["complete"]) if human_review is not None else None
        ),
        "outreach_ready": outreach_ready,
    }
    write_json(METRICS_PATHS[split], payload)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", required=True, choices=("development", "formal"))
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    result = evaluate_split(args.split)
    print(
        f"V5 {args.split} evaluation: {result['status']} "
        f"(outreach_ready={result['outreach_ready']})."
    )
