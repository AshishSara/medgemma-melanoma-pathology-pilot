#!/usr/bin/env python3
"""Evaluate the frozen 20-input formal v4 pilot without test-set peeking."""

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
from run_v4_inference import (
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
    FORMAL_EXECUTION_ATTEMPT_ID,
    MODEL_ID,
    MODEL_REVISION,
    PROTOCOL_LOCK_PATH,
    PROTOCOL_VERSION,
    RANDOM_SEED,
    SCRIPT_PATH,
    V4_MANIFEST,
    output_paths,
    parse_generated_json,
    render_audit_prompt,
    render_candidate_prompt,
    verify_existing,
    verify_protocol_lock,
)
from v3_pipeline import ATOMIC_FIELD_PATHS, CLINICAL_FIELD_PATHS, compile_prediction

DEVELOPMENT_OUTPUT_COUNT = 4
FORMAL_OUTPUT_COUNT = 20
CONDITIONS = ("clean", "ocr_degraded")
METRICS_PATH = ROOT / "results" / "v4" / "formal" / "metrics.json"
OUTPUT_SEAL_PATH = ROOT / "results" / "v4" / "formal" / "output_seal.json"
DEVELOPMENT_METRICS_PATH = ROOT / "results" / "v4" / "development" / "iteration-1" / "metrics.json"

PASS_POOLED_EXACT = 0.90
PASS_CONDITION_EXACT = 0.85
PASS_UNSUPPORTED_RATE = 0.02
HISTORY_SENTINEL_FIELDS = {
    "MEL-108": (
        "breslow_thickness_mm",
        "breslow_qualifier",
        "staging.pT",
    )
}


def _load_formal_manifest() -> List[Dict[str, str]]:
    with V4_MANIFEST.open(newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle) if row["split"] == "formal"]
    if len(rows) != FORMAL_OUTPUT_COUNT:
        raise ValueError(
            f"Expected {FORMAL_OUTPUT_COUNT} formal v4 manifest rows, found {len(rows)}"
        )
    keys = {(row["document_id"], row["condition"]) for row in rows}
    if len(keys) != FORMAL_OUTPUT_COUNT:
        raise ValueError("Formal v4 manifest has duplicate document/condition keys")
    if {row["condition"] for row in rows} != set(CONDITIONS):
        raise ValueError("Formal v4 manifest does not contain both frozen conditions")
    return rows


def _relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def _artifact_descriptor_errors(
    name: str,
    path: Path,
    descriptor: Any,
) -> List[str]:
    if not path.is_file() or path.is_symlink():
        return [f"{name}: missing or unsafe {_relative(path)}"]
    if not isinstance(descriptor, Mapping):
        return [f"{name}: missing run-record descriptor"]
    errors = []
    expected = {
        "path": _relative(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }
    for key, value in expected.items():
        if descriptor.get(key) != value:
            errors.append(f"{name}.{key}: expected {value!r}, found {descriptor.get(key)!r}")
    return errors


def _expected_record_values(
    row: Mapping[str, str],
    protocol_lock: Mapping[str, Any],
) -> Dict[str, Any]:
    safe_manifest_row = {key: value for key, value in row.items() if key != "ground_truth_path"}
    runtime = protocol_lock["runtime"]
    return {
        "attempt_status": "completed",
        "execution_attempt_id": FORMAL_EXECUTION_ATTEMPT_ID,
        "protocol_version": PROTOCOL_VERSION,
        "protocol_lock_path": _relative(PROTOCOL_LOCK_PATH),
        "protocol_lock_sha256": sha256_file(PROTOCOL_LOCK_PATH),
        "protocol_source_commit": protocol_lock["protocol_source_commit"],
        "dataset_version": "pilot-v4",
        "generator_version": row["generator_version"],
        "split": "formal",
        "development_iteration": None,
        "semantic_case_id": row["semantic_case_id"],
        "document_id": row["document_id"],
        "template_id": row["template_id"],
        "condition": row["condition"],
        "manifest_path": _relative(V4_MANIFEST),
        "manifest_sha256": sha256_file(V4_MANIFEST),
        "manifest_row_without_ground_truth": safe_manifest_row,
        "model_id": MODEL_ID,
        "requested_revision": MODEL_REVISION,
        "resolved_revision": MODEL_REVISION,
        "device": runtime["device"],
        "text_vocab_size": runtime["text_vocab_size"],
        "backend": "transformers-direct",
        "requested_dtype": DTYPE_NAME,
        "dtype": runtime["dtype"],
        "accelerate_version": runtime["accelerate_version"],
        "torch_version": runtime["torch_version"],
        "transformers_version": runtime["transformers_version"],
        "lm_format_enforcer_version": runtime["lm_format_enforcer_version"],
        "cuda_version": runtime["cuda_version"],
        "cuda_device_name": runtime["cuda_device_name"],
        "python_version": runtime["python_version"],
        "platform": runtime["platform"],
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
        "inference_script_path": _relative(SCRIPT_PATH),
        "inference_script_sha256": sha256_file(SCRIPT_PATH),
    }


def _read_artifact_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _generation_telemetry_issues(
    *,
    pass_name: str,
    raw_text: str,
    artifact: Any,
    record_value: Any,
) -> List[str]:
    """Validate mechanical generation termination and effective LMFE settings."""

    issues: List[str] = []
    if not isinstance(artifact, Mapping):
        return [f"{pass_name} generation metadata is not an object"]
    if record_value != artifact:
        issues.append(f"{pass_name} generation metadata differs from the run record")
    expected = {
        "pass": pass_name,
        "max_new_tokens": (
            CANDIDATE_MAX_NEW_TOKENS if pass_name == "candidate" else AUDIT_MAX_NEW_TOKENS
        ),
    }
    for key, value in expected.items():
        if artifact.get(key) != value:
            issues.append(
                f"{pass_name} generation {key}: expected {value!r}, found {artifact.get(key)!r}"
            )
    token_count = artifact.get("token_count")
    token_count_valid = not (
        not isinstance(token_count, int)
        or isinstance(token_count, bool)
        or token_count <= 0
        or token_count > expected["max_new_tokens"]
    )
    if not token_count_valid:
        issues.append(f"{pass_name} generation token_count is invalid")
    cap_hit = artifact.get("cap_hit")
    if not isinstance(cap_hit, bool):
        issues.append(f"{pass_name} generation cap_hit is not boolean")
    elif token_count_valid:
        expected_cap_hit = token_count >= expected["max_new_tokens"]
        if cap_hit is not expected_cap_hit:
            issues.append(
                f"{pass_name} generation cap_hit does not match token_count and max_new_tokens"
            )
        if cap_hit:
            issues.append(f"{pass_name} completed generation consumed its configured token maximum")
    prompt_token_count = artifact.get("prompt_token_count")
    if (
        not isinstance(prompt_token_count, int)
        or isinstance(prompt_token_count, bool)
        or prompt_token_count <= 0
    ):
        issues.append(f"{pass_name} generation prompt_token_count is invalid")
    generated_ids = artifact.get("generated_token_ids")
    if (
        not isinstance(generated_ids, list)
        or any(
            not isinstance(token_id, int) or isinstance(token_id, bool) or token_id < 0
            for token_id in generated_ids
        )
        or len(generated_ids) != token_count
    ):
        issues.append(f"{pass_name} generated token IDs are invalid")
    else:
        serialized_ids = json.dumps(
            generated_ids,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("ascii")
        expected_ids_sha = hashlib.sha256(serialized_ids).hexdigest()
        if artifact.get("generated_token_ids_sha256") != expected_ids_sha:
            issues.append(f"{pass_name} generated token-ID digest is invalid")
    eos_token_ids = artifact.get("eos_token_ids")
    if (
        not isinstance(eos_token_ids, list)
        or any(
            not isinstance(token_id, int) or isinstance(token_id, bool) or token_id < 0
            for token_id in eos_token_ids
        )
        or eos_token_ids != sorted(set(eos_token_ids))
    ):
        issues.append(f"{pass_name} EOS token-ID inventory is invalid")
    eos_observed = artifact.get("eos_observed")
    if not isinstance(eos_observed, bool):
        issues.append(f"{pass_name} generation eos_observed is not boolean")
    elif isinstance(generated_ids, list) and isinstance(eos_token_ids, list):
        expected_eos_observed = bool(generated_ids and generated_ids[-1] in set(eos_token_ids))
        if eos_observed is not expected_eos_observed:
            issues.append(f"{pass_name} eos_observed does not match the final generated token")
    elapsed = artifact.get("elapsed_seconds")
    if (
        not isinstance(elapsed, (int, float))
        or isinstance(elapsed, bool)
        or not math.isfinite(float(elapsed))
        or float(elapsed) < 0
    ):
        issues.append(f"{pass_name} generation elapsed_seconds is invalid")
    raw_descriptor = artifact.get("raw_text")
    expected_raw_descriptor = {
        "bytes": len(raw_text.encode("utf-8")),
        "sha256": hashlib.sha256(raw_text.encode("utf-8")).hexdigest(),
    }
    if raw_descriptor != expected_raw_descriptor:
        issues.append(f"{pass_name} generation raw-text descriptor is invalid")
    if artifact.get("decoding") != {
        "do_sample": False,
        "num_beams": 1,
        "skip_special_tokens": True,
    }:
        issues.append(f"{pass_name} generation decoding settings are invalid")

    serializer = artifact.get("serializer")
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
    }
    if not isinstance(serializer, Mapping):
        issues.append(f"{pass_name} effective serializer metadata is not an object")
    else:
        for key, value in expected_serializer.items():
            if serializer.get(key) != value:
                issues.append(
                    f"{pass_name} serializer {key}: expected {value!r}, "
                    f"found {serializer.get(key)!r}"
                )
        max_array = serializer.get("max_json_array_length")
        if max_array != 20:
            issues.append(f"{pass_name} serializer max_json_array_length is invalid")
    return issues


def _preflight_row(
    row: Mapping[str, str],
    protocol_lock: Mapping[str, Any],
) -> Dict[str, Any]:
    """Validate all formal artifacts and provenance without opening ground truth."""

    paths = output_paths(row)
    issues: List[str] = []
    for path in paths.values():
        if not path.is_file() or path.is_symlink():
            issues.append(f"missing or unsafe {_relative(path)}")
    if issues:
        return {"row": dict(row), "paths": paths, "issues": issues}

    try:
        record = read_json(paths["record"])
    except (OSError, ValueError) as exc:
        return {
            "row": dict(row),
            "paths": paths,
            "issues": [f"unreadable run record: {exc}"],
        }
    expected = _expected_record_values(row, protocol_lock)
    for key, value in expected.items():
        if record.get(key) != value:
            issues.append(f"{key}: expected {value!r}, found {record.get(key)!r}")
    record_ocr = record.get("ocr")
    lock_runtime = protocol_lock["runtime"]
    if not isinstance(record_ocr, Mapping):
        issues.append("ocr run metadata is not a mapping")
    else:
        for key in ("pytesseract_version", "tesseract_version"):
            if record_ocr.get(key) != lock_runtime.get(key):
                issues.append(
                    f"ocr.{key}: expected {lock_runtime.get(key)!r}, found {record_ocr.get(key)!r}"
                )
    repository_commit = record.get("repository_commit")
    if not isinstance(repository_commit, str) or not re.fullmatch(
        r"[0-9a-f]{40}", repository_commit
    ):
        issues.append("repository_commit is not a 40-character lowercase SHA")
    execution_attempt_id = record.get("execution_attempt_id")
    if not isinstance(execution_attempt_id, str) or not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9._-]*", execution_attempt_id
    ):
        issues.append("execution_attempt_id is missing or unsafe")

    descriptors = record.get("artifacts")
    if not isinstance(descriptors, Mapping):
        issues.append("artifacts is not a mapping")
        descriptors = {}
    for name, path in paths.items():
        if name != "record":
            issues.extend(_artifact_descriptor_errors(name, path, descriptors.get(name)))

    candidate = audit = normalized = ocr_payload = None
    compiler_log: Dict[str, Any] | None = None
    try:
        candidate_schema = read_json(CANDIDATE_SCHEMA_PATH)
        audit_schema = read_json(AUDIT_SCHEMA_PATH)
        candidate_raw = paths["candidate_raw"].read_text(encoding="utf-8")
        audit_raw = paths["audit_raw"].read_text(encoding="utf-8")
        candidate_generation = _read_artifact_json(paths["candidate_generation"])
        audit_generation = _read_artifact_json(paths["audit_generation"])
        issues.extend(
            _generation_telemetry_issues(
                pass_name="candidate",
                raw_text=candidate_raw,
                artifact=candidate_generation,
                record_value=record.get("candidate_generation"),
            )
        )
        issues.extend(
            _generation_telemetry_issues(
                pass_name="audit",
                raw_text=audit_raw,
                artifact=audit_generation,
                record_value=record.get("audit_generation"),
            )
        )
        candidate = parse_generated_json(candidate_raw, candidate_schema)
        audit = parse_generated_json(audit_raw, audit_schema)
        if candidate != _read_artifact_json(paths["candidate_parsed"]):
            issues.append("candidate raw JSON differs from parsed candidate artifact")
        if audit != _read_artifact_json(paths["audit_parsed"]):
            issues.append("audit raw JSON differs from parsed audit artifact")

        ocr_payload = _read_artifact_json(paths["ocr"])
        if not isinstance(ocr_payload, Mapping) or not isinstance(ocr_payload.get("lines"), list):
            issues.append("OCR artifact has no lines list")
        elif not isinstance(ocr_payload.get("raw_word_data"), Mapping):
            issues.append("OCR artifact has no retained raw word-level data")
        else:
            lines = ocr_payload["lines"]
            rendered_candidate = render_candidate_prompt(lines)
            rendered_audit = render_audit_prompt(lines)
            if paths["candidate_prompt"].read_text(encoding="utf-8") != rendered_candidate:
                issues.append("retained candidate prompt differs from deterministic rendering")
            if paths["audit_prompt"].read_text(encoding="utf-8") != rendered_audit:
                issues.append("retained audit prompt differs from deterministic rendering")
            prompt_expected = {
                "candidate_rendered_prompt": {
                    "bytes": len(rendered_candidate.encode("utf-8")),
                    "sha256": hashlib.sha256(rendered_candidate.encode("utf-8")).hexdigest(),
                },
                "audit_rendered_prompt": {
                    "bytes": len(rendered_audit.encode("utf-8")),
                    "sha256": hashlib.sha256(rendered_audit.encode("utf-8")).hexdigest(),
                },
            }
            for key, value in prompt_expected.items():
                if record.get(key) != value:
                    issues.append(f"{key} does not match reconstructed prompt")

            recompiled, compiler_log = compile_prediction(candidate, audit, lines)
            normalized = _read_artifact_json(paths["normalized"])
            if recompiled != normalized:
                issues.append("normalized artifact differs from deterministic recompilation")
            if "compiler_audit" in paths:
                recorded_compiler_log = _read_artifact_json(paths["compiler_audit"])
                if recorded_compiler_log != compiler_log:
                    issues.append(
                        "compiler audit artifact differs from deterministic recompilation"
                    )
    except Exception as exc:  # Preflight must report, not peek around, malformed artifacts.
        issues.append(f"artifact validation failed: {type(exc).__name__}: {exc}")

    return {
        "row": dict(row),
        "paths": paths,
        "record": record,
        "candidate": candidate,
        "audit": audit,
        "normalized": normalized,
        "ocr": ocr_payload,
        "compiler_log": compiler_log,
        "issues": issues,
    }


def _validate_execution_attempt_ledger(
    preflight: Sequence[Mapping[str, Any]],
    protocol_lock: Mapping[str, Any],
) -> tuple[Dict[str, Any], List[str]]:
    """Validate and hash the complete append-only formal execution ledger."""

    issues: List[str] = []
    attempts_dir = ROOT / "results" / "v4" / "formal" / "execution_attempts"
    if not attempts_dir.is_dir() or attempts_dir.is_symlink():
        return (
            {"attempt_count": 0, "completed_attempt_count": 0, "events": []},
            ["formal execution-attempt ledger is missing or unsafe"],
        )

    records = [item.get("record") for item in preflight if isinstance(item.get("record"), Mapping)]
    if len(records) != FORMAL_OUTPUT_COUNT:
        issues.append(
            f"execution ledger has {len(records)} bound records; expected {FORMAL_OUTPUT_COUNT}"
        )
    record_commits = {record.get("repository_commit") for record in records}
    repository_commit = next(iter(record_commits)) if len(record_commits) == 1 else None
    expected_context = {
        "protocol_version": PROTOCOL_VERSION,
        "protocol_lock_path": _relative(PROTOCOL_LOCK_PATH),
        "protocol_lock_sha256": sha256_file(PROTOCOL_LOCK_PATH),
        "protocol_source_commit": protocol_lock.get("protocol_source_commit"),
        "repository_commit": repository_commit,
        "manifest_path": _relative(V4_MANIFEST),
        "manifest_sha256": sha256_file(V4_MANIFEST),
        "selected_row_count": FORMAL_OUTPUT_COUNT,
        "condition": "all",
    }

    attempts: Dict[str, Dict[str, Dict[str, Any]]] = {}
    event_descriptors: List[Dict[str, Any]] = []
    entries = sorted(attempts_dir.iterdir(), key=lambda path: path.name)
    if not entries:
        issues.append("formal execution-attempt ledger is empty")
    for path in entries:
        match = re.fullmatch(
            r"([A-Za-z0-9][A-Za-z0-9._-]*)\.(started|failed|completed)\.json",
            path.name,
        )
        if match is None or not path.is_file() or path.is_symlink():
            issues.append(f"invalid or unsafe execution-attempt ledger entry: {_relative(path)}")
            continue
        attempt_id, event_name = match.groups()
        try:
            event = _read_artifact_json(path)
        except (OSError, ValueError) as exc:
            issues.append(f"unreadable execution-attempt event {_relative(path)}: {exc}")
            continue
        if not isinstance(event, Mapping):
            issues.append(f"execution-attempt event is not an object: {_relative(path)}")
            continue
        attempt_events = attempts.setdefault(attempt_id, {})
        if event_name in attempt_events:
            issues.append(f"duplicate {event_name} event for execution attempt {attempt_id}")
            continue
        attempt_events[event_name] = dict(event)
        event_descriptors.append(
            {
                "execution_attempt_id": attempt_id,
                "event": event_name,
                "path": _relative(path),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
        if event.get("execution_attempt_id") != attempt_id:
            issues.append(f"{_relative(path)} execution_attempt_id does not match its filename")
        if event.get("status") != event_name:
            issues.append(f"{_relative(path)} status does not match its event type")
        if not isinstance(event.get("timestamp_utc"), str) or not event["timestamp_utc"].strip():
            issues.append(f"{_relative(path)} has no timestamp")
        for key, value in expected_context.items():
            if event.get(key) != value:
                issues.append(
                    f"{_relative(path)} {key}: expected {value!r}, found {event.get(key)!r}"
                )

    records_by_attempt: Counter[str] = Counter()
    for record_index, record in enumerate(records, start=1):
        attempt_id = record.get("execution_attempt_id")
        if record.get("attempt_status") != "completed":
            issues.append(f"formal run record {record_index} is not marked completed")
        if not isinstance(attempt_id, str) or not re.fullmatch(
            r"[A-Za-z0-9][A-Za-z0-9._-]*",
            attempt_id,
        ):
            issues.append(f"formal run record {record_index} has an invalid execution_attempt_id")
            continue
        records_by_attempt[attempt_id] += 1

    completed_attempts = 0
    for attempt_id, event_map in sorted(attempts.items()):
        if "started" not in event_map:
            issues.append(f"execution attempt {attempt_id} has no started event")
        terminals = [name for name in ("failed", "completed") if name in event_map]
        if len(terminals) != 1:
            issues.append(
                f"execution attempt {attempt_id} must have exactly one "
                "failed/completed terminal event"
            )
            continue
        terminal_name = terminals[0]
        terminal = event_map[terminal_name]
        if terminal_name == "failed":
            issues.append(f"one-shot formal execution attempt {attempt_id} ended in failure")
        completed_new = terminal.get("completed_new_row_count")
        verified_existing = terminal.get("verified_existing_row_count")
        if (
            not isinstance(completed_new, int)
            or isinstance(completed_new, bool)
            or completed_new < 0
            or not isinstance(verified_existing, int)
            or isinstance(verified_existing, bool)
            or verified_existing < 0
        ):
            issues.append(f"execution attempt {attempt_id} has invalid row counts")
            continue
        if records_by_attempt[attempt_id] != completed_new:
            issues.append(
                f"execution attempt {attempt_id} completed-row count does not match bound records"
            )
        if completed_new + verified_existing > FORMAL_OUTPUT_COUNT:
            issues.append(f"execution attempt {attempt_id} row counts exceed the formal matrix")
        if terminal_name == "completed":
            completed_attempts += 1
            if completed_new != FORMAL_OUTPUT_COUNT or verified_existing != 0:
                issues.append(
                    f"completed execution attempt {attempt_id} must contain exactly "
                    f"{FORMAL_OUTPUT_COUNT} new rows and zero existing rows"
                )
        elif terminal.get("resume_permitted_for_current_row") is not False:
            issues.append(f"failed execution attempt {attempt_id} violates the no-resume rule")

    if len(attempts) != 1:
        issues.append("formal execution ledger must contain exactly one attempt total")
    if set(attempts) != {FORMAL_EXECUTION_ATTEMPT_ID}:
        issues.append("formal execution ledger does not use the fixed sole-attempt identifier")
    if completed_attempts != 1:
        issues.append("formal execution ledger must contain exactly one completed terminal attempt")
    for attempt_id in records_by_attempt:
        event_map = attempts.get(attempt_id)
        if (
            event_map is None
            or "started" not in event_map
            or not ({"failed", "completed"} & set(event_map))
        ):
            issues.append(
                f"run records reference execution attempt {attempt_id} "
                "without a complete event chain"
            )

    return (
        {
            "attempt_count": len(attempts),
            "completed_attempt_count": completed_attempts,
            "events": sorted(
                event_descriptors,
                key=lambda item: (item["execution_attempt_id"], item["event"]),
            ),
        },
        issues,
    )


def _incomplete_payload(preflight: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    valid = sum(not item["issues"] for item in preflight)
    issues = [
        {
            "document_id": item["row"]["document_id"],
            "condition": item["row"]["condition"],
            "errors": item["issues"],
        }
        for item in preflight
        if item["issues"]
    ]
    return {
        "status": "INCOMPLETE" if valid < FORMAL_OUTPUT_COUNT else "INVALID_PROVENANCE",
        "protocol_version": PROTOCOL_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "ground_truth_read": False,
        "intended_output_count": FORMAL_OUTPUT_COUNT,
        "provenance_valid_count": valid,
        "provenance": {
            "required_count": FORMAL_OUTPUT_COUNT,
            "valid_count": valid,
            "valid_rate": safe_divide(valid, FORMAL_OUTPUT_COUNT),
        },
        "artifact_issues": issues,
        "pass": False,
        "pass_criteria": {
            "provenance_20_of_20": False,
            "schema_100_percent": False,
            "evidence_acceptance_100_percent": False,
            "pooled_exact_at_least_90_percent": False,
            "each_condition_exact_at_least_85_percent": False,
            "unsupported_at_most_2_percent": False,
            "zero_history_carryover": False,
        },
    }


def _write_output_seal(
    preflight: Sequence[Mapping[str, Any]],
    execution_ledger: Mapping[str, Any],
) -> Dict[str, Any]:
    commits = {str(item["record"]["repository_commit"]) for item in preflight}
    if len(commits) != 1:
        raise ValueError("Formal run records do not share one repository commit")
    rows = []
    for item in preflight:
        paths = item["paths"]
        rows.append(
            {
                "document_id": item["row"]["document_id"],
                "condition": item["row"]["condition"],
                "run_record_sha256": sha256_file(paths["record"]),
                "artifacts": item["record"]["artifacts"],
            }
        )
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "repository_commit": commits.pop(),
        "manifest_sha256": sha256_file(V4_MANIFEST),
        "protocol_lock_sha256": sha256_file(PROTOCOL_LOCK_PATH),
        "formal_output_count": len(rows),
        "ground_truth_read": False,
        "rows": sorted(rows, key=lambda row: (row["document_id"], row["condition"])),
        "execution_ledger": dict(execution_ledger),
    }
    write_json(OUTPUT_SEAL_PATH, payload)
    return payload


def _verify_evaluation_only_corpus(
    protocol_lock: Mapping[str, Any],
) -> Dict[str, str]:
    """Verify frozen truth-side bytes only after the 20-output seal exists."""

    frozen = protocol_lock.get("evaluation_only_corpus_artifacts")
    if not isinstance(frozen, Mapping) or not frozen:
        raise ValueError("Protocol lock has no evaluation-only corpus mapping")
    with V4_MANIFEST.open(newline="", encoding="utf-8") as handle:
        all_rows = list(csv.DictReader(handle))
    expected_paths = {"data/v4/cases.csv"}
    for row in all_rows:
        expected_paths.update((row["source_text_path"], row["ground_truth_path"]))
    if set(frozen) != expected_paths:
        raise ValueError("Evaluation-only corpus mapping does not match the frozen manifest")

    verified: Dict[str, str] = {}
    for relative_path in sorted(expected_paths):
        expected_sha = frozen.get(relative_path)
        if (
            not isinstance(expected_sha, str)
            or not re.fullmatch(r"[0-9a-f]{64}", expected_sha)
            or (
                relative_path != "data/v4/cases.csv"
                and not relative_path.startswith(("data/v4/source_text/", "data/v4/ground_truth/"))
            )
            or ".." in Path(relative_path).parts
        ):
            raise ValueError(f"Unsafe evaluation-only corpus entry: {relative_path}")
        path = ROOT / relative_path
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"Missing or unsafe evaluation-only artifact: {relative_path}")
        if sha256_file(path) != expected_sha:
            raise ValueError(f"Evaluation-only corpus hash mismatch: {relative_path}")
        verified[relative_path] = expected_sha
    return verified


def _verify_v3_truth_side_reuse_aliases(
    protocol_lock: Mapping[str, Any],
) -> Dict[str, str]:
    """Hash reused v3 source/truth aliases only after formal outputs are sealed."""

    provenance = protocol_lock.get("v3_boundary_and_reuse_provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError("Protocol lock has no v3 boundary/reuse provenance")
    comparisons = provenance.get("byte_identical_reused_artifacts")
    if not isinstance(comparisons, Mapping):
        raise ValueError("Protocol lock has no v3/v4 reuse comparison mapping")
    frozen_v4 = protocol_lock.get("evaluation_only_corpus_artifacts")
    if not isinstance(frozen_v4, Mapping):
        raise ValueError("Protocol lock has no evaluation-only corpus mapping")

    verified: Dict[str, str] = {}
    for v4_relative, descriptor in comparisons.items():
        if not v4_relative.startswith(("data/v4/source_text/", "data/v4/ground_truth/")):
            continue
        if not isinstance(descriptor, Mapping):
            raise ValueError(f"Invalid v3/v4 reuse descriptor: {v4_relative}")
        v3_relative = descriptor.get("v3_path")
        expected_v3 = v4_relative.replace("/v4/", "/v3/", 1)
        expected_sha = descriptor.get("v3_sha256")
        if (
            v3_relative != expected_v3
            or not isinstance(expected_sha, str)
            or not re.fullmatch(r"[0-9a-f]{64}", expected_sha)
            or descriptor.get("v4_sha256") != expected_sha
            or frozen_v4.get(v4_relative) != expected_sha
        ):
            raise ValueError(f"Invalid v3/v4 truth-side reuse binding: {v4_relative}")
        for relative_path in (v3_relative, v4_relative):
            path = ROOT / relative_path
            if not path.is_file() or path.is_symlink():
                raise ValueError(f"Missing or unsafe reuse alias: {relative_path}")
            if sha256_file(path) != expected_sha:
                raise ValueError(f"Truth-side reuse alias hash mismatch: {relative_path}")
            verified[relative_path] = expected_sha
    return verified


def _aggregate(prediction_rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    metric_rows = [row["metrics"] for row in prediction_rows]
    exact = sum(item["field_exact_correct"] for item in metric_rows)
    total = sum(item["field_exact_total"] for item in metric_rows)
    true_positive = sum(item["true_positive"] for item in metric_rows)
    false_positive = sum(item["false_positive"] for item in metric_rows)
    false_negative = sum(item["false_negative"] for item in metric_rows)
    unsupported = sum(item["unsupported_count"] for item in metric_rows)
    unsupported_opportunities = sum(item["unsupported_opportunities"] for item in metric_rows)
    precision = safe_divide(true_positive, true_positive + false_positive)
    recall = safe_divide(true_positive, true_positive + false_negative)
    return {
        "output_count": len(prediction_rows),
        "schema_valid_count": sum(bool(row["schema_valid"]) for row in prediction_rows),
        "schema_valid_rate": safe_divide(
            sum(bool(row["schema_valid"]) for row in prediction_rows),
            len(prediction_rows),
        ),
        "document_id_correct_count": sum(bool(item["document_id_correct"]) for item in metric_rows),
        "field_exact_correct": exact,
        "field_exact_total": total,
        "field_exact_match": safe_divide(exact, total),
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "micro_precision": precision,
        "micro_recall": recall,
        "micro_f1": safe_divide(2 * precision * recall, precision + recall),
        "unsupported_count": unsupported,
        "unsupported_opportunities": unsupported_opportunities,
        "unsupported_field_rate": safe_divide(unsupported, unsupported_opportunities),
        "complete_report_count": sum(bool(item["complete_report"]) for item in metric_rows),
        "complete_report_accuracy": safe_divide(
            sum(bool(item["complete_report"]) for item in metric_rows),
            len(metric_rows),
        ),
    }


def _score_layer(
    preflight: Sequence[Mapping[str, Any]],
    truth_by_document: Mapping[str, Mapping[str, Any]],
    prediction_key: str,
    schema: Mapping[str, Any],
) -> tuple[Dict[str, Any], List[Dict[str, Any]]]:
    validator = Draft202012Validator(schema)
    rows = []
    for item in preflight:
        row = item["row"]
        prediction = item[prediction_key]
        truth = truth_by_document[row["document_id"]]
        schema_valid = not list(validator.iter_errors(prediction))
        rows.append(
            {
                "document_id": row["document_id"],
                "condition": row["condition"],
                "template_id": row["template_id"],
                "semantic_case_id": row["semantic_case_id"],
                "schema_valid": schema_valid,
                "metrics": compare_prediction(truth, prediction),
                "_truth": truth,
                "_prediction": prediction,
            }
        )
    return _aggregate(rows), rows


def _condition_metrics(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    groups: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[row["condition"]].append(row)
    return [{"condition": condition, **_aggregate(groups[condition])} for condition in CONDITIONS]


def _template_metrics(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    groups: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[row["template_id"]].append(row)
    return [
        {"template_id": template_id, **_aggregate(groups[template_id])}
        for template_id in sorted(groups)
    ]


def _per_field_metrics(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Dict[str, Any]]:
    output: Dict[str, Dict[str, Any]] = {}
    for path in PRIMARY_FIELD_PATHS:
        exact = true_positive = false_positive = false_negative = 0
        unsupported = unsupported_opportunities = 0
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
                unsupported_opportunities += 1
                unsupported += int(observed is not None)
        precision = safe_divide(true_positive, true_positive + false_positive)
        recall = safe_divide(true_positive, true_positive + false_negative)
        output[path] = {
            "exact_correct": exact,
            "total": len(rows),
            "exact_match": safe_divide(exact, len(rows)),
            "true_positive": true_positive,
            "false_positive": false_positive,
            "false_negative": false_negative,
            "precision": precision,
            "recall": recall,
            "f1": safe_divide(2 * precision * recall, precision + recall),
            "unsupported_count": unsupported,
            "unsupported_opportunities": unsupported_opportunities,
            "unsupported_field_rate": safe_divide(unsupported, unsupported_opportunities),
        }
    return output


def _paired_clean_to_degraded(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    pairs: Dict[str, Dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for row in rows:
        pairs[row["document_id"]][row["condition"]] = row
    pair_rows: List[Dict[str, Any]] = []
    for document_id in sorted(pairs):
        pair = pairs[document_id]
        if set(pair) != set(CONDITIONS):
            raise ValueError(f"Incomplete clean/degraded pair for {document_id}")
        clean = pair["clean"]["metrics"]
        degraded = pair["ocr_degraded"]["metrics"]
        clean_rate = clean["field_exact_match"]
        degraded_rate = degraded["field_exact_match"]
        pair_rows.append(
            {
                "document_id": document_id,
                "template_id": pair["clean"]["template_id"],
                "clean_field_exact_match": clean_rate,
                "ocr_degraded_field_exact_match": degraded_rate,
                "degraded_minus_clean": degraded_rate - clean_rate,
            }
        )
    return {
        "pair_count": len(pair_rows),
        "mean_clean_field_exact_match": safe_divide(
            sum(row["clean_field_exact_match"] for row in pair_rows),
            len(pair_rows),
        ),
        "mean_ocr_degraded_field_exact_match": safe_divide(
            sum(row["ocr_degraded_field_exact_match"] for row in pair_rows),
            len(pair_rows),
        ),
        "mean_degraded_minus_clean": safe_divide(
            sum(row["degraded_minus_clean"] for row in pair_rows),
            len(pair_rows),
        ),
        "pairs": pair_rows,
    }


def _normalization_is_structurally_valid(path: str, normalization: Any) -> bool:
    """Validate the compiler's prespecified normalization/linked-evidence shapes."""

    if normalization is None:
        return True
    if not isinstance(normalization, Mapping):
        return False

    normalization_type = normalization.get("type")
    linked_rules = {
        ("document_id", "labeled_header_identity_fallback"): "document_id",
        ("laterality", "linked_field_evidence"): "specimen_site",
        ("breslow_qualifier", "linked_measurement_qualifier"): ("breslow_thickness_mm"),
        ("mitotic_qualifier", "linked_measurement_qualifier"): ("mitotic_rate_per_mm2"),
    }
    source_field = linked_rules.get((path, normalization_type))
    if source_field is not None:
        return (
            set(normalization) == {"type", "source_field"}
            and normalization.get("source_field") == source_field
        )

    if path != "specimen_site" or normalization_type != ("single_confusable_glyph_substitution"):
        return False
    expected_keys = {
        "type",
        "token_index",
        "character_index",
        "candidate_token",
        "ocr_token",
        "candidate_character",
        "ocr_character",
    }
    if set(normalization) != expected_keys:
        return False

    token_index = normalization.get("token_index")
    character_index = normalization.get("character_index")
    candidate_token = normalization.get("candidate_token")
    ocr_token = normalization.get("ocr_token")
    candidate_character = normalization.get("candidate_character")
    ocr_character = normalization.get("ocr_character")
    if (
        not isinstance(token_index, int)
        or isinstance(token_index, bool)
        or token_index < 0
        or not isinstance(character_index, int)
        or isinstance(character_index, bool)
        or character_index < 0
        or not isinstance(candidate_token, str)
        or not isinstance(ocr_token, str)
        or len(candidate_token) != len(ocr_token)
        or len(candidate_token) < 3
        or character_index >= len(candidate_token)
        or not isinstance(candidate_character, str)
        or not isinstance(ocr_character, str)
        or len(candidate_character) != 1
        or len(ocr_character) != 1
    ):
        return False

    differing_positions = [
        index
        for index, (candidate_value, ocr_value) in enumerate(zip(candidate_token, ocr_token))
        if candidate_value != ocr_value
    ]
    allowed_confusions = {
        frozenset(("0", "o")),
        frozenset(("1", "i")),
        frozenset(("1", "l")),
        frozenset(("5", "s")),
        frozenset(("8", "b")),
        frozenset(("g", "q")),
        frozenset(("i", "l")),
    }
    return (
        differing_positions == [character_index]
        and candidate_token[character_index] == candidate_character
        and ocr_token[character_index] == ocr_character
        and frozenset((candidate_character, ocr_character)) in allowed_confusions
    )


def _accepted_decision_has_valid_evidence(
    path: str,
    decision: Mapping[str, Any],
) -> bool:
    """Return whether one explicit acceptance satisfies the evidence contract."""

    if decision.get("accepted") is not True or decision.get("final_value") is None:
        return False
    line_ids = decision.get("evidence_line_ids")
    sections = decision.get("evidence_sections")
    parsed_values = decision.get("parsed_evidence_values")
    allowed_section = "header" if path == "document_id" else "current"
    return (
        isinstance(line_ids, list)
        and 1 <= len(line_ids) <= 2
        and all(isinstance(line_id, str) and bool(line_id) for line_id in line_ids)
        and len(set(line_ids)) == len(line_ids)
        and isinstance(sections, list)
        and len(sections) == len(line_ids)
        and all(section == allowed_section for section in sections)
        and isinstance(parsed_values, list)
        and bool(parsed_values)
        and _normalization_is_structurally_valid(path, decision.get("normalization"))
    )


def _evidence_summary(preflight: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    reasons: Counter[str] = Counter()
    accepted = rejected = history_rejections = history_acceptances = 0
    accepted_with_valid_non_history_evidence = 0
    field_decisions: Counter[str] = Counter()
    for item in preflight:
        log = item["compiler_log"]
        history_rejections += int(log["history_evidence_rejection_count"])
        for path, decision in log["fields"].items():
            reasons[str(decision["reason"])] += 1
            field_decisions[f"{path}:{decision['reason']}"] += 1
            is_accepted_non_null = (
                decision.get("accepted") is True and decision.get("final_value") is not None
            )
            evidence_sections = decision.get("evidence_sections")
            if (
                is_accepted_non_null
                and isinstance(evidence_sections, list)
                and "history" in evidence_sections
            ):
                history_acceptances += 1
            if path not in CLINICAL_FIELD_PATHS:
                continue
            accepted += int(is_accepted_non_null)
            rejected += int(
                decision.get("candidate_value") is not None
                and not decision.get("candidate_omitted", False)
                and not is_accepted_non_null
            )
            if _accepted_decision_has_valid_evidence(path, decision):
                accepted_with_valid_non_history_evidence += 1
    return {
        "accepted_non_null_count": accepted,
        "accepted_with_valid_non_history_evidence_count": (
            accepted_with_valid_non_history_evidence
        ),
        "accepted_with_valid_non_history_evidence_rate": safe_divide(
            accepted_with_valid_non_history_evidence,
            accepted,
        ),
        "rejected_non_null_count": rejected,
        "history_evidence_rejection_count": history_rejections,
        "history_carryover_count": history_acceptances,
        "decision_reason_counts": dict(sorted(reasons.items())),
        "field_decision_counts": dict(sorted(field_decisions.items())),
    }


def _history_sentinel_carryover_count(rows: Sequence[Mapping[str, Any]]) -> int:
    count = 0
    for row in rows:
        paths = HISTORY_SENTINEL_FIELDS.get(row["semantic_case_id"], ())
        prediction = flatten_json(row["_prediction"])
        count += sum(prediction[path] is not None for path in paths)
    return count


def _generation_telemetry_summary(
    preflight: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    summary: Dict[str, Any] = {}
    for pass_name in ("candidate", "audit"):
        values = [read_json(item["paths"][f"{pass_name}_generation"]) for item in preflight]
        token_counts = [int(value["token_count"]) for value in values]
        prompt_counts = [int(value["prompt_token_count"]) for value in values]
        serializer_values = {
            json.dumps(value["serializer"], sort_keys=True, separators=(",", ":"))
            for value in values
        }
        if len(serializer_values) != 1:
            raise ValueError(f"{pass_name} serializer telemetry is not identical across rows")
        summary[pass_name] = {
            "output_count": len(values),
            "token_counts": token_counts,
            "token_count_min": min(token_counts),
            "token_count_max": max(token_counts),
            "token_count_mean": sum(token_counts) / len(token_counts),
            "prompt_token_counts": prompt_counts,
            "prompt_token_count_min": min(prompt_counts),
            "prompt_token_count_max": max(prompt_counts),
            "eos_observed_count": sum(bool(value["eos_observed"]) for value in values),
            "cap_hit_count": sum(bool(value["cap_hit"]) for value in values),
            "generated_token_ids_sha256": [
                {
                    "document_id": item["row"]["document_id"],
                    "condition": item["row"]["condition"],
                    "sha256": value["generated_token_ids_sha256"],
                }
                for item, value in zip(preflight, values)
            ],
            "serializer": json.loads(serializer_values.pop()),
            "decoding": values[0]["decoding"],
        }
    return summary


def evaluate() -> Dict[str, Any]:
    protocol_lock = verify_protocol_lock()
    manifest = _load_formal_manifest()
    preflight = [_preflight_row(row, protocol_lock) for row in manifest]
    execution_ledger, ledger_issues = _validate_execution_attempt_ledger(
        preflight,
        protocol_lock,
    )
    if ledger_issues:
        preflight[0]["issues"].extend(f"execution ledger: {issue}" for issue in ledger_issues)
    completed_commits = {
        item.get("record", {}).get("repository_commit") for item in preflight if not item["issues"]
    }
    if len(completed_commits) > 1:
        preflight[0]["issues"].append(
            "formal run records were produced from more than one repository commit"
        )
    if any(item["issues"] for item in preflight):
        payload = _incomplete_payload(preflight)
        write_json(METRICS_PATH, payload)
        print(
            f"V4 evaluation stopped before ground truth: "
            f"{payload['provenance_valid_count']}/{FORMAL_OUTPUT_COUNT} rows ready."
        )
        return payload

    output_seal = _write_output_seal(preflight, execution_ledger)
    _verify_evaluation_only_corpus(protocol_lock)
    _verify_v3_truth_side_reuse_aliases(protocol_lock)

    # This is intentionally the first semantic ground-truth deserialization.
    truth_by_document = {
        row["document_id"]: read_json(ROOT / row["ground_truth_path"]) for row in manifest
    }
    canonical_schema = read_json(CANONICAL_SCHEMA_PATH)
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
    candidate_conditions = _condition_metrics(candidate_rows)
    final_conditions = _condition_metrics(final_rows)
    candidate_templates = _template_metrics(candidate_rows)
    final_templates = _template_metrics(final_rows)
    evidence = _evidence_summary(preflight)
    evidence["history_sentinel_carryover_count"] = _history_sentinel_carryover_count(final_rows)
    evidence["history_carryover_count"] += evidence["history_sentinel_carryover_count"]
    generation_telemetry = _generation_telemetry_summary(preflight)

    criteria = {
        "provenance_20_of_20": True,
        "schema_100_percent": final_overall["schema_valid_rate"] == 1.0,
        "evidence_acceptance_100_percent": (
            evidence["accepted_with_valid_non_history_evidence_rate"] == 1.0
        ),
        "pooled_exact_at_least_90_percent": (
            final_overall["field_exact_match"] >= PASS_POOLED_EXACT
        ),
        "each_condition_exact_at_least_85_percent": all(
            item["field_exact_match"] >= PASS_CONDITION_EXACT for item in final_conditions
        ),
        "unsupported_at_most_2_percent": (
            final_overall["unsupported_field_rate"] <= PASS_UNSUPPORTED_RATE
        ),
        "zero_history_carryover": evidence["history_carryover_count"] == 0,
    }
    passed = all(criteria.values())
    payload = {
        "status": "PASS" if passed else "FAIL",
        "protocol_version": PROTOCOL_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "ground_truth_read": True,
        "intended_output_count": FORMAL_OUTPUT_COUNT,
        "provenance_valid_count": FORMAL_OUTPUT_COUNT,
        "provenance": {
            "required_count": FORMAL_OUTPUT_COUNT,
            "valid_count": FORMAL_OUTPUT_COUNT,
            "valid_rate": 1.0,
            "output_seal_path": _relative(OUTPUT_SEAL_PATH),
            "output_seal_sha256": sha256_file(OUTPUT_SEAL_PATH),
            "repository_commit": output_seal["repository_commit"],
        },
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "atomic_field_count_excluding_document_id": len(PRIMARY_FIELD_PATHS),
        "compiler_atomic_field_count_including_document_id": len(ATOMIC_FIELD_PATHS),
        "candidate_raw": {
            "raw_json_parse_count": FORMAL_OUTPUT_COUNT,
            "raw_json_parse_rate": 1.0,
            "candidate_schema_valid_count": FORMAL_OUTPUT_COUNT,
            "candidate_schema_valid_rate": 1.0,
            "overall": candidate_overall,
            "by_condition": candidate_conditions,
            "by_template": candidate_templates,
            "per_field": _per_field_metrics(candidate_rows),
            "paired_clean_to_degraded": _paired_clean_to_degraded(candidate_rows),
        },
        "blind_audit_raw": {
            "raw_json_parse_count": FORMAL_OUTPUT_COUNT,
            "raw_json_parse_rate": 1.0,
            "audit_schema_valid_count": FORMAL_OUTPUT_COUNT,
            "audit_schema_valid_rate": 1.0,
        },
        "generation_telemetry": generation_telemetry,
        "final_pipeline": {
            "overall": final_overall,
            "by_condition": final_conditions,
            "by_template": final_templates,
            "per_field": _per_field_metrics(final_rows),
            "paired_clean_to_degraded": _paired_clean_to_degraded(final_rows),
        },
        "evidence": evidence,
        "pass_thresholds": {
            "formal_outputs": FORMAL_OUTPUT_COUNT,
            "schema_valid_rate": 1.0,
            "accepted_non_null_evidence_rate": 1.0,
            "pooled_field_exact_match": PASS_POOLED_EXACT,
            "each_condition_field_exact_match": PASS_CONDITION_EXACT,
            "unsupported_field_rate_maximum": PASS_UNSUPPORTED_RATE,
            "history_carryover_count": 0,
        },
        "pass_criteria": criteria,
        "pass": passed,
    }
    write_json(METRICS_PATH, payload)
    print(
        f"V4 evaluation {payload['status']}: "
        f"exact={final_overall['field_exact_match']:.1%}, "
        f"unsupported={final_overall['unsupported_field_rate']:.1%}, "
        f"schema={final_overall['schema_valid_rate']:.1%}."
    )
    return payload


def _load_development_manifest() -> List[Dict[str, str]]:
    with V4_MANIFEST.open(newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle) if row["split"] == "development"]
    if len(rows) != DEVELOPMENT_OUTPUT_COUNT:
        raise ValueError(
            f"Expected {DEVELOPMENT_OUTPUT_COUNT} development v4 manifest rows, found {len(rows)}"
        )
    if len({(row["document_id"], row["condition"]) for row in rows}) != len(rows):
        raise ValueError("Development v4 manifest has duplicate document/condition keys")
    return rows


def _development_preflight_row(row: Mapping[str, str]) -> Dict[str, Any]:
    paths = output_paths(row, development_iteration=1)
    issues: List[str] = []
    try:
        verify_existing(paths, row, 1, None)
        record = read_json(paths["record"])
        candidate_raw = paths["candidate_raw"].read_text(encoding="utf-8")
        audit_raw = paths["audit_raw"].read_text(encoding="utf-8")
        candidate = parse_generated_json(candidate_raw, read_json(CANDIDATE_SCHEMA_PATH))
        audit = parse_generated_json(audit_raw, read_json(AUDIT_SCHEMA_PATH))
        normalized = read_json(paths["normalized"])
        compiler_log = read_json(paths["compiler_audit"])
        issues.extend(
            _generation_telemetry_issues(
                pass_name="candidate",
                raw_text=candidate_raw,
                artifact=read_json(paths["candidate_generation"]),
                record_value=record.get("candidate_generation"),
            )
        )
        issues.extend(
            _generation_telemetry_issues(
                pass_name="audit",
                raw_text=audit_raw,
                artifact=read_json(paths["audit_generation"]),
                record_value=record.get("audit_generation"),
            )
        )
        ocr_payload = read_json(paths["ocr"])
        lines = ocr_payload.get("lines") if isinstance(ocr_payload, Mapping) else None
        if not isinstance(lines, list):
            issues.append("OCR artifact has no lines list")
        else:
            recompiled, recomputed_compiler_log = compile_prediction(candidate, audit, lines)
            if recompiled != normalized:
                issues.append("normalized artifact differs from deterministic recompilation")
            if recomputed_compiler_log != compiler_log:
                issues.append("compiler audit differs from deterministic recompilation")
    except Exception as exc:
        issues.append(f"development artifact validation failed: {type(exc).__name__}: {exc}")
        record = {}
        candidate = {}
        normalized = {}
        compiler_log = {}
    return {
        "row": dict(row),
        "paths": paths,
        "record": record,
        "candidate": candidate,
        "normalized": normalized,
        "compiler_log": compiler_log,
        "issues": issues,
    }


def evaluate_development() -> Dict[str, Any]:
    """Evaluate the single four-input serializer smoke; accuracy is descriptive."""

    rows = _load_development_manifest()
    preflight = [_development_preflight_row(row) for row in rows]
    provenance_valid_count = sum(not item["issues"] for item in preflight)
    commits = {
        item["record"].get("repository_commit")
        for item in preflight
        if not item["issues"] and isinstance(item["record"], Mapping)
    }
    if len(commits) != 1 and preflight:
        preflight[0]["issues"].append("development run records do not share one repository commit")
        provenance_valid_count = sum(not item["issues"] for item in preflight)

    if any(item["issues"] for item in preflight):
        payload = {
            "status": "DEVELOPMENT_INCOMPLETE",
            "protocol_version": PROTOCOL_VERSION,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "iteration": 1,
            "output_count": DEVELOPMENT_OUTPUT_COUNT,
            "provenance_valid_count": provenance_valid_count,
            "ground_truth_read": False,
            "artifact_issues": [
                {
                    "document_id": item["row"]["document_id"],
                    "condition": item["row"]["condition"],
                    "errors": item["issues"],
                }
                for item in preflight
                if item["issues"]
            ],
            "pass": False,
        }
        write_json(DEVELOPMENT_METRICS_PATH, payload)
        return payload

    # Development truth is intentionally available for descriptive smoke scoring.
    truth_by_document = {
        row["document_id"]: read_json(ROOT / row["ground_truth_path"]) for row in rows
    }
    canonical_schema = read_json(CANONICAL_SCHEMA_PATH)
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
    evidence = _evidence_summary(preflight)
    telemetry = {
        pass_name: {
            "cap_hit_count": sum(
                bool(read_json(item["paths"][f"{pass_name}_generation"])["cap_hit"])
                for item in preflight
            ),
            "eos_observed_count": sum(
                bool(read_json(item["paths"][f"{pass_name}_generation"])["eos_observed"])
                for item in preflight
            ),
            "token_counts": [
                read_json(item["paths"][f"{pass_name}_generation"])["token_count"]
                for item in preflight
            ],
        }
        for pass_name in ("candidate", "audit")
    }
    criteria = {
        "provenance_4_of_4": provenance_valid_count == DEVELOPMENT_OUTPUT_COUNT,
        "candidate_strict_json_schema_4_of_4": True,
        "audit_strict_json_schema_4_of_4": True,
        "canonical_schema_100_percent": final_overall["schema_valid_rate"] == 1.0,
        "evidence_acceptance_100_percent": (
            evidence["accepted_with_valid_non_history_evidence_rate"] == 1.0
        ),
        "zero_candidate_cap_hits": telemetry["candidate"]["cap_hit_count"] == 0,
        "zero_audit_cap_hits": telemetry["audit"]["cap_hit_count"] == 0,
        "zero_unsupported_fields": final_overall["unsupported_count"] == 0,
        "zero_history_carryover": evidence["history_carryover_count"] == 0,
    }
    passed = all(criteria.values())
    payload = {
        "status": "DEVELOPMENT_PASS" if passed else "DEVELOPMENT_FAIL",
        "protocol_version": PROTOCOL_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "iteration": 1,
        "output_count": DEVELOPMENT_OUTPUT_COUNT,
        "provenance_valid_count": provenance_valid_count,
        "ground_truth_read": True,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "candidate": {
            "overall": candidate_overall,
            "by_condition": _condition_metrics(candidate_rows),
            "by_template": _template_metrics(candidate_rows),
            "per_field": _per_field_metrics(candidate_rows),
            "paired_clean_to_degraded": _paired_clean_to_degraded(candidate_rows),
        },
        "final_pipeline": {
            "overall": final_overall,
            "by_condition": _condition_metrics(final_rows),
            "by_template": _template_metrics(final_rows),
            "per_field": _per_field_metrics(final_rows),
            "paired_clean_to_degraded": _paired_clean_to_degraded(final_rows),
        },
        "generation_telemetry": telemetry,
        "evidence": evidence,
        "smoke_criteria": criteria,
        "accuracy_selection_note": (
            "Exact-match accuracy is reported descriptively only; this four-input "
            "serialization stress smoke is not used to estimate or tune formal accuracy."
        ),
        "pass": passed,
    }
    write_json(DEVELOPMENT_METRICS_PATH, payload)
    base = DEVELOPMENT_METRICS_PATH.parent
    run_record_hashes = {
        f"{row['document_id']}:{row['condition']}": sha256_file(
            output_paths(row, development_iteration=1)["record"]
        )
        for row in rows
    }
    write_json(
        base / "iteration_record.json",
        {
            "status": payload["status"],
            "protocol_version": PROTOCOL_VERSION,
            "iteration": 1,
            "selection_rule": (
                "Prespecified serialization completion gate only; accuracy is descriptive."
            ),
            "manifest_sha256": sha256_file(V4_MANIFEST),
            "candidate_schema_sha256": sha256_file(CANDIDATE_SCHEMA_PATH),
            "audit_schema_sha256": sha256_file(AUDIT_SCHEMA_PATH),
            "canonical_schema_sha256": sha256_file(CANONICAL_SCHEMA_PATH),
            "repository_commit": commits.pop(),
            "metrics_path": _relative(DEVELOPMENT_METRICS_PATH),
            "metrics_sha256": sha256_file(DEVELOPMENT_METRICS_PATH),
            "run_record_sha256": run_record_hashes,
        },
    )
    print(
        f"V4 development smoke {payload['status']}: "
        f"exact={final_overall['field_exact_match']:.1%}, "
        f"cap_hits={telemetry['candidate']['cap_hit_count'] + telemetry['audit']['cap_hit_count']}."
    )
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("development", "formal"), default="formal")
    parser.add_argument("--development-iteration", type=int, choices=(1,))
    args = parser.parse_args()
    if args.split == "development" and args.development_iteration != 1:
        parser.error("development evaluation requires --development-iteration 1")
    if args.split == "formal" and args.development_iteration is not None:
        parser.error("--development-iteration is not valid for formal evaluation")
    return args


if __name__ == "__main__":
    cli_args = parse_args()
    if cli_args.split == "development":
        evaluate_development()
    else:
        evaluate()
