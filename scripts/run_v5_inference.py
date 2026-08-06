#!/usr/bin/env python3
"""Run the locked v5 MedGemma 1.5 qualification or formal split."""

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
)
from v3_pipeline import compile_prediction, sectionize_ocr_lines

PROTOCOL_VERSION = "fresh-confirmatory-v5"
PROTOCOL_IDENTIFIER = "fresh-heldout-confirmatory-pilot-v5"
FORMAL_EXECUTION_ATTEMPT_ID = "formal-attempt"
DEVELOPMENT_ATTEMPT_ID = "qualification-attempt"
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

V5_MANIFEST = ROOT / "data" / "v5" / "report_manifest.csv"
PROTOCOL_LOCK_PATH = ROOT / "data" / "v5" / "protocol_lock.json"
CANDIDATE_PROMPT_PATH = ROOT / "prompts" / "v3" / "candidate_prompt.txt"
AUDIT_PROMPT_PATH = ROOT / "prompts" / "v3" / "audit_prompt.txt"
CANDIDATE_SCHEMA_PATH = ROOT / "schema" / "v3" / "candidate.schema.json"
AUDIT_SCHEMA_PATH = ROOT / "schema" / "v3" / "audit.schema.json"
CANONICAL_SCHEMA_PATH = ROOT / "schema" / "extraction.schema.json"
COMPILER_PATH = ROOT / "scripts" / "v3_pipeline.py"
SCRIPT_PATH = Path(__file__).resolve()
OCR_MARKER = "{{OCR_LINES}}"
REQUIRED_LOCKED_SOURCE_PATHS = (
    "data/v5/report_manifest.csv",
    "data/v5/runtime_profile.json",
    "docs/protocol_v5.md",
    "prompts/v3/audit_prompt.txt",
    "prompts/v3/candidate_prompt.txt",
    "pyproject.toml",
    "requirements/colab-v5.txt",
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


def committed_artifact_binding_errors(
    root: Path,
    source_commit: str,
    locked_artifacts: Mapping[str, Any],
) -> List[str]:
    """Bind every declared digest to both the prospective source commit and HEAD."""

    errors: List[str] = []
    for relative_path, expected_sha in sorted(locked_artifacts.items()):
        if not _is_canonical_repo_relative_path(relative_path) or not _valid_sha256(expected_sha):
            continue
        for revision, label in (
            (source_commit, "protocol source commit"),
            ("HEAD", "execution HEAD"),
        ):
            completed = subprocess.run(
                ["git", "cat-file", "blob", f"{revision}:{relative_path}"],
                cwd=root,
                check=False,
                capture_output=True,
            )
            if completed.returncode != 0:
                errors.append(f"locked artifact is absent from {label}: {relative_path}")
            elif sha256_bytes(completed.stdout) != expected_sha:
                errors.append(f"locked artifact differs from {label}: {relative_path}")
    return errors


def prospective_lock_origin_errors(root: Path, source_commit: str, lock_path: str) -> List[str]:
    """Require the lock to have been created only after the prospective source commit."""

    completed = subprocess.run(
        ["git", "cat-file", "-e", f"{source_commit}:{lock_path}"],
        cwd=root,
        check=False,
        capture_output=True,
    )
    if completed.returncode == 0:
        return ["protocol lock already existed in the protocol source commit"]
    return []


def verify_protocol_lock() -> Dict[str, Any]:
    """Fail closed unless the committed pre-inference v5 lock matches every source byte."""

    if (
        not PROTOCOL_LOCK_PATH.is_file()
        or PROTOCOL_LOCK_PATH.is_symlink()
        or not V5_MANIFEST.is_file()
        or V5_MANIFEST.is_symlink()
    ):
        raise SystemExit("The committed v5 protocol lock or manifest is missing or unsafe")
    lock = read_json(PROTOCOL_LOCK_PATH)
    if not isinstance(lock, Mapping):
        raise SystemExit("The v5 protocol lock is not an object")

    errors: List[str] = []
    expected_scalars = {
        "status": "frozen_pre_inference",
        "protocol_version": PROTOCOL_VERSION,
        "protocol_identifier": PROTOCOL_IDENTIFIER,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "resolved_model_revision": MODEL_REVISION,
        "dtype": DTYPE_NAME,
        "random_seed": RANDOM_SEED,
        "development_output_count": 4,
        "formal_output_count": 20,
        "development_case_ids": ["MEL-190"],
        "formal_case_ids": ["MEL-201", "MEL-202", "MEL-203", "MEL-204", "MEL-205"],
        "template_ids": ["A", "B"],
        "conditions": list(CONDITIONS),
    }
    for key, value in expected_scalars.items():
        if lock.get(key) != value:
            errors.append(f"{key}: expected {value!r}, found {lock.get(key)!r}")

    expected_configuration = {
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
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
        "audit_serializer_force_json_field_order": (AUDIT_SERIALIZER_FORCE_JSON_FIELD_ORDER),
        "audit_serializer_max_consecutive_whitespaces": (
            AUDIT_SERIALIZER_MAX_CONSECUTIVE_WHITESPACES
        ),
        "tesseract_language": TESSERACT_LANGUAGE,
        "tesseract_config": TESSERACT_CONFIG,
        "tesseract_timeout_seconds": OCR_TIMEOUT_SECONDS,
        "image_preprocessing": "none",
        "assistant_prefill": False,
        "do_sample": False,
        "num_beams": 1,
        "candidate_calls_per_input": 1,
        "audit_calls_per_input": 1,
        "development_output_count": 4,
        "formal_output_count": 20,
        "manifest_order": "exact CSV row order; candidate then blind audit per input",
        "resume_policy": (
            "reuse valid persisted responses; exactly one unchanged infrastructure retry "
            "only after call-start without a durable response"
        ),
    }
    if lock.get("fixed_configuration") != expected_configuration:
        errors.append("fixed_configuration differs from the prospective v5 configuration")

    source_commit = lock.get("protocol_source_commit")
    if not isinstance(source_commit, str) or re.fullmatch(r"[0-9a-f]{40}", source_commit) is None:
        errors.append("protocol_source_commit is not a 40-character lowercase SHA")
    else:
        ancestor = subprocess.run(
            ["git", "merge-base", "--is-ancestor", source_commit, "HEAD"],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        if ancestor.returncode != 0:
            errors.append("protocol_source_commit is not an ancestor of execution HEAD")

    try:
        committed_lock = subprocess.check_output(
            ["git", "cat-file", "blob", f"HEAD:{relative(PROTOCOL_LOCK_PATH)}"],
            cwd=ROOT,
        )
    except (OSError, subprocess.CalledProcessError):
        errors.append("protocol_lock.json is not committed in execution HEAD")
    else:
        if committed_lock != PROTOCOL_LOCK_PATH.read_bytes():
            errors.append("protocol_lock.json bytes differ from execution HEAD")

    locked_sources = lock.get("locked_source_files")
    if not isinstance(locked_sources, Mapping) or set(locked_sources) != set(
        REQUIRED_LOCKED_SOURCE_PATHS
    ):
        errors.append("locked source inventory differs from the required v5 inventory")
        locked_sources = {}
    for relative_path in REQUIRED_LOCKED_SOURCE_PATHS:
        path = ROOT / relative_path
        expected_sha = locked_sources.get(relative_path)
        if (
            not _valid_sha256(expected_sha)
            or not path.is_file()
            or path.is_symlink()
            or sha256_file(path) != expected_sha
        ):
            errors.append(f"locked source hash mismatch: {relative_path}")
    runtime_profile_path = ROOT / "data" / "v5" / "runtime_profile.json"
    if (
        lock.get("runtime_profile_path") != relative(runtime_profile_path)
        or not runtime_profile_path.is_file()
        or runtime_profile_path.is_symlink()
        or lock.get("runtime_profile") != read_json(runtime_profile_path)
    ):
        errors.append("runtime profile is missing, unsafe, or differs from the lock")

    expected_gate_thresholds = {
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
    if lock.get("gate_thresholds") != expected_gate_thresholds:
        errors.append("gate_thresholds differ from the prospective count-based gates")

    expected_denominators = {
        "development": {
            "assigned_input_count": 4,
            "field_opportunities": 64,
            "ground_truth_non_null_opportunities": 52,
            "ground_truth_null_opportunities": 12,
            "by_condition": {
                condition: {
                    "assigned_input_count": 2,
                    "field_opportunities": 32,
                    "ground_truth_non_null_opportunities": 26,
                    "ground_truth_null_opportunities": 6,
                }
                for condition in CONDITIONS
            },
        },
        "formal": {
            "assigned_input_count": 20,
            "field_opportunities": 320,
            "ground_truth_non_null_opportunities": 196,
            "ground_truth_null_opportunities": 124,
            "by_condition": {
                condition: {
                    "assigned_input_count": 10,
                    "field_opportunities": 160,
                    "ground_truth_non_null_opportunities": 98,
                    "ground_truth_null_opportunities": 62,
                }
                for condition in CONDITIONS
            },
        },
    }
    if lock.get("denominators") != expected_denominators:
        errors.append("denominators differ from the frozen v5 corpus counts")

    artifact_mappings = {
        "corpus_artifacts": 49,
        "evaluation_only_corpus_artifacts": 25,
        "freshness_reference_artifacts": 3,
    }
    verified_mappings: Dict[str, Mapping[str, Any]] = {}
    for mapping_name, expected_count in artifact_mappings.items():
        mapping = lock.get(mapping_name)
        if not isinstance(mapping, Mapping) or len(mapping) != expected_count:
            errors.append(f"{mapping_name} does not contain exactly {expected_count} entries")
            continue
        verified_mappings[mapping_name] = mapping
        for relative_path, expected_sha in mapping.items():
            path = ROOT / str(relative_path)
            if (
                not isinstance(relative_path, str)
                or not _is_canonical_repo_relative_path(relative_path)
                or not _valid_sha256(expected_sha)
                or not path.is_file()
                or path.is_symlink()
                or sha256_file(path) != expected_sha
            ):
                errors.append(f"{mapping_name} hash mismatch: {relative_path}")

    corpus = verified_mappings.get("corpus_artifacts", {})
    manifest_sha = corpus.get(relative(V5_MANIFEST))
    if not _valid_sha256(manifest_sha) or manifest_sha != sha256_file(V5_MANIFEST):
        errors.append("locked manifest hash is invalid")
    if lock.get("requirements_path") != "requirements/colab-v5.txt":
        errors.append("requirements_path differs from the fixed v5 package pins")

    committed_artifacts: Dict[str, Any] = {}
    for mapping_name, mapping in (
        ("locked_source_files", locked_sources),
        *verified_mappings.items(),
    ):
        for relative_path, expected_sha in mapping.items():
            if not _is_canonical_repo_relative_path(relative_path) or not _valid_sha256(
                expected_sha
            ):
                continue
            prior_sha = committed_artifacts.get(relative_path)
            if prior_sha is not None and prior_sha != expected_sha:
                errors.append(
                    f"conflicting hashes for {relative_path} across locked mappings "
                    f"(including {mapping_name})"
                )
            committed_artifacts[relative_path] = expected_sha
    if isinstance(source_commit, str) and re.fullmatch(r"[0-9a-f]{40}", source_commit):
        errors.extend(
            committed_artifact_binding_errors(
                ROOT,
                source_commit,
                committed_artifacts,
            )
        )
        errors.extend(
            prospective_lock_origin_errors(
                ROOT,
                source_commit,
                relative(PROTOCOL_LOCK_PATH),
            )
        )

    semantic_freshness = lock.get("semantic_freshness")
    expected_fingerprint_fields = [
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
    ]
    if (
        not isinstance(semantic_freshness, Mapping)
        or semantic_freshness.get("fields") != expected_fingerprint_fields
        or semantic_freshness.get("method")
        != (
            "exact ordered 16-string tuple comparison; empty strings retained; "
            "no normalization or fuzzy matching"
        )
        or semantic_freshness.get("collision_count") != 0
        or semantic_freshness.get("case_id_collision_count") != 0
        or semantic_freshness.get("report_date_collision_count") != 0
        or semantic_freshness.get("v5_row_count") != 6
        or semantic_freshness.get("v5_unique_fingerprint_count") != 6
        or set(semantic_freshness.get("v5_fingerprint_sha256", {}))
        != {"MEL-190", "MEL-201", "MEL-202", "MEL-203", "MEL-204", "MEL-205"}
    ):
        errors.append("semantic_freshness does not match the mechanical v5 rule")

    corpus_inventory = lock.get("corpus_inventory")
    expected_operational_paths = sorted(corpus)
    evaluation_mapping = verified_mappings.get("evaluation_only_corpus_artifacts", {})
    expected_evaluation_paths = sorted(evaluation_mapping)
    if corpus_inventory != {
        "operational_count": 49,
        "operational_paths": expected_operational_paths,
        "evaluation_only_count": 25,
        "evaluation_only_paths": expected_evaluation_paths,
        "total_count": 74,
    }:
        errors.append("corpus_inventory differs from the frozen artifact mappings")

    if errors:
        detail = "\n".join(f"- {error}" for error in errors)
        raise SystemExit(f"V5 protocol lock validation failed:\n{detail}")
    return dict(lock)


def load_manifest(split: str, condition: str) -> List[Dict[str, str]]:
    with V5_MANIFEST.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    selected = [
        row
        for row in rows
        if row["split"] == split and (condition == "all" or row["condition"] == condition)
    ]
    if not selected:
        raise SystemExit(f"No v5 manifest rows found for split={split!r}, condition={condition!r}")
    if any(row["dataset_version"] != "pilot-v5" for row in selected):
        raise SystemExit("The selected manifest contains a non-v5 dataset row")
    return selected


def output_paths(row: Mapping[str, str]) -> Dict[str, Path]:
    if row["split"] not in SPLITS:
        raise ValueError(f"Unsupported v5 split: {row['split']!r}")
    base = ROOT / "results" / "v5" / row["split"]
    condition = row["condition"]
    document_id = row["document_id"]
    return {
        "ocr": base / "ocr" / condition / f"{document_id}.json",
        "candidate_prompt": (
            base / "rendered_prompts" / "candidate" / condition / f"{document_id}.txt"
        ),
        "audit_prompt": (base / "rendered_prompts" / "audit" / condition / f"{document_id}.txt"),
        "candidate_bundle": (
            base / "response_bundles" / "candidate" / MODEL_SLUG / condition / f"{document_id}.json"
        ),
        "audit_bundle": (
            base / "response_bundles" / "audit" / MODEL_SLUG / condition / f"{document_id}.json"
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
        raise SystemExit("V5 OCR requires `uv sync --extra inference`.") from exc

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
        raise SystemExit("V5 inference requires `uv sync --extra inference`.") from exc

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
            "cuda_device_count": torch.cuda.device_count(),
            "cuda_device_name": (
                torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
            ),
            "gpu_memory_mib": (
                int(torch.cuda.get_device_properties(0).total_memory / (1024 * 1024))
                if torch.cuda.is_available()
                else None
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
    runtime = protocol_lock.get("runtime_profile")
    if not isinstance(runtime, Mapping):
        raise RuntimeError("Protocol lock has no runtime_profile mapping")
    packages = runtime.get("packages")
    if not isinstance(packages, Mapping):
        raise RuntimeError("Protocol runtime profile has no package mapping")
    expected_backend = {
        "resolved_revision": MODEL_REVISION,
        "random_seed": RANDOM_SEED,
        "accelerate_version": packages.get("accelerate"),
        "torch_version": runtime.get("torch_version"),
        "transformers_version": packages.get("transformers"),
        "lm_format_enforcer_version": packages.get("lm-format-enforcer"),
        "cuda_version": runtime.get("cuda_version"),
        "cuda_device_count": runtime.get("cuda_device_count"),
        "cuda_device_name": runtime.get("cuda_device_name"),
        "gpu_memory_mib": runtime.get("gpu_memory_mib"),
        "dtype": "torch.bfloat16",
    }
    mismatches = [
        f"{key}: lock {value!r}, runtime {backend_metadata.get(key)!r}"
        for key, value in expected_backend.items()
        if backend_metadata.get(key) != value
    ]
    device = backend_metadata.get("device")
    if not isinstance(device, str) or not re.fullmatch(r"cuda(?::0)?", device):
        mismatches.append(f"device: expected the sole CUDA device, runtime {device!r}")
    if ocr_metadata is not None:
        expected_ocr = {
            "pytesseract_version": packages.get("pytesseract"),
            "tesseract_version": str(runtime.get("tesseract_version", "")).removeprefix(
                "tesseract "
            ),
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
        raise RuntimeError(f"Frozen v5 runtime does not match the protocol lock:\n{detail}")


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


def write_exclusive_json(path: Path, payload: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(dict(payload), indent=2, sort_keys=True, ensure_ascii=True) + "\n"
    with path.open("x", encoding="utf-8") as handle:
        handle.write(serialized)
        handle.flush()
        os.fsync(handle.fileno())
    return path


def execution_attempt_id(split: str) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    return f"{split}-{timestamp}-pid{os.getpid()}"


def write_execution_event(
    *,
    split: str,
    attempt_id: str,
    event_name: str,
    repository_sha: str,
    protocol_lock: Mapping[str, Any],
    payload: Mapping[str, Any] | None = None,
) -> Path:
    if event_name not in {"started", "completed", "failed"}:
        raise ValueError(f"Unsupported execution event: {event_name}")
    path = (
        ROOT / "results" / "v5" / split / "execution_attempts" / f"{attempt_id}.{event_name}.json"
    )
    return write_exclusive_json(
        path,
        {
            "protocol_version": PROTOCOL_VERSION,
            "protocol_lock_path": relative(PROTOCOL_LOCK_PATH),
            "protocol_lock_sha256": sha256_file(PROTOCOL_LOCK_PATH),
            "protocol_source_commit": protocol_lock["protocol_source_commit"],
            "split": split,
            "execution_attempt_id": attempt_id,
            "repository_commit": repository_sha,
            "event": event_name,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            **dict(payload or {}),
        },
    )


def call_event_path(
    row: Mapping[str, str],
    pass_name: str,
    call_ordinal: int,
    event_name: str,
) -> Path:
    if pass_name not in {"candidate", "audit"}:
        raise ValueError(f"Unsupported pass: {pass_name}")
    if event_name not in {"started", "no_response", "response_persisted"}:
        raise ValueError(f"Unsupported call event: {event_name}")
    return (
        ROOT
        / "results"
        / "v5"
        / row["split"]
        / "call_events"
        / pass_name
        / row["condition"]
        / row["document_id"]
        / f"{call_ordinal:03d}.{event_name}.json"
    )


def call_request_identity(
    *,
    row: Mapping[str, str],
    pass_name: str,
    image_path: Path,
    rendered_prompt: str,
    schema: Mapping[str, Any],
    schema_path: Path,
    max_new_tokens: int,
    force_json_field_order: bool,
    max_consecutive_whitespaces: int,
) -> Dict[str, Any]:
    if pass_name not in {"candidate", "audit"}:
        raise ValueError(f"Unsupported pass: {pass_name}")
    candidate = pass_name == "candidate"
    if relative(image_path) != row["image_path"] or sha256_file(image_path) != row["image_sha256"]:
        raise RuntimeError("Generation image differs from the locked manifest row")
    locked_schema = read_json(schema_path)
    if not isinstance(locked_schema, Mapping) or dict(schema) != dict(locked_schema):
        raise RuntimeError("In-memory generation schema differs from its locked file")
    return {
        "image_path": relative(image_path),
        "image_sha256": sha256_file(image_path),
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "requested_dtype": DTYPE_NAME,
        "do_sample": False,
        "num_beams": 1,
        "random_seed": RANDOM_SEED,
        "assistant_prefill": False,
        "prompt_template_path": relative(CANDIDATE_PROMPT_PATH if candidate else AUDIT_PROMPT_PATH),
        "prompt_template_sha256": sha256_file(
            CANDIDATE_PROMPT_PATH if candidate else AUDIT_PROMPT_PATH
        ),
        "rendered_prompt": text_metadata(rendered_prompt),
        "response_schema_path": relative(schema_path),
        "response_schema_sha256": sha256_file(schema_path),
        "max_new_tokens": max_new_tokens,
        "serializer_force_json_field_order": force_json_field_order,
        "serializer_max_consecutive_whitespaces": max_consecutive_whitespaces,
    }


def write_call_event(
    *,
    row: Mapping[str, str],
    pass_name: str,
    call_ordinal: int,
    event_name: str,
    row_attempt_id: str,
    launch_attempt_id: str,
    repository_sha: str,
    protocol_lock: Mapping[str, Any],
    request_identity: Mapping[str, Any],
    payload: Mapping[str, Any] | None = None,
) -> Path:
    return write_exclusive_json(
        call_event_path(row, pass_name, call_ordinal, event_name),
        {
            "protocol_version": PROTOCOL_VERSION,
            "protocol_lock_path": relative(PROTOCOL_LOCK_PATH),
            "protocol_lock_sha256": sha256_file(PROTOCOL_LOCK_PATH),
            "protocol_source_commit": protocol_lock["protocol_source_commit"],
            "dataset_version": "pilot-v5",
            "split": row["split"],
            "semantic_case_id": row["semantic_case_id"],
            "document_id": row["document_id"],
            "condition": row["condition"],
            "pass": pass_name,
            "call_ordinal": call_ordinal,
            "execution_attempt_id": row_attempt_id,
            "launch_attempt_id": launch_attempt_id,
            "repository_commit": repository_sha,
            "call_request": dict(request_identity),
            "event": event_name,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            **dict(payload or {}),
        },
    )


def _read_call_chain(
    row: Mapping[str, str],
    pass_name: str,
    repository_sha: str,
    protocol_lock: Mapping[str, Any],
    request_identity: Mapping[str, Any],
) -> Dict[int, Dict[str, Mapping[str, Any]]]:
    directory = call_event_path(row, pass_name, 1, "started").parent
    if not directory.exists():
        return {}
    if not directory.is_dir() or directory.is_symlink():
        raise RuntimeError(f"Unsafe call-event directory: {relative(directory)}")
    chains: Dict[int, Dict[str, Mapping[str, Any]]] = {}
    pattern = re.compile(r"([0-9]{3})\.(started|no_response|response_persisted)\.json")
    for path in sorted(directory.iterdir()):
        match = pattern.fullmatch(path.name)
        if match is None or not path.is_file() or path.is_symlink():
            raise RuntimeError(f"Unexpected call-event entry: {relative(path)}")
        ordinal = int(match.group(1))
        event_name = match.group(2)
        event = read_json(path)
        if not isinstance(event, Mapping):
            raise RuntimeError(f"Call event is not an object: {relative(path)}")
        expected = {
            "protocol_version": PROTOCOL_VERSION,
            "protocol_lock_path": relative(PROTOCOL_LOCK_PATH),
            "protocol_lock_sha256": sha256_file(PROTOCOL_LOCK_PATH),
            "protocol_source_commit": protocol_lock["protocol_source_commit"],
            "dataset_version": "pilot-v5",
            "split": row["split"],
            "semantic_case_id": row["semantic_case_id"],
            "document_id": row["document_id"],
            "condition": row["condition"],
            "pass": pass_name,
            "call_ordinal": ordinal,
            "event": event_name,
            "repository_commit": repository_sha,
            "call_request": dict(request_identity),
        }
        if any(event.get(key) != value for key, value in expected.items()):
            raise RuntimeError(f"Call-event identity mismatch: {relative(path)}")
        chains.setdefault(ordinal, {})[event_name] = event
    return chains


def _next_call_ordinal(
    *,
    row: Mapping[str, str],
    pass_name: str,
    row_attempt_id: str,
    launch_attempt_id: str,
    repository_sha: str,
    protocol_lock: Mapping[str, Any],
    request_identity: Mapping[str, Any],
) -> int:
    chains = _read_call_chain(
        row,
        pass_name,
        repository_sha,
        protocol_lock,
        request_identity,
    )
    if not chains:
        return 1
    if sorted(chains) != list(range(1, len(chains) + 1)) or len(chains) > 2:
        raise RuntimeError(f"Invalid {pass_name} call ordinals for {row['document_id']}")
    last_ordinal = max(chains)
    for ordinal, events in chains.items():
        terminals = {"no_response", "response_persisted"} & set(events)
        if "started" not in events or len(terminals) > 1:
            raise RuntimeError(
                f"Invalid {pass_name} call chain for {row['document_id']} ordinal {ordinal}"
            )
        start = events["started"]
        for terminal_name in terminals:
            terminal = events[terminal_name]
            if any(
                not isinstance(start.get(key), str) or terminal.get(key) != start.get(key)
                for key in ("execution_attempt_id", "launch_attempt_id")
            ):
                raise RuntimeError(
                    f"Inconsistent {pass_name} call provenance for "
                    f"{row['document_id']} ordinal {ordinal}"
                )
        if ordinal < last_ordinal and (
            set(events) != {"started", "no_response"}
            or events["no_response"].get("retry_permitted") is not True
        ):
            raise RuntimeError(f"Invalid prior {pass_name} retry chain for {row['document_id']}")
    last = chains[last_ordinal]
    if "response_persisted" in last:
        raise RuntimeError(
            f"{pass_name} has a persisted-response event but no durable response bundle"
        )
    if "no_response" not in last:
        start_event = last["started"]
        start_attempt_id = start_event.get("execution_attempt_id")
        start_launch_id = start_event.get("launch_attempt_id")
        if not isinstance(start_attempt_id, str) or not isinstance(start_launch_id, str):
            raise RuntimeError(
                f"Invalid {pass_name} call-start provenance for {row['document_id']}"
            )
        write_call_event(
            row=row,
            pass_name=pass_name,
            call_ordinal=last_ordinal,
            event_name="no_response",
            row_attempt_id=start_attempt_id,
            launch_attempt_id=start_launch_id,
            repository_sha=repository_sha,
            protocol_lock=protocol_lock,
            request_identity=request_identity,
            payload={
                "error_type": "RuntimeInterrupted",
                "error_message": (
                    "Prior call-start had no durable response bundle or terminal event"
                ),
                "retry_permitted": last_ordinal == 1,
            },
        )
    no_response = chains[last_ordinal].get("no_response")
    if last_ordinal >= 2:
        raise RuntimeError(
            f"{pass_name} exhausted its single infrastructure retry for {row['document_id']}"
        )
    if isinstance(no_response, Mapping) and no_response.get("retry_permitted") is not True:
        raise RuntimeError(
            f"{pass_name} ended in a non-retryable no-response failure for {row['document_id']}"
        )
    return 2


def _validate_existing_bundle(
    *,
    bundle_path: Path,
    row: Mapping[str, str],
    pass_name: str,
    rendered_prompt: str,
    schema_path: Path,
    repository_sha: str,
    protocol_lock: Mapping[str, Any],
    request_identity: Mapping[str, Any],
) -> Dict[str, Any]:
    if not bundle_path.is_file() or bundle_path.is_symlink():
        raise RuntimeError(f"Unsafe response bundle: {relative(bundle_path)}")
    bundle = read_json(bundle_path)
    if not isinstance(bundle, Mapping):
        raise RuntimeError(f"Response bundle is not an object: {relative(bundle_path)}")
    expected = {
        "protocol_version": PROTOCOL_VERSION,
        "protocol_lock_path": relative(PROTOCOL_LOCK_PATH),
        "protocol_lock_sha256": sha256_file(PROTOCOL_LOCK_PATH),
        "protocol_source_commit": protocol_lock["protocol_source_commit"],
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
        "repository_commit": repository_sha,
        "prompt_template_path": relative(
            CANDIDATE_PROMPT_PATH if pass_name == "candidate" else AUDIT_PROMPT_PATH
        ),
        "prompt_template_sha256": sha256_file(
            CANDIDATE_PROMPT_PATH if pass_name == "candidate" else AUDIT_PROMPT_PATH
        ),
        "rendered_prompt": text_metadata(rendered_prompt),
        "response_schema_path": relative(schema_path),
        "response_schema_sha256": sha256_file(schema_path),
        "call_request": dict(request_identity),
    }
    mismatches = [key for key, value in expected.items() if bundle.get(key) != value]
    if mismatches:
        raise RuntimeError(
            f"Existing {pass_name} bundle identity mismatch: {', '.join(mismatches)}"
        )
    raw_text = bundle.get("raw_text")
    if not isinstance(raw_text, str) or bundle.get("raw_text_descriptor") != text_metadata(
        raw_text
    ):
        raise RuntimeError(f"Existing {pass_name} bundle raw-text descriptor is invalid")
    ordinal = bundle.get("call_ordinal")
    if not isinstance(ordinal, int) or isinstance(ordinal, bool) or ordinal not in {1, 2}:
        raise RuntimeError(f"Existing {pass_name} bundle call ordinal is invalid")
    chains = _read_call_chain(
        row,
        pass_name,
        repository_sha,
        protocol_lock,
        request_identity,
    )
    expected_chain = (
        {1: {"started", "response_persisted"}}
        if ordinal == 1
        else {
            1: {"started", "no_response"},
            2: {"started", "response_persisted"},
        }
    )
    if set(chains) != set(expected_chain) or any(
        set(chains[chain_ordinal]) != event_names
        for chain_ordinal, event_names in expected_chain.items()
    ):
        raise RuntimeError(
            f"Existing {pass_name} bundle lacks its exact contiguous call-event chain"
        )
    for chain_ordinal, events in chains.items():
        start = events["started"]
        terminal_name = "response_persisted" if "response_persisted" in events else "no_response"
        terminal = events[terminal_name]
        for provenance_key in ("execution_attempt_id", "launch_attempt_id"):
            provenance_value = start.get(provenance_key)
            if (
                not isinstance(provenance_value, str)
                or terminal.get(provenance_key) != provenance_value
            ):
                raise RuntimeError(
                    f"Existing {pass_name} bundle has inconsistent ordinal "
                    f"{chain_ordinal} call-event provenance"
                )
        if terminal_name == "no_response" and terminal.get("retry_permitted") is not True:
            raise RuntimeError(f"Existing {pass_name} bundle follows a non-retryable first call")
    binding = bundle.get("response_persisted_event")
    expected_event_path = call_event_path(row, pass_name, ordinal, "response_persisted")
    if not isinstance(binding, Mapping) or binding != artifact_metadata(expected_event_path):
        raise RuntimeError(f"Existing {pass_name} bundle event binding is invalid")
    response_event = read_json(expected_event_path)
    expected_event = {
        "protocol_version": PROTOCOL_VERSION,
        "protocol_lock_path": relative(PROTOCOL_LOCK_PATH),
        "protocol_lock_sha256": sha256_file(PROTOCOL_LOCK_PATH),
        "protocol_source_commit": protocol_lock["protocol_source_commit"],
        "dataset_version": "pilot-v5",
        "split": row["split"],
        "semantic_case_id": row["semantic_case_id"],
        "document_id": row["document_id"],
        "condition": row["condition"],
        "pass": pass_name,
        "call_ordinal": ordinal,
        "execution_attempt_id": bundle.get("execution_attempt_id"),
        "repository_commit": repository_sha,
        "call_request": dict(request_identity),
        "event": "response_persisted",
        "raw_text_descriptor": bundle.get("raw_text_descriptor"),
        "generation": bundle.get("generation"),
    }
    if not isinstance(response_event, Mapping) or any(
        response_event.get(key) != value for key, value in expected_event.items()
    ):
        raise RuntimeError(f"Existing {pass_name} bundle differs from its persisted-response event")
    return dict(bundle)


def obtain_response_bundle(
    *,
    row: Mapping[str, str],
    pass_name: str,
    bundle_path: Path,
    image_path: Path,
    rendered_prompt: str,
    schema: Mapping[str, Any],
    schema_path: Path,
    max_new_tokens: int,
    force_json_field_order: bool,
    max_consecutive_whitespaces: int,
    backend: Backend,
    repository_sha: str,
    protocol_lock: Mapping[str, Any],
    row_attempt_id: str,
    launch_attempt_id: str,
) -> Dict[str, Any]:
    request_identity = call_request_identity(
        row=row,
        pass_name=pass_name,
        image_path=image_path,
        rendered_prompt=rendered_prompt,
        schema=schema,
        schema_path=schema_path,
        max_new_tokens=max_new_tokens,
        force_json_field_order=force_json_field_order,
        max_consecutive_whitespaces=max_consecutive_whitespaces,
    )
    if bundle_path.exists():
        return _validate_existing_bundle(
            bundle_path=bundle_path,
            row=row,
            pass_name=pass_name,
            rendered_prompt=rendered_prompt,
            schema_path=schema_path,
            repository_sha=repository_sha,
            protocol_lock=protocol_lock,
            request_identity=request_identity,
        )

    ordinal = _next_call_ordinal(
        row=row,
        pass_name=pass_name,
        row_attempt_id=row_attempt_id,
        launch_attempt_id=launch_attempt_id,
        repository_sha=repository_sha,
        protocol_lock=protocol_lock,
        request_identity=request_identity,
    )
    while ordinal <= 2:
        write_call_event(
            row=row,
            pass_name=pass_name,
            call_ordinal=ordinal,
            event_name="started",
            row_attempt_id=row_attempt_id,
            launch_attempt_id=launch_attempt_id,
            repository_sha=repository_sha,
            protocol_lock=protocol_lock,
            request_identity=request_identity,
        )
        try:
            result = constrained_generate(
                backend,
                image_path,
                rendered_prompt,
                schema,
                max_new_tokens=max_new_tokens,
                force_json_field_order=force_json_field_order,
                max_consecutive_whitespaces=max_consecutive_whitespaces,
            )
        except BaseException as exc:
            retryable = _retryable_no_response(exc)
            write_call_event(
                row=row,
                pass_name=pass_name,
                call_ordinal=ordinal,
                event_name="no_response",
                row_attempt_id=row_attempt_id,
                launch_attempt_id=launch_attempt_id,
                repository_sha=repository_sha,
                protocol_lock=protocol_lock,
                request_identity=request_identity,
                payload={
                    "error_type": type(exc).__name__,
                    "error_message": redact_error_message(exc),
                    "retry_permitted": retryable and ordinal == 1,
                },
            )
            if ordinal >= 2 or not retryable:
                raise
            ordinal += 1
            continue

        metadata = generation_metadata(result, pass_name=pass_name)
        response_event_path = write_call_event(
            row=row,
            pass_name=pass_name,
            call_ordinal=ordinal,
            event_name="response_persisted",
            row_attempt_id=row_attempt_id,
            launch_attempt_id=launch_attempt_id,
            repository_sha=repository_sha,
            protocol_lock=protocol_lock,
            request_identity=request_identity,
            payload={
                "raw_text_descriptor": text_metadata(result.raw_text),
                "generation": metadata,
            },
        )
        bundle = {
            "protocol_version": PROTOCOL_VERSION,
            "protocol_lock_path": relative(PROTOCOL_LOCK_PATH),
            "protocol_lock_sha256": sha256_file(PROTOCOL_LOCK_PATH),
            "protocol_source_commit": protocol_lock["protocol_source_commit"],
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
            "repository_commit": repository_sha,
            "execution_attempt_id": row_attempt_id,
            "call_ordinal": ordinal,
            "prompt_template_path": relative(
                CANDIDATE_PROMPT_PATH if pass_name == "candidate" else AUDIT_PROMPT_PATH
            ),
            "prompt_template_sha256": sha256_file(
                CANDIDATE_PROMPT_PATH if pass_name == "candidate" else AUDIT_PROMPT_PATH
            ),
            "rendered_prompt": text_metadata(rendered_prompt),
            "response_schema_path": relative(schema_path),
            "response_schema_sha256": sha256_file(schema_path),
            "call_request": request_identity,
            "raw_text": result.raw_text,
            "raw_text_descriptor": text_metadata(result.raw_text),
            "generation": metadata,
            "response_persisted_event": artifact_metadata(response_event_path),
        }
        write_json_atomic(bundle_path, bundle)
        return bundle
    raise AssertionError("Unreachable call retry state")


def _write_or_verify_json(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists():
        if not path.is_file() or path.is_symlink() or read_json(path) != value:
            raise RuntimeError(f"Existing deterministic artifact differs: {relative(path)}")
        return
    write_json_atomic(path, value)


def _write_or_verify_text(path: Path, value: str) -> None:
    if path.exists():
        if not path.is_file() or path.is_symlink() or path.read_text(encoding="utf-8") != value:
            raise RuntimeError(f"Existing deterministic artifact differs: {relative(path)}")
        return
    write_text_atomic(path, value)


def load_or_create_ocr(
    *,
    row: Mapping[str, str],
    paths: Mapping[str, Path],
    image_path: Path,
    backend: Backend,
    protocol_lock: Mapping[str, Any],
) -> Tuple[List[Dict[str, Any]], str, str, Dict[str, Any]]:
    ocr_path = paths["ocr"]
    prompt_paths = (paths["candidate_prompt"], paths["audit_prompt"])
    existence = [ocr_path.exists(), *(path.exists() for path in prompt_paths)]
    if any(existence) and not all(existence):
        raise RuntimeError(f"Partial OCR/prompt checkpoint for {row['document_id']}")
    if all(existence):
        if any(path.is_symlink() or not path.is_file() for path in (ocr_path, *prompt_paths)):
            raise RuntimeError(f"Unsafe OCR/prompt checkpoint for {row['document_id']}")
        payload = read_json(ocr_path)
        if (
            not isinstance(payload, Mapping)
            or payload.get("document_id") != row["document_id"]
            or payload.get("condition") != row["condition"]
            or not isinstance(payload.get("lines"), list)
            or not isinstance(payload.get("raw_word_data"), Mapping)
        ):
            raise RuntimeError(f"Invalid OCR checkpoint for {row['document_id']}")
        lines = validate_sectionized_lines(payload["lines"])
        candidate_prompt = render_candidate_prompt(lines)
        audit_prompt = render_audit_prompt(lines)
        _write_or_verify_text(paths["candidate_prompt"], candidate_prompt)
        _write_or_verify_text(paths["audit_prompt"], audit_prompt)
        ocr_metadata = {
            key: value
            for key, value in payload.items()
            if key not in {"document_id", "condition", "raw_word_data", "lines"}
        }
        verify_runtime_against_lock(protocol_lock, backend.metadata, ocr_metadata)
        return lines, candidate_prompt, audit_prompt, ocr_metadata

    lines, raw_word_data, ocr_metadata = run_ocr(image_path)
    verify_runtime_against_lock(protocol_lock, backend.metadata, ocr_metadata)
    candidate_prompt = render_candidate_prompt(lines)
    audit_prompt = render_audit_prompt(lines)
    write_json_atomic(
        paths["ocr"],
        {
            "document_id": row["document_id"],
            "condition": row["condition"],
            **ocr_metadata,
            "raw_word_data": raw_word_data,
            "lines": lines,
        },
    )
    write_text_atomic(paths["candidate_prompt"], candidate_prompt)
    write_text_atomic(paths["audit_prompt"], audit_prompt)
    return lines, candidate_prompt, audit_prompt, ocr_metadata


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


def _retryable_no_response(error: BaseException) -> bool:
    if isinstance(error, (TimeoutError, ConnectionError, OSError)):
        return True
    if not isinstance(error, RuntimeError):
        return False
    message = str(error).lower()
    return any(
        marker in message
        for marker in (
            "cuda",
            "cublas",
            "cudnn",
            "out of memory",
            "connection",
            "disconnected",
            "device",
            "runtime",
        )
    )


def verify_complete_record(
    paths: Mapping[str, Path],
    row: Mapping[str, str],
    repository_sha: str,
    protocol_lock: Mapping[str, Any],
) -> None:
    record_path = paths["record"]
    if not record_path.is_file() or record_path.is_symlink():
        raise RuntimeError(f"Existing row lacks a safe run record: {relative(record_path)}")
    record = read_json(record_path)
    expected = {
        "attempt_status": "completed",
        "protocol_version": PROTOCOL_VERSION,
        "protocol_lock_path": relative(PROTOCOL_LOCK_PATH),
        "protocol_lock_sha256": sha256_file(PROTOCOL_LOCK_PATH),
        "protocol_source_commit": protocol_lock["protocol_source_commit"],
        "repository_commit": repository_sha,
        "dataset_version": "pilot-v5",
        "split": row["split"],
        "semantic_case_id": row["semantic_case_id"],
        "document_id": row["document_id"],
        "condition": row["condition"],
        "image_path": row["image_path"],
        "image_sha256": row["image_sha256"],
        "model_id": MODEL_ID,
        "requested_revision": MODEL_REVISION,
        "resolved_revision": MODEL_REVISION,
    }
    if not isinstance(record, Mapping) or any(
        record.get(key) != value for key, value in expected.items()
    ):
        raise RuntimeError(f"Existing run-record identity mismatch: {relative(record_path)}")
    artifacts = record.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise RuntimeError(f"Existing run record has no artifact map: {relative(record_path)}")
    for name, path in paths.items():
        if name == "record":
            continue
        if (
            not path.is_file()
            or path.is_symlink()
            or artifacts.get(name) != artifact_metadata(path)
        ):
            raise RuntimeError(f"Existing row artifact mismatch: {relative(path)}")


def _existing_row_attempt_id(
    paths: Mapping[str, Path],
    repository_sha: str,
) -> str | None:
    attempt_ids = set()
    for key in ("candidate_bundle", "audit_bundle"):
        path = paths[key]
        if not path.exists():
            continue
        if not path.is_file() or path.is_symlink():
            raise RuntimeError(f"Unsafe existing response bundle: {relative(path)}")
        bundle = read_json(path)
        if (
            not isinstance(bundle, Mapping)
            or bundle.get("repository_commit") != repository_sha
            or not isinstance(bundle.get("execution_attempt_id"), str)
        ):
            raise RuntimeError(f"Existing response bundle has invalid provenance: {relative(path)}")
        attempt_ids.add(bundle["execution_attempt_id"])
    if len(attempt_ids) > 1:
        raise RuntimeError("Candidate and audit bundles have different row attempt IDs")
    return next(iter(attempt_ids)) if attempt_ids else None


def run_row(
    *,
    row: Mapping[str, str],
    backend: Backend,
    repository_sha: str,
    candidate_schema: Mapping[str, Any],
    audit_schema: Mapping[str, Any],
    canonical_schema: Mapping[str, Any],
    protocol_lock: Mapping[str, Any],
    launch_attempt_id: str,
) -> bool:
    paths = output_paths(row)
    if paths["record"].exists():
        verify_complete_record(paths, row, repository_sha, protocol_lock)
        print(f"ROW {row['document_id']} {row['condition']} verified_complete")
        return False

    image_path = validate_manifest_image(row)
    ocr_lines, candidate_prompt, audit_prompt, ocr_metadata = load_or_create_ocr(
        row=row,
        paths=paths,
        image_path=image_path,
        backend=backend,
        protocol_lock=protocol_lock,
    )
    row_attempt_id = _existing_row_attempt_id(paths, repository_sha) or launch_attempt_id

    candidate_bundle = obtain_response_bundle(
        row=row,
        pass_name="candidate",
        bundle_path=paths["candidate_bundle"],
        image_path=image_path,
        rendered_prompt=candidate_prompt,
        schema=candidate_schema,
        schema_path=CANDIDATE_SCHEMA_PATH,
        max_new_tokens=CANDIDATE_MAX_NEW_TOKENS,
        force_json_field_order=CANDIDATE_SERIALIZER_FORCE_JSON_FIELD_ORDER,
        max_consecutive_whitespaces=(CANDIDATE_SERIALIZER_MAX_CONSECUTIVE_WHITESPACES),
        backend=backend,
        repository_sha=repository_sha,
        protocol_lock=protocol_lock,
        row_attempt_id=row_attempt_id,
        launch_attempt_id=launch_attempt_id,
    )
    candidate_generation = candidate_bundle["generation"]
    if candidate_generation.get("cap_hit") is not False:
        raise RuntimeError("Candidate generation reached or ambiguously reported its token cap")
    candidate = parse_generated_json(candidate_bundle["raw_text"], candidate_schema)
    _write_or_verify_json(paths["candidate_parsed"], candidate)

    audit_bundle = obtain_response_bundle(
        row=row,
        pass_name="audit",
        bundle_path=paths["audit_bundle"],
        image_path=image_path,
        rendered_prompt=audit_prompt,
        schema=audit_schema,
        schema_path=AUDIT_SCHEMA_PATH,
        max_new_tokens=AUDIT_MAX_NEW_TOKENS,
        force_json_field_order=AUDIT_SERIALIZER_FORCE_JSON_FIELD_ORDER,
        max_consecutive_whitespaces=AUDIT_SERIALIZER_MAX_CONSECUTIVE_WHITESPACES,
        backend=backend,
        repository_sha=repository_sha,
        protocol_lock=protocol_lock,
        row_attempt_id=row_attempt_id,
        launch_attempt_id=launch_attempt_id,
    )
    if audit_bundle.get("execution_attempt_id") != row_attempt_id:
        raise RuntimeError("Candidate and audit response bundles have different attempt IDs")
    audit_generation = audit_bundle["generation"]
    if audit_generation.get("cap_hit") is not False:
        raise RuntimeError("Audit generation reached or ambiguously reported its token cap")
    audit = parse_generated_json(audit_bundle["raw_text"], audit_schema)
    _write_or_verify_json(paths["audit_parsed"], audit)

    prediction, compiler_audit = compile_prediction(candidate, audit, ocr_lines)
    if not isinstance(prediction, dict) or not isinstance(compiler_audit, dict):
        raise TypeError("compile_prediction() must return prediction and audit dictionaries")
    canonical_errors = sorted(
        Draft202012Validator(canonical_schema).iter_errors(prediction),
        key=lambda error: list(error.absolute_path),
    )
    if canonical_errors:
        details = "; ".join(
            f"{'.'.join(str(part) for part in error.absolute_path) or '<root>'}: {error.message}"
            for error in canonical_errors
        )
        raise ValueError(f"Compiled prediction violates the canonical schema: {details}")
    _write_or_verify_json(paths["normalized"], prediction)
    _write_or_verify_json(paths["compiler_audit"], compiler_audit)

    artifacts = {name: artifact_metadata(path) for name, path in paths.items() if name != "record"}
    safe_manifest_row = {key: value for key, value in row.items() if key != "ground_truth_path"}
    record = {
        "attempt_status": "completed",
        "execution_attempt_id": row_attempt_id,
        "run_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_version": PROTOCOL_VERSION,
        "protocol_lock_path": relative(PROTOCOL_LOCK_PATH),
        "protocol_lock_sha256": sha256_file(PROTOCOL_LOCK_PATH),
        "protocol_source_commit": protocol_lock["protocol_source_commit"],
        "repository_commit": repository_sha,
        "dataset_version": "pilot-v5",
        "generator_version": row["generator_version"],
        "split": row["split"],
        "semantic_case_id": row["semantic_case_id"],
        "document_id": row["document_id"],
        "template_id": row["template_id"],
        "condition": row["condition"],
        "manifest_path": relative(V5_MANIFEST),
        "manifest_sha256": sha256_file(V5_MANIFEST),
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
        "candidate_prompt_template_path": relative(CANDIDATE_PROMPT_PATH),
        "candidate_prompt_template_sha256": sha256_file(CANDIDATE_PROMPT_PATH),
        "audit_prompt_template_path": relative(AUDIT_PROMPT_PATH),
        "audit_prompt_template_sha256": sha256_file(AUDIT_PROMPT_PATH),
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
        **backend.metadata,
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
    }
    write_json_atomic(paths["record"], record)
    print(f"ROW {row['document_id']} {row['condition']} completed")
    return True


def verify_development_prerequisite() -> Dict[str, Any]:
    base = ROOT / "results" / "v5" / "development"
    metrics_path = base / "metrics.json"
    seal_path = base / "output_seal.json"
    for path in (metrics_path, seal_path):
        if not path.is_file() or path.is_symlink():
            raise SystemExit(f"Formal inference requires committed {relative(path)}")
        try:
            committed = subprocess.check_output(
                ["git", "cat-file", "blob", f"HEAD:{relative(path)}"],
                cwd=ROOT,
            )
        except (OSError, subprocess.CalledProcessError) as exc:
            raise SystemExit(
                f"Formal inference requires {relative(path)} in execution HEAD"
            ) from exc
        if committed != path.read_bytes():
            raise SystemExit(f"Committed development artifact differs: {relative(path)}")
    status = subprocess.check_output(
        [
            "git",
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
            "--",
            relative(base),
        ],
        cwd=ROOT,
        text=True,
    ).strip()
    if status:
        raise SystemExit(
            "Formal inference requires the complete development namespace committed and clean"
        )
    metrics = read_json(metrics_path)
    seal = read_json(seal_path)
    expected_seal = {
        "protocol_version": PROTOCOL_VERSION,
        "split": "development",
        "output_count": 4,
        "protocol_lock_sha256": sha256_file(PROTOCOL_LOCK_PATH),
        "manifest_sha256": sha256_file(V5_MANIFEST),
        "ground_truth_read": False,
    }
    if not isinstance(seal, Mapping) or any(
        seal.get(key) != value for key, value in expected_seal.items()
    ):
        raise SystemExit("Development output seal is not bound to the current v5 protocol")
    expected_thresholds = {
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
    }
    expected_metrics = {
        "status": "DEVELOPMENT_PASS",
        "protocol_version": PROTOCOL_VERSION,
        "split": "development",
        "ground_truth_read": True,
        "intended_output_count": 4,
        "output_count": 4,
        "provenance_valid_count": 4,
        "gate_thresholds": expected_thresholds,
        "automated_gate_pass": True,
        "pass": True,
        "outreach_ready": False,
    }
    if not isinstance(metrics, Mapping) or any(
        metrics.get(key) != value for key, value in expected_metrics.items()
    ):
        raise SystemExit("Development qualification did not pass every frozen criterion")
    expected_criteria = {
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
    }
    criteria = metrics.get("gate_criteria")
    if (
        not isinstance(criteria, Mapping)
        or set(criteria) != expected_criteria
        or any(criteria[key] is not True for key in expected_criteria)
    ):
        raise SystemExit("Development qualification criteria are incomplete or altered")
    provenance = metrics.get("provenance")
    if (
        not isinstance(provenance, Mapping)
        or provenance.get("output_seal_path") != relative(seal_path)
        or provenance.get("output_seal_sha256") != sha256_file(seal_path)
    ):
        raise SystemExit("Development metrics do not hash-bind the output seal")
    rows = seal.get("rows")
    if not isinstance(rows, list) or len(rows) != 4:
        raise SystemExit("Development output seal does not contain all four assigned rows")
    final_overall = metrics.get("final_pipeline", {}).get("overall", {})
    if (
        final_overall.get("schema_valid_count") != 4
        or final_overall.get("field_exact_total") != 64
        or not isinstance(final_overall.get("field_exact_correct"), int)
        or final_overall["field_exact_correct"] < 58
        or final_overall.get("nonnull_total") != 52
        or not isinstance(final_overall.get("nonnull_recalled"), int)
        or final_overall["nonnull_recalled"] < 45
        or final_overall.get("unsupported_opportunities") != 12
        or final_overall.get("unsupported_count") != 0
    ):
        raise SystemExit("Development primary counts do not meet the frozen gate")
    by_condition = metrics.get("final_pipeline", {}).get("by_condition")
    if (
        not isinstance(by_condition, list)
        or {row.get("condition") for row in by_condition} != set(CONDITIONS)
        or any(
            row.get("field_exact_total") != 32
            or not isinstance(row.get("field_exact_correct"), int)
            or row["field_exact_correct"] < 28
            for row in by_condition
        )
    ):
        raise SystemExit("Development condition counts do not meet the frozen gate")
    telemetry = metrics.get("generation_telemetry", {})
    evidence = metrics.get("evidence", {})
    if (
        metrics.get("candidate", {}).get("strict_schema_valid_count") != 4
        or metrics.get("blind_audit", {}).get("strict_schema_valid_count") != 4
        or telemetry.get("candidate", {}).get("cap_hit_count") != 0
        or telemetry.get("audit", {}).get("cap_hit_count") != 0
        or evidence.get("accepted_with_valid_current_report_evidence_rate") != 1.0
        or evidence.get("history_carryover_count") != 0
    ):
        raise SystemExit("Development schema, cap, or evidence counts do not meet the gate")
    return {
        "metrics": artifact_metadata(metrics_path),
        "output_seal": artifact_metadata(seal_path),
    }


def run(args: argparse.Namespace) -> None:
    protocol_lock = verify_protocol_lock()
    rows = load_manifest(args.split, "all")
    expected_count = 4 if args.split == "development" else 20
    if len(rows) != expected_count:
        raise SystemExit(
            f"V5 {args.split} requires exactly {expected_count} assigned inputs; found {len(rows)}"
        )
    development_gate = verify_development_prerequisite() if args.split == "formal" else None
    repository_sha = repository_commit()
    launch_attempt_id = args.execution_attempt_id or execution_attempt_id(args.split)
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", launch_attempt_id) is None:
        raise SystemExit("--execution-attempt-id contains unsupported characters")

    candidate_schema = read_json(CANDIDATE_SCHEMA_PATH)
    audit_schema = read_json(AUDIT_SCHEMA_PATH)
    canonical_schema = read_json(CANONICAL_SCHEMA_PATH)
    write_execution_event(
        split=args.split,
        attempt_id=launch_attempt_id,
        event_name="started",
        repository_sha=repository_sha,
        protocol_lock=protocol_lock,
        payload={
            "manifest_path": relative(V5_MANIFEST),
            "manifest_sha256": sha256_file(V5_MANIFEST),
            "assigned_input_count": expected_count,
            "development_gate": development_gate,
        },
    )

    backend: Backend | None = None
    completed = 0
    verified = 0
    current_row: Mapping[str, str] | None = None
    try:
        backend = build_backend()
        verify_runtime_against_lock(protocol_lock, backend.metadata)
        for row in rows:
            current_row = row
            wrote = run_row(
                row=row,
                backend=backend,
                repository_sha=repository_sha,
                candidate_schema=candidate_schema,
                audit_schema=audit_schema,
                canonical_schema=canonical_schema,
                protocol_lock=protocol_lock,
                launch_attempt_id=launch_attempt_id,
            )
            completed += int(wrote)
            verified += int(not wrote)
    except BaseException as exc:
        try:
            write_execution_event(
                split=args.split,
                attempt_id=launch_attempt_id,
                event_name="failed",
                repository_sha=repository_sha,
                protocol_lock=protocol_lock,
                payload={
                    "completed_new_row_count": completed,
                    "verified_existing_row_count": verified,
                    "current_row": (
                        {
                            "document_id": current_row["document_id"],
                            "condition": current_row["condition"],
                        }
                        if current_row is not None
                        else None
                    ),
                    "error_type": type(exc).__name__,
                    "error_message": redact_error_message(exc),
                },
            )
        except FileExistsError:
            pass
        raise
    finally:
        if backend is not None:
            del backend

    write_execution_event(
        split=args.split,
        attempt_id=launch_attempt_id,
        event_name="completed",
        repository_sha=repository_sha,
        protocol_lock=protocol_lock,
        payload={
            "completed_new_row_count": completed,
            "verified_existing_row_count": verified,
            "assigned_input_count": expected_count,
        },
    )
    print(f"V5 {args.split} inference complete: {completed} new rows, {verified} verified rows.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=SPLITS, required=True)
    parser.add_argument(
        "--execution-attempt-id",
        help="Optional unique mechanical launch ID; generated automatically when omitted.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
