#!/usr/bin/env python3
"""Run the evidence-gated v3 MedGemma 1.5 protocol."""

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

PROTOCOL_VERSION = "evidence-gated-v3"
MODEL_ID = "google/medgemma-1.5-4b-it"
MODEL_REVISION = MODEL_REVISIONS[MODEL_ID]
MODEL_SLUG = slugify_model_id(MODEL_ID)
DTYPE_NAME = "bfloat16"
CANDIDATE_MAX_NEW_TOKENS = 512
AUDIT_MAX_NEW_TOKENS = 768
TESSERACT_LANGUAGE = "eng"
TESSERACT_CONFIG = "--oem 1 --psm 6"
OCR_TIMEOUT_SECONDS = 120
RANDOM_SEED = 0
CONDITIONS = ("clean", "ocr_degraded")
SPLITS = ("development", "formal")

V3_MANIFEST = ROOT / "data" / "v3" / "report_manifest.csv"
PROTOCOL_LOCK_PATH = ROOT / "data" / "v3" / "protocol_lock.json"
CANDIDATE_PROMPT_PATH = ROOT / "prompts" / "v3" / "candidate_prompt.txt"
AUDIT_PROMPT_PATH = ROOT / "prompts" / "v3" / "audit_prompt.txt"
CANDIDATE_SCHEMA_PATH = ROOT / "schema" / "v3" / "candidate.schema.json"
AUDIT_SCHEMA_PATH = ROOT / "schema" / "v3" / "audit.schema.json"
CANONICAL_SCHEMA_PATH = ROOT / "schema" / "extraction.schema.json"
COMPILER_PATH = ROOT / "scripts" / "v3_pipeline.py"
SCRIPT_PATH = Path(__file__).resolve()
OCR_MARKER = "{{OCR_LINES}}"
REQUIRED_LOCKED_SOURCE_PATHS = (
    "data/v3/report_manifest.csv",
    "docs/protocol_v3.md",
    "prompts/v3/audit_prompt.txt",
    "prompts/v3/candidate_prompt.txt",
    "pyproject.toml",
    "schema/extraction.schema.json",
    "schema/v3/audit.schema.json",
    "schema/v3/candidate.schema.json",
    "scripts/evaluate_v3.py",
    "scripts/evaluate_v3_development.py",
    "scripts/freeze_v3_protocol.py",
    "scripts/generate_v3_reports.py",
    "scripts/pilot_utils.py",
    "scripts/run_v3_inference.py",
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


def verify_protocol_lock() -> Dict[str, Any]:
    def is_evaluation_only_corpus_path(relative_path: str) -> bool:
        return relative_path == "data/v3/cases.csv" or relative_path.startswith(
            ("data/v3/ground_truth/", "data/v3/source_text/")
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
            "Formal v3 is locked out until a safe data/v3/protocol_lock.json "
            "is created and committed."
        )
    lock = read_json(PROTOCOL_LOCK_PATH)
    expected_scalars = {
        "status": "frozen",
        "protocol_version": PROTOCOL_VERSION,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "dtype": DTYPE_NAME,
        "random_seed": RANDOM_SEED,
        "candidate_max_new_tokens": CANDIDATE_MAX_NEW_TOKENS,
        "audit_max_new_tokens": AUDIT_MAX_NEW_TOKENS,
        "tesseract_language": TESSERACT_LANGUAGE,
        "tesseract_config": TESSERACT_CONFIG,
        "formal_output_count": 20,
        "candidate_calls_per_input": 1,
        "audit_calls_per_input": 1,
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
        if relative_path == "data/v3/report_manifest.csv":
            manifest_source_valid = True

    expected_operational_paths = {"data/v3/report_manifest.csv"}
    expected_evaluation_only_paths = {"data/v3/cases.csv"}
    if manifest_source_valid:
        try:
            with (ROOT / "data" / "v3" / "report_manifest.csv").open(
                newline="",
                encoding="utf-8",
            ) as handle:
                manifest_rows = list(csv.DictReader(handle))
        except OSError as exc:
            errors.append(f"could not read the locked v3 manifest: {exc}")
            manifest_rows = []
    else:
        manifest_rows = []
    manifest_path_fields = (
        ("pdf_path", "output/v3/pdf/", ".pdf", expected_operational_paths),
        ("image_path", "output/v3/rendered/", ".png", expected_operational_paths),
        ("source_text_path", "data/v3/source_text/", ".txt", expected_evaluation_only_paths),
        ("ground_truth_path", "data/v3/ground_truth/", ".json", expected_evaluation_only_paths),
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

    if lock.get("development_iterations_used") not in {1, 2}:
        errors.append("development_iterations_used must be 1 or 2")
    runtime = lock.get("runtime")
    required_runtime_keys = (
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
    if not isinstance(runtime, Mapping):
        errors.append("runtime is not a mapping")
    else:
        for key in required_runtime_keys:
            if not isinstance(runtime.get(key), str) or not runtime.get(key):
                errors.append(f"runtime.{key} is missing or empty")
    if lock.get("schema_enforcement") != (
        "lm-format-enforcer JsonSchemaParser via Transformers "
        "prefix_allowed_tokens_fn; force_json_field_order=true"
    ):
        errors.append("schema_enforcement does not match the implemented constrained decoder")
    if lock.get("formal_execution_order") != "manifest order; candidate then blind audit per input":
        errors.append("formal_execution_order does not match the implemented runner")
    if lock.get("infrastructure_retry_rule") != (
        "no automatic retry; resume only unchanged missing rows after preserving the failed attempt"
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
        errors.append("pass_criteria does not match the prespecified v3 gate")
    if errors:
        detail = "\n".join(f"- {error}" for error in errors)
        raise SystemExit(f"Formal v3 protocol lock validation failed:\n{detail}")
    return dict(lock)


def load_manifest(split: str, condition: str) -> List[Dict[str, str]]:
    with V3_MANIFEST.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    selected = [
        row
        for row in rows
        if row["split"] == split and (condition == "all" or row["condition"] == condition)
    ]
    if not selected:
        raise SystemExit(f"No v3 manifest rows found for split={split!r}, condition={condition!r}")
    if any(row["dataset_version"] != "pilot-v3" for row in selected):
        raise SystemExit("The selected manifest contains a non-v3 dataset row")
    return selected


def output_paths(
    row: Mapping[str, str],
    development_iteration: Optional[int] = None,
) -> Dict[str, Path]:
    if row["split"] == "development":
        if development_iteration not in {1, 2}:
            raise ValueError("Development output paths require iteration 1 or 2")
        base = ROOT / "results" / "v3" / "development" / f"iteration-{development_iteration}"
    else:
        if development_iteration is not None:
            raise ValueError("Formal output paths do not take a development iteration")
        base = ROOT / "results" / "v3" / "formal"
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
        raise SystemExit("V3 OCR requires `uv sync --extra inference`.") from exc

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


def build_backend() -> Backend:
    try:
        import torch
        from lmformatenforcer.integrations.transformers import (
            build_token_enforcer_tokenizer_data,
        )
        from transformers import AutoModelForImageTextToText, AutoProcessor
    except ImportError as exc:
        raise SystemExit("V3 inference requires `uv sync --extra inference`.") from exc

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
    if mismatches:
        detail = "\n".join(f"- {item}" for item in mismatches)
        raise RuntimeError(f"Frozen v3 runtime does not match the protocol lock:\n{detail}")


def constrained_generate(
    backend: Backend,
    image_path: Path,
    prompt: str,
    schema: Mapping[str, Any],
    max_new_tokens: int,
) -> Tuple[str, float]:
    from lmformatenforcer import JsonSchemaParser
    from lmformatenforcer.characterlevelparser import CharacterLevelParserConfig
    from lmformatenforcer.integrations.transformers import (
        build_transformers_prefix_allowed_tokens_fn,
    )

    parser = JsonSchemaParser(
        dict(schema),
        config=CharacterLevelParserConfig(force_json_field_order=True),
    )
    prefix_allowed_tokens_fn = build_transformers_prefix_allowed_tokens_fn(
        backend.tokenizer_data,
        parser,
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
    raw_text = backend.processor.decode(generated, skip_special_tokens=True)
    return raw_text, elapsed


def write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")


def formal_attempt_id() -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    return f"{timestamp}-pid{os.getpid()}"


def write_formal_attempt_event(
    attempt_id: str,
    event: str,
    payload: Mapping[str, Any],
) -> Path:
    if event not in {"started", "failed", "completed"}:
        raise ValueError(f"Unsupported formal attempt event: {event}")
    path = ROOT / "results" / "v3" / "formal" / "execution_attempts" / f"{attempt_id}.{event}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(dict(payload), indent=2, sort_keys=True, ensure_ascii=True) + "\n"
    with path.open("x", encoding="utf-8") as handle:
        handle.write(serialized)
    return path


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


def verify_formal_resume_safety(
    rows: Sequence[Mapping[str, str]],
    protocol_lock_sha256: str,
) -> None:
    """Refuse a semantic retry after a failed formal row obtained a valid response."""
    attempts_dir = ROOT / "results" / "v3" / "formal" / "execution_attempts"
    if not attempts_dir.is_dir():
        return
    rows_by_key = {(row["document_id"], row["condition"]): row for row in rows}
    events: Dict[str, Dict[str, Tuple[Path, Mapping[str, Any]]]] = {}
    errors: List[str] = []
    for event_name in ("started", "failed", "completed"):
        suffix = f".{event_name}.json"
        paths = sorted(attempts_dir.glob(f"*{suffix}"))
        for path in paths:
            try:
                event = read_json(path)
            except (OSError, json.JSONDecodeError) as exc:
                raise SystemExit(
                    f"Unreadable formal attempt record {relative(path)}: {exc}"
                ) from exc
            if not isinstance(event, Mapping):
                raise SystemExit(f"Formal attempt record is not an object: {relative(path)}")
            if event.get("protocol_lock_sha256") != protocol_lock_sha256:
                continue
            attempt_id = path.name.removesuffix(suffix)
            if event.get("execution_attempt_id") != attempt_id:
                errors.append(f"{relative(path)}: execution_attempt_id does not match filename")
                continue
            events.setdefault(attempt_id, {})[event_name] = (path, event)

    for attempt_id, attempt_events in events.items():
        started = attempt_events.get("started")
        failed = attempt_events.get("failed")
        completed = attempt_events.get("completed")
        if started is None:
            errors.append(f"{attempt_id}: terminal event has no matching started event")
        if failed is not None and completed is not None:
            errors.append(f"{attempt_id}: attempt has both failed and completed events")
        if failed is None and completed is None:
            errors.append(
                f"{attempt_id}: started attempt has no terminal event; retry eligibility "
                "cannot be established"
            )
        if failed is None:
            continue
        path, event = failed
        try:
            current_row = event["current_row"]
            resume_permitted = event["resume_permitted_for_current_row"]
        except KeyError as exc:
            errors.append(f"{relative(path)}: missing resume decision field {exc}")
            continue
        if not isinstance(resume_permitted, bool):
            errors.append(f"{relative(path)}: resume decision is not boolean")
            continue
        if current_row is None:
            continue
        if not isinstance(current_row, Mapping):
            errors.append(f"{relative(path)}: current_row is invalid")
            continue
        key = (current_row.get("document_id"), current_row.get("condition"))
        row = rows_by_key.get(key)
        if row is None:
            errors.append(f"{relative(path)}: attempt references an unknown row")
            continue
        if not resume_permitted:
            row_paths = output_paths(row)
            if not row_paths["record"].is_file():
                errors.append(
                    f"{relative(path)}: {key[0]} {key[1]} obtained a valid model "
                    "response or failed semantic validation before completion"
                )
    if errors:
        detail = "\n".join(f"- {item}" for item in errors)
        raise SystemExit(
            f"Formal v3 resume safety validation failed; no model calls were made:\n{detail}"
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
            f"Existing v3 artifacts lack a run record: {relative(paths['record'])}. "
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
            f"Existing v3 output provenance does not match this protocol:\n{detail}\n"
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

    progress["stage"] = "candidate_generation"
    progress["resume_permitted"] = True
    candidate_raw, candidate_elapsed = constrained_generate(
        backend,
        image_path,
        candidate_prompt,
        candidate_schema,
        CANDIDATE_MAX_NEW_TOKENS,
    )
    progress["stage"] = "candidate_validation"
    progress["resume_permitted"] = False
    candidate = parse_generated_json(candidate_raw, candidate_schema)
    progress["valid_model_response_count"] = 1

    # Persist the first valid response immediately. Any later failure makes this
    # an immutable partial row, which prevents a second candidate call.
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
    write_text(paths["candidate_raw"], candidate_raw)
    write_json(paths["candidate_parsed"], candidate)

    # The audit prompt was finalized before candidate generation and receives no candidate data.
    progress["stage"] = "audit_generation"
    audit_raw, audit_elapsed = constrained_generate(
        backend,
        image_path,
        audit_prompt,
        audit_schema,
        AUDIT_MAX_NEW_TOKENS,
    )
    progress["stage"] = "audit_validation"
    audit = parse_generated_json(audit_raw, audit_schema)
    progress["valid_model_response_count"] = 2
    write_text(paths["audit_raw"], audit_raw)
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
            "manifest_path": relative(V3_MANIFEST),
            "manifest_sha256": sha256_file(V3_MANIFEST),
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
            "candidate_elapsed_seconds": candidate_elapsed,
            "audit_elapsed_seconds": audit_elapsed,
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
        f"candidate {candidate_elapsed:.1f}s, audit {audit_elapsed:.1f}s)"
    )
    progress["stage"] = "completed"
    return True


def run(args: argparse.Namespace) -> None:
    if args.split == "development" and args.development_iteration not in {1, 2}:
        raise SystemExit(
            "Development runs require --development-iteration 1 or 2 so attempts are preserved."
        )
    if args.split == "formal":
        if args.development_iteration is not None:
            raise SystemExit("--development-iteration is not valid for the formal split.")
        if args.limit is not None:
            raise SystemExit("Formal v3 forbids --limit; all 20 frozen inputs must run.")
        if args.condition != "all":
            raise SystemExit("Formal v3 requires --condition all in manifest order.")
        if args.overwrite:
            raise SystemExit("Formal v3 forbids overwriting any prior artifact.")
    elif args.overwrite:
        raise SystemExit(
            "V3 outputs are immutable. Use the other development iteration after preserving "
            "the earlier attempt."
        )

    protocol_lock = verify_protocol_lock() if args.split == "formal" else None
    candidate_schema = read_json(CANDIDATE_SCHEMA_PATH)
    audit_schema = read_json(AUDIT_SCHEMA_PATH)
    canonical_schema = read_json(CANONICAL_SCHEMA_PATH)
    rows = load_manifest(args.split, args.condition)
    if args.limit is not None:
        if args.limit <= 0:
            raise SystemExit("--limit must be positive")
        rows = rows[: args.limit]

    repository_sha = repository_commit()
    protocol_lock_sha256 = sha256_file(PROTOCOL_LOCK_PATH) if protocol_lock is not None else None
    execution_attempt_id = formal_attempt_id() if protocol_lock is not None else None
    attempt_context: Dict[str, Any] = {
        "execution_attempt_id": execution_attempt_id,
        "protocol_version": PROTOCOL_VERSION,
        "protocol_lock_path": (relative(PROTOCOL_LOCK_PATH) if protocol_lock is not None else None),
        "protocol_lock_sha256": protocol_lock_sha256,
        "protocol_source_commit": (
            protocol_lock.get("protocol_source_commit") if protocol_lock is not None else None
        ),
        "repository_commit": repository_sha,
        "manifest_path": relative(V3_MANIFEST),
        "manifest_sha256": sha256_file(V3_MANIFEST),
        "selected_row_count": len(rows),
        "condition": args.condition,
    }
    if protocol_lock is not None:
        verify_formal_resume_safety(rows, protocol_lock_sha256)
        write_formal_attempt_event(
            execution_attempt_id,
            "started",
            {
                **attempt_context,
                "status": "started",
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            },
        )

    backend: Optional[Backend] = None
    completed = 0
    skipped = 0
    current_row: Optional[Mapping[str, str]] = None
    row_progress: Dict[str, Any] = {
        "stage": "backend_initialization",
        "valid_model_response_count": 0,
        "resume_permitted": True,
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
                    "resume_permitted_for_current_row": bool(
                        row_progress.get("resume_permitted", False)
                    ),
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
    print(f"Completed {completed} new v3 two-pass inference runs.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=SPLITS, required=True)
    parser.add_argument("--condition", choices=(*CONDITIONS, "all"), default="all")
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--development-iteration",
        type=int,
        choices=(1, 2),
        help="Required for development; routes artifacts to immutable iteration-1 or iteration-2.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
