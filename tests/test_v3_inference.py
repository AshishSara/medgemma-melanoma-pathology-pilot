from __future__ import annotations

import inspect
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from jsonschema import Draft202012Validator

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import run_v3_inference as v3_inference  # noqa: E402
from pilot_utils import MODEL_REVISIONS  # noqa: E402
from run_v3_inference import (  # noqa: E402
    AUDIT_PROMPT_PATH,
    AUDIT_SCHEMA_PATH,
    CANDIDATE_SCHEMA_PATH,
    MODEL_ID,
    MODEL_REVISION,
    StrictGeneratedJSONError,
    lines_from_tesseract_data,
    output_paths,
    parse_generated_json,
    redact_error_message,
    render_audit_prompt,
    verify_formal_resume_safety,
    write_formal_attempt_event,
)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def audit_entry(status: str = "absent", evidence: list[str] | None = None) -> dict[str, Any]:
    return {
        "status": status,
        "evidence_line_ids": [] if evidence is None else evidence,
    }


def complete_audit() -> dict[str, Any]:
    return {
        "document_id": audit_entry("present", ["L001"]),
        "specimen_site": audit_entry(),
        "laterality": audit_entry(),
        "diagnosis": audit_entry(),
        "breslow_thickness_mm": audit_entry(),
        "breslow_qualifier": audit_entry(),
        "ulceration": audit_entry(),
        "mitotic_rate_per_mm2": audit_entry(),
        "mitotic_qualifier": audit_entry(),
        "margins": {
            "invasive_peripheral": audit_entry(),
            "invasive_deep": audit_entry(),
            "in_situ_peripheral": audit_entry(),
            "in_situ_deep": audit_entry(),
        },
        "staging": {
            "pT": audit_entry(),
            "pN": audit_entry(),
            "pM": audit_entry(),
            "stage_group": audit_entry(),
        },
    }


def walk(value: Any):
    yield value
    if isinstance(value, dict):
        for child in value.values():
            yield from walk(child)
    elif isinstance(value, list):
        for child in value:
            yield from walk(child)


def test_v3_schemas_avoid_lmfe_incompatible_conditionals_and_mixed_enums() -> None:
    for path in (CANDIDATE_SCHEMA_PATH, AUDIT_SCHEMA_PATH):
        schema = load_json(path)
        assert all("allOf" not in node for node in walk(schema) if isinstance(node, dict))
        for node in walk(schema):
            if isinstance(node, dict) and "enum" in node:
                assert None not in node["enum"]


def test_audit_schema_accepts_complete_nested_presence_ledger() -> None:
    schema = load_json(AUDIT_SCHEMA_PATH)
    assert list(Draft202012Validator(schema).iter_errors(complete_audit())) == []


def test_audit_rendering_has_no_candidate_input_or_canonical_value_list() -> None:
    assert tuple(inspect.signature(render_audit_prompt).parameters) == ("lines",)
    rendered = render_audit_prompt(
        [{"line_id": "L001", "text": "Document ID: MEL-104-A", "section": "header"}]
    )
    template = AUDIT_PROMPT_PATH.read_text(encoding="utf-8")
    assert "{{CANDIDATE" not in template.upper()
    assert "invasive_melanoma" not in rendered
    assert "not_involved" not in rendered
    assert "[L001] [header] Document ID: MEL-104-A" in rendered


def test_tesseract_words_are_grouped_into_stable_line_ids() -> None:
    data = {
        "text": ["Document", "ID:", "", "MEL-104-A", "Diagnosis:", "Melanoma"],
        "conf": ["98", "96", "-1", "94", "93", "91"],
        "page_num": [1, 1, 1, 1, 1, 1],
        "block_num": [1, 1, 1, 1, 1, 1],
        "par_num": [1, 1, 1, 1, 1, 1],
        "line_num": [1, 1, 1, 1, 2, 2],
    }
    lines = lines_from_tesseract_data(data)
    assert lines == [
        {
            "line_id": "L001",
            "text": "Document ID: MEL-104-A",
            "mean_confidence": 96.0,
            "page_num": 1,
            "block_num": 1,
            "paragraph_num": 1,
            "source_line_num": 1,
        },
        {
            "line_id": "L002",
            "text": "Diagnosis: Melanoma",
            "mean_confidence": 92.0,
            "page_num": 1,
            "block_num": 1,
            "paragraph_num": 1,
            "source_line_num": 2,
        },
    ]


def test_strict_generation_parser_rejects_duplicate_keys() -> None:
    schema = {
        "type": "object",
        "additionalProperties": False,
        "required": ["value"],
        "properties": {"value": {"type": "string"}},
    }
    with pytest.raises(StrictGeneratedJSONError, match="Duplicate JSON key"):
        parse_generated_json('{"value":"a","value":"b"}', schema)


def test_v3_paths_are_split_and_model_versioned() -> None:
    paths = output_paths(
        {
            "split": "formal",
            "condition": "clean",
            "document_id": "MEL-104-A",
        }
    )
    assert "results/v3/formal/" in paths["record"].as_posix()
    assert "google__medgemma-1.5-4b-it" in paths["record"].as_posix()
    assert paths["compiler_audit"].name == "MEL-104-A.json"


def test_v3_development_paths_are_iteration_scoped_and_immutable() -> None:
    row = {
        "split": "development",
        "condition": "clean",
        "document_id": "MEL-101-A",
    }
    with pytest.raises(ValueError, match="iteration 1 or 2"):
        output_paths(row)
    paths = output_paths(row, development_iteration=1)
    assert "results/v3/development/iteration-1/" in paths["record"].as_posix()
    assert paths["candidate_prompt"].name == "MEL-101-A.txt"
    assert paths["audit_prompt"].name == "MEL-101-A.txt"


def test_v3_model_is_the_pinned_medgemma_15_revision() -> None:
    assert MODEL_ID == "google/medgemma-1.5-4b-it"
    assert MODEL_REVISION == "91850547d9f0b2fdd21aa7c5f4f3d1a8a52c243b"
    assert MODEL_REVISIONS[MODEL_ID] == MODEL_REVISION


def test_formal_attempt_events_are_append_only_and_redact_tokens(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(v3_inference, "ROOT", tmp_path)
    monkeypatch.setenv("HF_TOKEN", "hf_secret_value")
    payload = {"status": "failed"}
    path = write_formal_attempt_event("attempt-1", "failed", payload)
    assert load_json(path) == payload
    with pytest.raises(FileExistsError):
        write_formal_attempt_event("attempt-1", "failed", payload)
    message = redact_error_message(
        RuntimeError(
            "hf_secret_value Authorization: Bearer other-secret "
            "https://example.test/?access_token=query-secret"
        )
    )
    assert "secret" not in message
    assert message.count("[REDACTED]") == 3


def test_formal_resume_rejects_failed_row_after_valid_response(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(v3_inference, "ROOT", tmp_path)
    row = {
        "split": "formal",
        "condition": "clean",
        "document_id": "MEL-104-A",
    }
    write_formal_attempt_event(
        "attempt-1",
        "started",
        {
            "execution_attempt_id": "attempt-1",
            "protocol_lock_sha256": "lock-sha",
        },
    )
    write_formal_attempt_event(
        "attempt-1",
        "failed",
        {
            "execution_attempt_id": "attempt-1",
            "protocol_lock_sha256": "lock-sha",
            "current_row": {
                "document_id": "MEL-104-A",
                "condition": "clean",
            },
            "resume_permitted_for_current_row": False,
        },
    )
    with pytest.raises(SystemExit, match="resume safety validation failed"):
        verify_formal_resume_safety([row], "lock-sha")


def test_formal_resume_allows_missing_row_before_valid_response(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(v3_inference, "ROOT", tmp_path)
    row = {
        "split": "formal",
        "condition": "clean",
        "document_id": "MEL-104-A",
    }
    write_formal_attempt_event(
        "attempt-1",
        "started",
        {
            "execution_attempt_id": "attempt-1",
            "protocol_lock_sha256": "lock-sha",
        },
    )
    write_formal_attempt_event(
        "attempt-1",
        "failed",
        {
            "execution_attempt_id": "attempt-1",
            "protocol_lock_sha256": "lock-sha",
            "current_row": {
                "document_id": "MEL-104-A",
                "condition": "clean",
            },
            "resume_permitted_for_current_row": True,
        },
    )
    verify_formal_resume_safety([row], "lock-sha")


def test_formal_resume_rejects_started_attempt_without_terminal_event(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(v3_inference, "ROOT", tmp_path)
    write_formal_attempt_event(
        "attempt-1",
        "started",
        {
            "execution_attempt_id": "attempt-1",
            "protocol_lock_sha256": "lock-sha",
        },
    )
    with pytest.raises(SystemExit, match="retry eligibility cannot be established"):
        verify_formal_resume_safety([], "lock-sha")


def test_formal_run_requires_full_condition_matrix() -> None:
    args = SimpleNamespace(
        split="formal",
        development_iteration=None,
        limit=None,
        condition="clean",
        overwrite=False,
    )
    with pytest.raises(SystemExit, match="condition all"):
        v3_inference.run(args)


def test_formal_runner_records_failure_and_does_not_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_path = tmp_path / "data" / "v3" / "report_manifest.csv"
    lock_path = tmp_path / "data" / "v3" / "protocol_lock.json"
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text("manifest\n", encoding="utf-8")
    lock_path.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(v3_inference, "ROOT", tmp_path)
    monkeypatch.setattr(v3_inference, "V3_MANIFEST", manifest_path)
    monkeypatch.setattr(v3_inference, "PROTOCOL_LOCK_PATH", lock_path)
    monkeypatch.setattr(
        v3_inference,
        "verify_protocol_lock",
        lambda: {"protocol_source_commit": "a" * 40},
    )
    monkeypatch.setattr(v3_inference, "read_json", lambda _path: {})
    row = {
        "dataset_version": "pilot-v3",
        "split": "formal",
        "document_id": "MEL-104-A",
        "condition": "clean",
    }
    monkeypatch.setattr(v3_inference, "load_manifest", lambda *_args: [row])
    monkeypatch.setattr(
        v3_inference,
        "build_backend",
        lambda: SimpleNamespace(metadata={}),
    )
    monkeypatch.setattr(v3_inference, "verify_runtime_against_lock", lambda *_args: None)
    monkeypatch.setattr(v3_inference, "repository_commit", lambda: "b" * 40)
    monkeypatch.setattr(v3_inference, "formal_attempt_id", lambda: "attempt-1")
    calls = 0

    def fail_once(*call_args: Any) -> bool:
        nonlocal calls
        calls += 1
        progress = call_args[-1]
        progress.update(
            {
                "stage": "candidate_generation",
                "valid_model_response_count": 0,
                "resume_permitted": True,
            }
        )
        raise RuntimeError("transient GPU failure")

    monkeypatch.setattr(v3_inference, "run_row", fail_once)
    args = SimpleNamespace(
        split="formal",
        development_iteration=None,
        limit=None,
        condition="all",
        overwrite=False,
    )
    with pytest.raises(RuntimeError, match="transient GPU failure"):
        v3_inference.run(args)
    assert calls == 1
    attempts = tmp_path / "results" / "v3" / "formal" / "execution_attempts"
    assert (attempts / "attempt-1.started.json").is_file()
    failed = load_json(attempts / "attempt-1.failed.json")
    assert failed["status"] == "failed"
    assert failed["resume_permitted_for_current_row"] is True
    assert failed["valid_model_response_count_for_current_row"] == 0
    assert not (attempts / "attempt-1.completed.json").exists()


def test_completed_row_is_hash_verified_and_never_overwritten(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(v3_inference, "ROOT", tmp_path)
    row = {
        "dataset_version": "pilot-v3",
        "split": "development",
        "document_id": "MEL-101-A",
        "condition": "clean",
        "image_sha256": "image-sha",
    }
    paths = output_paths(row, development_iteration=1)
    for name, path in paths.items():
        if name == "record":
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"{name}\n", encoding="utf-8")
    artifacts = {
        name: {"sha256": v3_inference.sha256_file(path)}
        for name, path in paths.items()
        if name != "record"
    }
    record = {
        "attempt_status": "completed",
        "protocol_version": v3_inference.PROTOCOL_VERSION,
        "protocol_lock_path": None,
        "dataset_version": "pilot-v3",
        "split": "development",
        "development_iteration": 1,
        "document_id": "MEL-101-A",
        "condition": "clean",
        "model_id": MODEL_ID,
        "requested_revision": MODEL_REVISION,
        "image_sha256": "image-sha",
        "candidate_prompt_template_sha256": v3_inference.sha256_file(
            v3_inference.CANDIDATE_PROMPT_PATH
        ),
        "audit_prompt_template_sha256": v3_inference.sha256_file(v3_inference.AUDIT_PROMPT_PATH),
        "candidate_schema_sha256": v3_inference.sha256_file(v3_inference.CANDIDATE_SCHEMA_PATH),
        "audit_schema_sha256": v3_inference.sha256_file(v3_inference.AUDIT_SCHEMA_PATH),
        "canonical_schema_sha256": v3_inference.sha256_file(v3_inference.CANONICAL_SCHEMA_PATH),
        "compiler_sha256": v3_inference.sha256_file(v3_inference.COMPILER_PATH),
        "protocol_lock_sha256": None,
        "protocol_source_commit": None,
        "artifacts": artifacts,
    }
    paths["record"].parent.mkdir(parents=True, exist_ok=True)
    paths["record"].write_text(json.dumps(record), encoding="utf-8")
    before = {name: v3_inference.sha256_file(path) for name, path in paths.items()}
    wrote = v3_inference.run_row(
        row,
        SimpleNamespace(),
        "repository-sha",
        {},
        {},
        {},
        1,
        None,
    )
    after = {name: v3_inference.sha256_file(path) for name, path in paths.items()}
    assert wrote is False
    assert after == before


def test_valid_candidate_is_preserved_and_blocks_audit_failure_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(v3_inference, "ROOT", tmp_path)
    image_path = tmp_path / "input.png"
    monkeypatch.setattr(v3_inference, "validate_manifest_image", lambda _row: image_path)
    monkeypatch.setattr(
        v3_inference,
        "run_ocr",
        lambda _path: (
            [{"line_id": "L001", "text": "Document ID: MEL-101-A"}],
            {"text": ["Document ID: MEL-101-A"]},
            {"elapsed_seconds": 0.1},
        ),
    )
    generation_calls = 0

    def generate(*_args: Any) -> tuple[str, float]:
        nonlocal generation_calls
        generation_calls += 1
        if generation_calls == 1:
            return '{"value":1}', 0.2
        raise RuntimeError("audit GPU failure")

    monkeypatch.setattr(v3_inference, "constrained_generate", generate)
    row = {
        "split": "development",
        "condition": "clean",
        "document_id": "MEL-101-A",
    }
    candidate_schema = {
        "type": "object",
        "additionalProperties": False,
        "required": ["value"],
        "properties": {"value": {"type": "integer"}},
    }
    progress: dict[str, Any] = {}
    with pytest.raises(RuntimeError, match="audit GPU failure"):
        v3_inference.run_row(
            row,
            SimpleNamespace(),
            "repository-sha",
            candidate_schema,
            {},
            {},
            1,
            None,
            progress=progress,
        )
    paths = output_paths(row, development_iteration=1)
    assert paths["candidate_raw"].read_text(encoding="utf-8") == '{"value":1}'
    assert load_json(paths["candidate_parsed"]) == {"value": 1}
    assert paths["ocr"].is_file()
    assert paths["candidate_prompt"].is_file()
    assert paths["audit_prompt"].is_file()
    assert not paths["audit_raw"].exists()
    assert not paths["record"].exists()
    assert progress["valid_model_response_count"] == 1
    assert progress["resume_permitted"] is False
