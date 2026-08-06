from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import run_v5_inference as inference  # noqa: E402

REPOSITORY_SHA = "b" * 40
SOURCE_SHA = "a" * 40
PASS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["value"],
    "properties": {"value": {"type": "integer"}},
}
CANONICAL_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["ok"],
    "properties": {"ok": {"type": "boolean"}},
}


def _configure_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    paths = {
        "PROTOCOL_LOCK_PATH": "data/v5/protocol_lock.json",
        "V5_MANIFEST": "data/v5/report_manifest.csv",
        "CANDIDATE_PROMPT_PATH": "prompts/v3/candidate_prompt.txt",
        "AUDIT_PROMPT_PATH": "prompts/v3/audit_prompt.txt",
        "CANDIDATE_SCHEMA_PATH": "schema/v3/candidate.schema.json",
        "AUDIT_SCHEMA_PATH": "schema/v3/audit.schema.json",
        "CANONICAL_SCHEMA_PATH": "schema/extraction.schema.json",
        "COMPILER_PATH": "scripts/v3_pipeline.py",
        "SCRIPT_PATH": "scripts/run_v5_inference.py",
    }
    monkeypatch.setattr(inference, "ROOT", tmp_path)
    for name, relative_path in paths.items():
        path = tmp_path / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        if name == "PROTOCOL_LOCK_PATH":
            path.write_text("{}\n", encoding="utf-8")
        elif name == "V5_MANIFEST":
            path.write_text("dataset_version\n", encoding="utf-8")
        elif name in {"CANDIDATE_SCHEMA_PATH", "AUDIT_SCHEMA_PATH"}:
            path.write_text(json.dumps(PASS_SCHEMA) + "\n", encoding="utf-8")
        elif name == "CANONICAL_SCHEMA_PATH":
            path.write_text(json.dumps(CANONICAL_SCHEMA) + "\n", encoding="utf-8")
        else:
            path.write_text(f"{name}\n", encoding="utf-8")
        monkeypatch.setattr(inference, name, path)
    image_path = tmp_path / "output/v5/rendered/clean/input.png"
    image_path.parent.mkdir(parents=True, exist_ok=True)
    image_path.write_bytes(b"fake-image")


def _row(split: str = "development") -> dict[str, str]:
    return {
        "dataset_version": "pilot-v5",
        "generator_version": "5.0.0",
        "split": split,
        "semantic_case_id": "MEL-190" if split == "development" else "MEL-201",
        "document_id": "MEL-190-A" if split == "development" else "MEL-201-A",
        "template_id": "A",
        "condition": "clean",
        "image_path": "output/v5/rendered/clean/input.png",
        "image_sha256": inference.sha256_bytes(b"fake-image"),
        "ground_truth_path": "data/v5/ground_truth/unused.json",
    }


def _protocol_lock() -> dict[str, Any]:
    return {"protocol_source_commit": SOURCE_SHA}


def test_locked_source_hashes_must_match_protocol_source_commit(tmp_path: Path) -> None:
    locked = tmp_path / "locked.txt"
    locked.write_text("prospectively frozen\n", encoding="utf-8")
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(
        ["git", "config", "user.name", "V5 Source Binding Test"],
        cwd=tmp_path,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.email", "v5-source-binding@example.test"],
        cwd=tmp_path,
        check=True,
    )
    subprocess.run(["git", "add", "locked.txt"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-qm", "prospective source"], cwd=tmp_path, check=True)
    source_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=tmp_path,
        text=True,
    ).strip()

    locked.write_text("retrospectively changed\n", encoding="utf-8")
    subprocess.run(["git", "add", "locked.txt"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-qm", "descendant mutation"], cwd=tmp_path, check=True)
    forged_lock = {"locked.txt": inference.sha256_file(locked)}

    errors = inference.committed_artifact_binding_errors(
        tmp_path,
        source_commit,
        forged_lock,
    )
    assert errors == ["locked artifact differs from protocol source commit: locked.txt"]


def test_locked_source_hashes_must_match_head_even_if_worktree_is_restored(
    tmp_path: Path,
) -> None:
    locked = tmp_path / "locked.txt"
    locked.write_text("prospectively frozen\n", encoding="utf-8")
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(
        ["git", "config", "user.name", "V5 HEAD Binding Test"],
        cwd=tmp_path,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.email", "v5-head-binding@example.test"],
        cwd=tmp_path,
        check=True,
    )
    subprocess.run(["git", "add", "locked.txt"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-qm", "prospective source"], cwd=tmp_path, check=True)
    source_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=tmp_path,
        text=True,
    ).strip()
    frozen_sha = inference.sha256_file(locked)

    locked.write_text("committed descendant mutation\n", encoding="utf-8")
    subprocess.run(["git", "add", "locked.txt"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-qm", "descendant mutation"], cwd=tmp_path, check=True)
    locked.write_text("prospectively frozen\n", encoding="utf-8")

    errors = inference.committed_artifact_binding_errors(
        tmp_path,
        source_commit,
        {"locked.txt": frozen_sha},
    )
    assert errors == ["locked artifact differs from execution HEAD: locked.txt"]


def test_protocol_lock_must_be_absent_from_source_commit(tmp_path: Path) -> None:
    lock = tmp_path / "data" / "v5" / "protocol_lock.json"
    lock.parent.mkdir(parents=True)
    lock.write_text("{}\n", encoding="utf-8")
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(
        ["git", "config", "user.name", "V5 Prospective Lock Test"],
        cwd=tmp_path,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.email", "v5-lock-origin@example.test"],
        cwd=tmp_path,
        check=True,
    )
    subprocess.run(["git", "add", "data/v5/protocol_lock.json"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-qm", "improper source lock"], cwd=tmp_path, check=True)
    source_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=tmp_path,
        text=True,
    ).strip()

    assert inference.prospective_lock_origin_errors(
        tmp_path,
        source_commit,
        "data/v5/protocol_lock.json",
    ) == ["protocol lock already existed in the protocol source commit"]


def _result(
    raw_text: str = '{"value":1}',
    *,
    max_new_tokens: int = inference.CANDIDATE_MAX_NEW_TOKENS,
    cap_hit: bool = False,
    force_json_field_order: bool = False,
    max_consecutive_whitespaces: int = 12,
) -> inference.GenerationResult:
    token_ids = tuple(range(max_new_tokens)) if cap_hit else (20, 21)
    return inference.GenerationResult(
        raw_text=raw_text,
        elapsed_seconds=0.01,
        prompt_token_count=12,
        generated_token_ids=token_ids,
        generated_token_ids_sha256=inference.sha256_bytes(
            json.dumps(list(token_ids), separators=(",", ":")).encode("ascii")
        ),
        eos_token_ids=(1,),
        token_count=len(token_ids),
        eos_observed=False,
        cap_hit=cap_hit,
        max_new_tokens=max_new_tokens,
        serializer={
            "configuration_stage": "after_transformers_prefix_construction",
            "tokenizer_alphabet_preserved": True,
            "force_json_field_order": force_json_field_order,
            "max_consecutive_whitespaces": max_consecutive_whitespaces,
            "max_json_array_length": 20,
        },
    )


def _backend() -> SimpleNamespace:
    return SimpleNamespace(
        metadata={
            "resolved_revision": inference.MODEL_REVISION,
            "device": "cuda:0",
            "dtype": "torch.bfloat16",
        }
    )


def _generation_settings(
    args: tuple[Any, ...],
    kwargs: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "max_new_tokens": (kwargs["max_new_tokens"] if "max_new_tokens" in kwargs else args[4]),
        "force_json_field_order": kwargs["force_json_field_order"],
        "max_consecutive_whitespaces": kwargs["max_consecutive_whitespaces"],
    }


def _obtain(
    *,
    row: Mapping[str, str],
    bundle_path: Path,
    backend: Any | None = None,
    pass_name: str = "candidate",
    launch_attempt_id: str = "launch-1",
    rendered_prompt: str | None = None,
    repository_sha: str = REPOSITORY_SHA,
) -> dict[str, Any]:
    candidate = pass_name == "candidate"
    return inference.obtain_response_bundle(
        row=row,
        pass_name=pass_name,
        bundle_path=bundle_path,
        image_path=inference.ROOT / row["image_path"],
        rendered_prompt=rendered_prompt or f"{pass_name} prompt",
        schema=PASS_SCHEMA,
        schema_path=(inference.CANDIDATE_SCHEMA_PATH if candidate else inference.AUDIT_SCHEMA_PATH),
        max_new_tokens=(
            inference.CANDIDATE_MAX_NEW_TOKENS if candidate else inference.AUDIT_MAX_NEW_TOKENS
        ),
        force_json_field_order=(
            inference.CANDIDATE_SERIALIZER_FORCE_JSON_FIELD_ORDER
            if candidate
            else inference.AUDIT_SERIALIZER_FORCE_JSON_FIELD_ORDER
        ),
        max_consecutive_whitespaces=(
            inference.CANDIDATE_SERIALIZER_MAX_CONSECUTIVE_WHITESPACES
            if candidate
            else inference.AUDIT_SERIALIZER_MAX_CONSECUTIVE_WHITESPACES
        ),
        backend=backend or _backend(),
        repository_sha=repository_sha,
        protocol_lock=_protocol_lock(),
        row_attempt_id="row-attempt-1",
        launch_attempt_id=launch_attempt_id,
    )


def _write_retryable_first_call(
    row: Mapping[str, str],
    *,
    rendered_prompt: str = "candidate prompt",
    repository_sha: str = REPOSITORY_SHA,
) -> None:
    request_identity = inference.call_request_identity(
        row=row,
        pass_name="candidate",
        image_path=inference.ROOT / row["image_path"],
        rendered_prompt=rendered_prompt,
        schema=PASS_SCHEMA,
        schema_path=inference.CANDIDATE_SCHEMA_PATH,
        max_new_tokens=inference.CANDIDATE_MAX_NEW_TOKENS,
        force_json_field_order=inference.CANDIDATE_SERIALIZER_FORCE_JSON_FIELD_ORDER,
        max_consecutive_whitespaces=(inference.CANDIDATE_SERIALIZER_MAX_CONSECUTIVE_WHITESPACES),
    )
    common = {
        "row": row,
        "pass_name": "candidate",
        "call_ordinal": 1,
        "row_attempt_id": "row-attempt-1",
        "launch_attempt_id": "launch-1",
        "repository_sha": repository_sha,
        "protocol_lock": _protocol_lock(),
        "request_identity": request_identity,
    }
    inference.write_call_event(event_name="started", **common)
    inference.write_call_event(
        event_name="no_response",
        payload={
            "error_type": "TimeoutError",
            "error_message": "runtime timed out",
            "retry_permitted": True,
        },
        **common,
    )


def _patch_row_prerequisites(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        inference,
        "validate_manifest_image",
        lambda row: tmp_path / row["image_path"],
    )

    def fake_ocr(
        **kwargs: Any,
    ) -> tuple[
        list[dict[str, str]],
        str,
        str,
        dict[str, Any],
    ]:
        paths = kwargs["paths"]
        lines = [{"line_id": "L001", "text": "Document ID: MEL-190-A"}]
        metadata = {"elapsed_seconds": 0.01}
        inference._write_or_verify_json(  # noqa: SLF001
            paths["ocr"],
            {
                "document_id": kwargs["row"]["document_id"],
                "condition": kwargs["row"]["condition"],
                **metadata,
                "raw_word_data": {"text": ["Document ID: MEL-190-A"]},
                "lines": lines,
            },
        )
        inference._write_or_verify_text(  # noqa: SLF001
            paths["candidate_prompt"],
            "candidate prompt",
        )
        inference._write_or_verify_text(  # noqa: SLF001
            paths["audit_prompt"],
            "audit prompt",
        )
        return lines, "candidate prompt", "audit prompt", metadata

    monkeypatch.setattr(inference, "load_or_create_ocr", fake_ocr)
    monkeypatch.setattr(
        inference,
        "compile_prediction",
        lambda *_args: (
            {"ok": True},
            {
                "compiler_version": "fake",
                "accepted_non_null_count": 0,
                "rejected_non_null_count": 0,
                "history_evidence_rejection_count": 0,
                "schema_valid": True,
            },
        ),
    )


def _run_one_row(
    row: Mapping[str, str],
    *,
    launch_attempt_id: str = "launch-1",
) -> bool:
    return inference.run_row(
        row=row,
        backend=_backend(),
        repository_sha=REPOSITORY_SHA,
        candidate_schema=PASS_SCHEMA,
        audit_schema=PASS_SCHEMA,
        canonical_schema=CANONICAL_SCHEMA,
        protocol_lock=_protocol_lock(),
        launch_attempt_id=launch_attempt_id,
    )


@pytest.mark.parametrize("split", ["development", "formal"])
def test_run_enforces_protocol_lock_before_any_split_work(
    split: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    touched: list[str] = []

    def reject_lock() -> dict[str, Any]:
        touched.append("lock")
        raise SystemExit("locked out")

    monkeypatch.setattr(inference, "verify_protocol_lock", reject_lock)
    monkeypatch.setattr(
        inference,
        "load_manifest",
        lambda *_args: pytest.fail("manifest read preceded lock verification"),
    )

    with pytest.raises(SystemExit, match="locked out"):
        inference.run(argparse.Namespace(split=split, execution_attempt_id=f"{split}-attempt"))

    assert touched == ["lock"]


def test_formal_run_checks_qualification_before_backend_or_attempt_claim(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    order: list[str] = []
    monkeypatch.setattr(
        inference,
        "verify_protocol_lock",
        lambda: order.append("lock") or _protocol_lock(),
    )
    monkeypatch.setattr(
        inference,
        "load_manifest",
        lambda *_args: [_row("formal") for _ in range(20)],
    )

    def reject_qualification() -> dict[str, Any]:
        order.append("qualification")
        raise SystemExit("qualification absent")

    monkeypatch.setattr(inference, "verify_development_prerequisite", reject_qualification)
    monkeypatch.setattr(
        inference,
        "build_backend",
        lambda: pytest.fail("backend loaded before qualification passed"),
    )
    monkeypatch.setattr(
        inference,
        "write_execution_event",
        lambda **_kwargs: pytest.fail("attempt claimed before qualification passed"),
    )

    with pytest.raises(SystemExit, match="qualification absent"):
        inference.run(argparse.Namespace(split="formal", execution_attempt_id="formal-attempt"))

    assert order == ["lock", "qualification"]


def _write_qualification_files(
    tmp_path: Path,
    *,
    status: str = "DEVELOPMENT_PASS",
    passed: bool = True,
) -> tuple[Path, Path]:
    base = tmp_path / "results/v5/development"
    base.mkdir(parents=True, exist_ok=True)
    seal_path = base / "output_seal.json"
    seal = {
        "protocol_version": inference.PROTOCOL_VERSION,
        "split": "development",
        "output_count": 4,
        "protocol_lock_sha256": inference.sha256_file(inference.PROTOCOL_LOCK_PATH),
        "manifest_sha256": inference.sha256_file(inference.V5_MANIFEST),
        "ground_truth_read": False,
    }
    seal_path.write_text(json.dumps(seal, sort_keys=True) + "\n", encoding="utf-8")
    metrics_path = base / "metrics.json"
    metrics = {
        "status": status,
        "protocol_version": inference.PROTOCOL_VERSION,
        "split": "development",
        "automated_gate_pass": passed,
        "pass": passed,
        "gate_criteria": {"all_outputs_complete": passed},
        "provenance": {"output_seal_sha256": inference.sha256_file(seal_path)},
    }
    metrics_path.write_text(json.dumps(metrics, sort_keys=True) + "\n", encoding="utf-8")
    return metrics_path, seal_path


def test_formal_prerequisite_rejects_absent_qualification(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _configure_paths(tmp_path, monkeypatch)

    with pytest.raises(SystemExit, match="requires committed"):
        inference.verify_development_prerequisite()


def test_formal_prerequisite_rejects_uncommitted_qualification(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _configure_paths(tmp_path, monkeypatch)
    _write_qualification_files(tmp_path)
    monkeypatch.setattr(
        inference.subprocess,
        "check_output",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(subprocess.CalledProcessError(128, "git")),
    )

    with pytest.raises(SystemExit, match="in execution HEAD"):
        inference.verify_development_prerequisite()


def test_formal_prerequisite_rejects_nonpassing_qualification(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _configure_paths(tmp_path, monkeypatch)
    _write_qualification_files(
        tmp_path,
        status="DEVELOPMENT_FAIL",
        passed=False,
    )

    def git_output(command: list[str], **_kwargs: Any) -> bytes | str:
        if command[1] == "cat-file":
            relative_path = command[-1].removeprefix("HEAD:")
            return (tmp_path / relative_path).read_bytes()
        if command[1] == "status":
            return ""
        raise AssertionError(command)

    monkeypatch.setattr(inference.subprocess, "check_output", git_output)

    with pytest.raises(SystemExit, match="did not pass every frozen criterion"):
        inference.verify_development_prerequisite()


def test_output_paths_are_split_scoped_and_only_use_v5_namespace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _configure_paths(tmp_path, monkeypatch)

    for split in ("development", "formal"):
        paths = inference.output_paths(_row(split))
        assert set(paths) == {
            "ocr",
            "candidate_prompt",
            "audit_prompt",
            "candidate_bundle",
            "audit_bundle",
            "candidate_parsed",
            "audit_parsed",
            "normalized",
            "compiler_audit",
            "record",
        }
        assert len(set(paths.values())) == len(paths)
        for path in paths.values():
            relative = path.relative_to(tmp_path).as_posix()
            assert relative.startswith(f"results/v5/{split}/")
            assert "/v2/" not in relative
            assert "/v3/" not in relative
            assert "/v4/" not in relative


def test_response_bundle_is_atomic_and_durable_before_parse(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _configure_paths(tmp_path, monkeypatch)
    _patch_row_prerequisites(tmp_path, monkeypatch)
    row = _row()
    paths = inference.output_paths(row)
    monkeypatch.setattr(
        inference,
        "constrained_generate",
        lambda *_args, **_kwargs: _result(raw_text='{"value":'),
    )

    def reject_after_bundle(raw_text: str, _schema: Mapping[str, Any]) -> dict[str, Any]:
        assert raw_text == '{"value":'
        assert paths["candidate_bundle"].is_file()
        bundle = json.loads(paths["candidate_bundle"].read_text(encoding="utf-8"))
        assert bundle["raw_text"] == raw_text
        assert bundle["raw_text_descriptor"] == inference.text_metadata(raw_text)
        raise inference.StrictGeneratedJSONError("malformed")

    monkeypatch.setattr(inference, "parse_generated_json", reject_after_bundle)

    with pytest.raises(inference.StrictGeneratedJSONError, match="malformed"):
        _run_one_row(row)

    assert not list(paths["candidate_bundle"].parent.glob("*.tmp"))
    assert not paths["candidate_parsed"].exists()
    assert not paths["audit_bundle"].exists()


def test_call_start_no_response_gets_exactly_one_unchanged_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _configure_paths(tmp_path, monkeypatch)
    row = _row()
    bundle_path = inference.output_paths(row)["candidate_bundle"]
    calls: list[dict[str, Any]] = []

    def generate(*args: Any, **kwargs: Any) -> inference.GenerationResult:
        settings = _generation_settings(args, kwargs)
        calls.append(settings)
        if len(calls) == 1:
            raise RuntimeError("CUDA runtime disconnected")
        return _result(
            **settings,
        )

    monkeypatch.setattr(inference, "constrained_generate", generate)
    bundle = _obtain(row=row, bundle_path=bundle_path)

    assert len(calls) == 2
    assert calls[0] == calls[1]
    assert bundle["call_ordinal"] == 2
    first_start = inference.call_event_path(row, "candidate", 1, "started")
    first_failure = inference.call_event_path(row, "candidate", 1, "no_response")
    second_start = inference.call_event_path(row, "candidate", 2, "started")
    second_response = inference.call_event_path(row, "candidate", 2, "response_persisted")
    assert all(
        path.is_file()
        for path in (first_start, first_failure, second_start, second_response, bundle_path)
    )
    assert json.loads(first_failure.read_text(encoding="utf-8"))["retry_permitted"] is True


def test_persisted_response_is_reused_without_regeneration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _configure_paths(tmp_path, monkeypatch)
    row = _row()
    bundle_path = inference.output_paths(row)["candidate_bundle"]
    call_count = 0

    def generate(*args: Any, **kwargs: Any) -> inference.GenerationResult:
        nonlocal call_count
        call_count += 1
        return _result(**_generation_settings(args, kwargs))

    monkeypatch.setattr(inference, "constrained_generate", generate)
    first = _obtain(row=row, bundle_path=bundle_path)
    before = bundle_path.read_bytes()

    monkeypatch.setattr(
        inference,
        "constrained_generate",
        lambda *_args, **_kwargs: pytest.fail("durable response was regenerated"),
    )
    second = _obtain(
        row=row,
        bundle_path=bundle_path,
        launch_attempt_id="resume-launch",
    )

    assert call_count == 1
    assert second == first
    assert bundle_path.read_bytes() == before
    assert not inference.call_event_path(row, "candidate", 2, "started").exists()


def test_repository_commit_drift_in_retry_chain_fails_before_generation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _configure_paths(tmp_path, monkeypatch)
    row = _row()
    bundle_path = inference.output_paths(row)["candidate_bundle"]
    _write_retryable_first_call(row)
    monkeypatch.setattr(
        inference,
        "constrained_generate",
        lambda *_args, **_kwargs: pytest.fail("generation occurred after repository drift"),
    )

    with pytest.raises(RuntimeError, match="Call-event identity mismatch"):
        _obtain(
            row=row,
            bundle_path=bundle_path,
            repository_sha="d" * 40,
            launch_attempt_id="resume-launch",
        )

    assert not inference.call_event_path(row, "candidate", 2, "started").exists()
    assert not bundle_path.exists()


def test_changed_rendered_prompt_in_retry_chain_fails_before_generation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _configure_paths(tmp_path, monkeypatch)
    row = _row()
    bundle_path = inference.output_paths(row)["candidate_bundle"]
    _write_retryable_first_call(row)
    monkeypatch.setattr(
        inference,
        "constrained_generate",
        lambda *_args, **_kwargs: pytest.fail("generation occurred after request drift"),
    )

    with pytest.raises(RuntimeError, match="Call-event identity mismatch"):
        _obtain(
            row=row,
            bundle_path=bundle_path,
            rendered_prompt="changed candidate prompt",
            launch_attempt_id="resume-launch",
        )

    assert not inference.call_event_path(row, "candidate", 2, "started").exists()
    assert not bundle_path.exists()


def test_bundle_tampering_against_persisted_response_event_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _configure_paths(tmp_path, monkeypatch)
    row = _row()
    bundle_path = inference.output_paths(row)["candidate_bundle"]
    monkeypatch.setattr(
        inference,
        "constrained_generate",
        lambda *args, **kwargs: _result(**_generation_settings(args, kwargs)),
    )
    _obtain(row=row, bundle_path=bundle_path)

    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    bundle["generation"]["token_count"] = 999
    inference.write_json_atomic(bundle_path, bundle)
    monkeypatch.setattr(
        inference,
        "constrained_generate",
        lambda *_args, **_kwargs: pytest.fail("tampered bundle was regenerated"),
    )

    with pytest.raises(RuntimeError, match="differs from its persisted-response event"):
        _obtain(
            row=row,
            bundle_path=bundle_path,
            launch_attempt_id="resume-launch",
        )


@pytest.mark.parametrize(
    "missing_event",
    ["ordinal_one_started", "ordinal_one_terminal_before_ordinal_two"],
)
def test_existing_bundle_requires_complete_contiguous_call_event_chain(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    missing_event: str,
) -> None:
    _configure_paths(tmp_path, monkeypatch)
    row = _row()
    bundle_path = inference.output_paths(row)["candidate_bundle"]
    call_count = 0

    def generate(*args: Any, **kwargs: Any) -> inference.GenerationResult:
        nonlocal call_count
        call_count += 1
        if missing_event == "ordinal_one_terminal_before_ordinal_two" and call_count == 1:
            raise TimeoutError("runtime timed out")
        return _result(**_generation_settings(args, kwargs))

    monkeypatch.setattr(inference, "constrained_generate", generate)
    bundle = _obtain(row=row, bundle_path=bundle_path)
    if missing_event == "ordinal_one_started":
        assert bundle["call_ordinal"] == 1
        inference.call_event_path(row, "candidate", 1, "started").unlink()
    else:
        assert bundle["call_ordinal"] == 2
        inference.call_event_path(row, "candidate", 1, "no_response").unlink()

    monkeypatch.setattr(
        inference,
        "constrained_generate",
        lambda *_args, **_kwargs: pytest.fail("incomplete call chain was regenerated"),
    )
    with pytest.raises(RuntimeError, match="call chain|Call chain|call-event|Call-event"):
        _obtain(
            row=row,
            bundle_path=bundle_path,
            launch_attempt_id="resume-launch",
        )


def test_second_no_response_failure_is_terminal_across_resume(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _configure_paths(tmp_path, monkeypatch)
    row = _row()
    bundle_path = inference.output_paths(row)["candidate_bundle"]
    call_count = 0

    def fail(*_args: Any, **_kwargs: Any) -> inference.GenerationResult:
        nonlocal call_count
        call_count += 1
        raise TimeoutError("runtime timed out")

    monkeypatch.setattr(inference, "constrained_generate", fail)
    with pytest.raises(TimeoutError, match="timed out"):
        _obtain(row=row, bundle_path=bundle_path)
    assert call_count == 2
    second_failure_path = inference.call_event_path(row, "candidate", 2, "no_response")
    assert json.loads(second_failure_path.read_text(encoding="utf-8"))["retry_permitted"] is False

    with pytest.raises(
        RuntimeError,
        match="non-retryable no-response|exhausted its single infrastructure retry",
    ):
        _obtain(
            row=row,
            bundle_path=bundle_path,
            launch_attempt_id="resume-launch",
        )
    assert call_count == 2
    assert not bundle_path.exists()


@pytest.mark.parametrize(
    ("raw_text", "cap_hit", "error_type", "error_match"),
    [
        ('{"value":1}', True, RuntimeError, "token cap"),
        ('{"value":', False, inference.StrictGeneratedJSONError, "line 1"),
    ],
)
def test_durable_content_failure_is_terminal_and_never_regenerated(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    raw_text: str,
    cap_hit: bool,
    error_type: type[BaseException],
    error_match: str,
) -> None:
    _configure_paths(tmp_path, monkeypatch)
    _patch_row_prerequisites(tmp_path, monkeypatch)
    row = _row()
    paths = inference.output_paths(row)
    calls = 0

    def generate(*args: Any, **kwargs: Any) -> inference.GenerationResult:
        nonlocal calls
        calls += 1
        settings = _generation_settings(args, kwargs)
        return _result(
            raw_text,
            cap_hit=cap_hit,
            **settings,
        )

    monkeypatch.setattr(inference, "constrained_generate", generate)
    with pytest.raises(error_type, match=error_match):
        _run_one_row(row)
    assert paths["candidate_bundle"].is_file()
    durable_bytes = paths["candidate_bundle"].read_bytes()

    monkeypatch.setattr(
        inference,
        "constrained_generate",
        lambda *_args, **_kwargs: pytest.fail("durable content failure was regenerated"),
    )
    with pytest.raises(error_type, match=error_match):
        _run_one_row(row, launch_attempt_id="resume-launch")

    assert calls == 1
    assert paths["candidate_bundle"].read_bytes() == durable_bytes
    assert not inference.call_event_path(row, "candidate", 2, "started").exists()


def test_row_resume_reuses_candidate_and_continues_with_audit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _configure_paths(tmp_path, monkeypatch)
    _patch_row_prerequisites(tmp_path, monkeypatch)
    row = _row()
    paths = inference.output_paths(row)
    generation_passes: list[str] = []

    def generate(*args: Any, **kwargs: Any) -> inference.GenerationResult:
        settings = _generation_settings(args, kwargs)
        pass_name = (
            "candidate"
            if settings["max_new_tokens"] == inference.CANDIDATE_MAX_NEW_TOKENS
            else "audit"
        )
        generation_passes.append(pass_name)
        return _result(**settings)

    monkeypatch.setattr(inference, "constrained_generate", generate)
    real_obtain = inference.obtain_response_bundle

    def stop_before_audit(**kwargs: Any) -> dict[str, Any]:
        if kwargs["pass_name"] == "audit":
            raise KeyboardInterrupt("simulated process interruption before audit call")
        return real_obtain(**kwargs)

    monkeypatch.setattr(inference, "obtain_response_bundle", stop_before_audit)
    with pytest.raises(KeyboardInterrupt, match="before audit"):
        _run_one_row(row)

    assert generation_passes == ["candidate"]
    assert paths["candidate_bundle"].is_file()
    assert paths["candidate_parsed"].is_file()
    assert not paths["audit_bundle"].exists()

    monkeypatch.setattr(inference, "obtain_response_bundle", real_obtain)
    assert _run_one_row(row, launch_attempt_id="resume-launch") is True

    assert generation_passes == ["candidate", "audit"]
    assert paths["audit_bundle"].is_file()
    assert paths["record"].is_file()
    assert not inference.call_event_path(row, "candidate", 2, "started").exists()


def test_run_record_binds_every_artifact_descriptor_and_detects_tampering(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _configure_paths(tmp_path, monkeypatch)
    _patch_row_prerequisites(tmp_path, monkeypatch)
    row = _row()
    paths = inference.output_paths(row)

    def generate(*args: Any, **kwargs: Any) -> inference.GenerationResult:
        return _result(**_generation_settings(args, kwargs))

    monkeypatch.setattr(inference, "constrained_generate", generate)
    assert _run_one_row(row) is True

    record = json.loads(paths["record"].read_text(encoding="utf-8"))
    assert set(record["artifacts"]) == set(paths) - {"record"}
    for name, path in paths.items():
        if name != "record":
            assert record["artifacts"][name] == inference.artifact_metadata(path)

    paths["candidate_parsed"].write_text('{"value":2}\n', encoding="utf-8")
    with pytest.raises(RuntimeError, match="artifact mismatch"):
        inference.verify_complete_record(
            paths,
            row,
            REPOSITORY_SHA,
            _protocol_lock(),
        )
