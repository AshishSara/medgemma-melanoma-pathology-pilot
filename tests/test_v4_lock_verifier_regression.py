from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import evaluate_v4 as evaluator  # noqa: E402
import freeze_v4_protocol as freezer  # noqa: E402
import run_v4_inference as runner  # noqa: E402

PASSING_SMOKE_CRITERIA = {
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


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def _development_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[dict[str, Any], dict[str, Path]]:
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    manifest_path = tmp_path / "data" / "v4" / "report_manifest.csv"
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text("manifest\n", encoding="utf-8")
    monkeypatch.setattr(runner, "V4_MANIFEST", manifest_path)

    development_commit = "a" * 40
    base = tmp_path / "results" / "v4" / "development" / "iteration-1"
    metrics_path = base / "metrics.json"
    iteration_path = base / "iteration_record.json"
    claim_path = base / "execution_attempts" / f"{runner.DEVELOPMENT_ATTEMPT_ID}.claimed.json"
    _write_json(
        metrics_path,
        {
            "status": "DEVELOPMENT_PASS",
            "protocol_version": runner.PROTOCOL_VERSION,
            "iteration": 1,
            "output_count": 4,
            "provenance_valid_count": 4,
            "pass": True,
            "model_id": runner.MODEL_ID,
            "model_revision": runner.MODEL_REVISION,
            "smoke_criteria": PASSING_SMOKE_CRITERIA,
            "generation_telemetry": {
                "candidate": {"cap_hit_count": 0},
                "audit": {"cap_hit_count": 0},
            },
        },
    )
    _write_json(
        iteration_path,
        {
            "status": "DEVELOPMENT_PASS",
            "protocol_version": runner.PROTOCOL_VERSION,
            "iteration": 1,
            "repository_commit": development_commit,
            "metrics_path": ("results/v4/development/iteration-1/metrics.json"),
            "metrics_sha256": runner.sha256_file(metrics_path),
        },
    )
    _write_json(
        claim_path,
        {
            "execution_attempt_id": runner.DEVELOPMENT_ATTEMPT_ID,
            "status": "claimed",
            "protocol_version": runner.PROTOCOL_VERSION,
            "repository_commit": development_commit,
            "manifest_path": "data/v4/report_manifest.csv",
            "manifest_sha256": runner.sha256_file(manifest_path),
            "selected_row_count": 4,
            "condition": "all",
        },
    )
    artifacts = {
        path.relative_to(tmp_path).as_posix(): runner.sha256_file(path)
        for path in (metrics_path, iteration_path, claim_path)
    }
    gate = {
        "status": "PASS",
        "criteria": PASSING_SMOKE_CRITERIA,
        "provenance_valid_count": 4,
        "canonical_schema_valid_rate": 1.0,
        "pooled_field_exact_match": 0.0,
        "condition_field_exact_match": {
            "clean": 0.0,
            "ocr_degraded": 0.0,
        },
        "unsupported_field_rate": 0.0,
        "unsupported_field_count": 0,
        "generation_cap_hit_count": {
            "candidate": 0,
            "audit": 0,
        },
        "accuracy_selection_note": (
            "Exact-match accuracy is descriptive only and is not a development selection threshold."
        ),
        "accepted_non_null_count": 2,
        "accepted_with_valid_non_history_evidence_count": 2,
        "accepted_non_null_evidence_rate": 1.0,
        "history_carryover_count": 0,
    }
    lock = {
        "development_gate": gate,
        "development_artifacts": artifacts,
        "development_metrics_sha256": artifacts["results/v4/development/iteration-1/metrics.json"],
        "development_iteration_record_sha256": artifacts[
            "results/v4/development/iteration-1/iteration_record.json"
        ],
        "development_iterations": [
            {
                "iteration": 1,
                "status": "selected_complete",
                "documentation_path": ("results/v4/development/iteration-1/iteration_record.json"),
                "documentation_sha256": artifacts[
                    "results/v4/development/iteration-1/iteration_record.json"
                ],
                "artifact_count": len(artifacts),
            }
        ],
        "development_repository_commit": development_commit,
    }
    return lock, {
        "metrics": metrics_path,
        "iteration": iteration_path,
        "claim": claim_path,
    }


def test_development_lock_accepts_exact_pass_and_rejects_gate_tampering(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lock, _ = _development_lock(tmp_path, monkeypatch)
    assert runner.development_lock_errors(lock) == []

    tampered = copy.deepcopy(lock)
    tampered["development_gate"]["criteria"]["zero_audit_cap_hits"] = False
    errors = runner.development_lock_errors(tampered)
    assert any("criteria" in error for error in errors)


@pytest.mark.parametrize("artifact_name", ["metrics", "claim"])
def test_development_lock_rejects_mutated_or_missing_bound_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    artifact_name: str,
) -> None:
    lock, paths = _development_lock(tmp_path, monkeypatch)
    if artifact_name == "metrics":
        paths[artifact_name].write_text("tampered\n", encoding="utf-8")
    else:
        paths[artifact_name].unlink()

    errors = runner.development_lock_errors(lock)

    assert errors
    assert any(
        term in error for error in errors for term in ("hash mismatch", "inventory", "missing")
    )


def _base_protocol_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[dict[str, Any], Path]:
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    manifest_path = tmp_path / "data" / "v4" / "report_manifest.csv"
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text(
        "document_id,condition,pdf_path,image_path,source_text_path,ground_truth_path\n",
        encoding="utf-8",
    )
    lock_path = tmp_path / "data" / "v4" / "protocol_lock.json"
    monkeypatch.setattr(runner, "V4_MANIFEST", manifest_path)
    monkeypatch.setattr(runner, "PROTOCOL_LOCK_PATH", lock_path)
    monkeypatch.setattr(
        runner,
        "REQUIRED_LOCKED_SOURCE_PATHS",
        ("data/v4/report_manifest.csv",),
    )
    monkeypatch.setattr(runner, "development_lock_errors", lambda _lock: [])
    monkeypatch.setattr(
        runner,
        "v3_provenance_lock_errors",
        lambda _lock, _rows: [],
    )
    monkeypatch.setattr(
        runner,
        "post_development_immutability_lock_errors",
        lambda _lock: [],
    )
    monkeypatch.setattr(
        runner.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(
            returncode=0,
            stdout="",
            stderr="",
        ),
    )
    monkeypatch.setattr(
        runner.subprocess,
        "check_output",
        lambda *_args, **_kwargs: lock_path.read_bytes(),
    )
    manifest_sha = runner.sha256_file(manifest_path)
    lock = {
        "status": "frozen",
        "protocol_version": runner.PROTOCOL_VERSION,
        "protocol_identifier": runner.PROTOCOL_IDENTIFIER,
        "protocol_source_commit": "b" * 40,
        "model_id": runner.MODEL_ID,
        "model_revision": runner.MODEL_REVISION,
        "dtype": runner.DTYPE_NAME,
        "random_seed": runner.RANDOM_SEED,
        "candidate_max_new_tokens": runner.CANDIDATE_MAX_NEW_TOKENS,
        "audit_max_new_tokens": runner.AUDIT_MAX_NEW_TOKENS,
        "candidate_serializer_force_json_field_order": False,
        "candidate_serializer_max_consecutive_whitespaces": 12,
        "audit_serializer_force_json_field_order": True,
        "audit_serializer_max_consecutive_whitespaces": 0,
        "tesseract_language": runner.TESSERACT_LANGUAGE,
        "tesseract_config": runner.TESSERACT_CONFIG,
        "formal_output_count": 20,
        "development_iterations_used": 1,
        "selected_development_iteration": 1,
        "candidate_calls_per_input": 1,
        "audit_calls_per_input": 1,
        "formal_execution_attempt_id": runner.FORMAL_EXECUTION_ATTEMPT_ID,
        "development_attempt_id": runner.DEVELOPMENT_ATTEMPT_ID,
        "assistant_prefill": False,
        "do_sample": False,
        "num_beams": 1,
        "locked_source_files": {
            "data/v4/report_manifest.csv": manifest_sha,
        },
        "corpus_artifacts": {
            "data/v4/report_manifest.csv": manifest_sha,
        },
        "evaluation_only_corpus_artifacts": {
            "data/v4/cases.csv": "c" * 64,
        },
        "development_repository_commit": "a" * 40,
        "runtime": {
            "resolved_revision": runner.MODEL_REVISION,
            "device": "cuda:0",
            "text_vocab_size": 262208,
            "random_seed": 0,
            "accelerate_version": "1.14.0",
            "torch_version": "2.11.0+cu128",
            "transformers_version": "4.57.6",
            "lm_format_enforcer_version": "0.11.3",
            "pytesseract_version": "0.3.13",
            "tesseract_version": "4.1.1",
            "cuda_version": "12.8",
            "cuda_device_name": "Tesla T4",
            "dtype": "torch.bfloat16",
            "python_version": "3.12.13",
            "platform": "Linux-test",
        },
        "schema_enforcement": freezer.SCHEMA_ENFORCEMENT,
        "formal_execution_order": freezer.FORMAL_EXECUTION_ORDER,
        "infrastructure_retry_rule": freezer.INFRASTRUCTURE_RETRY_RULE,
        "pass_criteria": freezer.PASS_CRITERIA,
    }
    return lock, lock_path


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("protocol_identifier", None),
        ("protocol_identifier", "wrong-protocol"),
        ("selected_development_iteration", 2),
    ],
)
def test_top_level_lock_verifier_rejects_identity_or_iteration_tampering(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: Any,
) -> None:
    lock, lock_path = _base_protocol_lock(tmp_path, monkeypatch)
    if value is None:
        del lock[field]
    else:
        lock[field] = value
    _write_json(lock_path, lock)
    original_sha256_file = runner.sha256_file

    def truth_firewall(path: Path) -> str:
        relative_path = Path(path).relative_to(tmp_path).as_posix()
        if relative_path == "data/v4/cases.csv" or relative_path.startswith(
            ("data/v4/source_text/", "data/v4/ground_truth/")
        ):
            raise AssertionError(f"formal truth-side file was opened: {relative_path}")
        return original_sha256_file(Path(path))

    monkeypatch.setattr(runner, "sha256_file", truth_firewall)

    with pytest.raises(SystemExit) as exc_info:
        runner.verify_protocol_lock()

    assert field in str(exc_info.value)


def _real_v3_lock_fixture() -> tuple[list[dict[str, str]], dict[str, Any]]:
    rows = freezer.load_and_validate_manifest(ROOT)
    provenance, _ = freezer.collect_v3_reuse_provenance(ROOT, rows)
    corpus = freezer.partition_corpus_artifacts(
        rows,
        freezer.collect_corpus_artifacts(ROOT, rows),
    )
    return rows, {
        "v3_boundary_and_reuse_provenance": provenance,
        "corpus_artifacts": corpus["operational"],
        "evaluation_only_corpus_artifacts": corpus["evaluation_only"],
    }


@pytest.mark.parametrize(
    "section",
    [
        "v3_formal_attempt_events",
        "v3_formal_model_artifacts",
    ],
)
def test_v3_provenance_rejects_tampered_bound_hash_without_v4_truth_reads(
    monkeypatch: pytest.MonkeyPatch,
    section: str,
) -> None:
    rows, lock = _real_v3_lock_fixture()
    original_open = Path.open

    def truth_firewall(
        path: Path,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        try:
            relative_path = path.relative_to(ROOT).as_posix()
        except ValueError:
            relative_path = ""
        if relative_path == "data/v4/cases.csv" or relative_path.startswith(
            (
                "data/v4/source_text/",
                "data/v4/ground_truth/",
                "data/v3/source_text/",
                "data/v3/ground_truth/",
            )
        ):
            raise AssertionError(f"formal truth-side file was opened: {relative_path}")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", truth_firewall)
    assert runner.v3_provenance_lock_errors(lock, rows) == []

    tampered = copy.deepcopy(lock)
    mapping = tampered["v3_boundary_and_reuse_provenance"][section]
    first_path = next(iter(mapping))
    mapping[first_path] = "0" * 64
    errors = runner.v3_provenance_lock_errors(tampered, rows)

    assert any("mismatch" in error for error in errors)


def test_post_seal_evaluator_hashes_both_v3_and_v4_truth_aliases(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _rows, lock = _real_v3_lock_fixture()
    original_sha256_file = evaluator.sha256_file
    observed: list[str] = []

    def recording_sha256(path: Path) -> str:
        relative_path = Path(path).relative_to(ROOT).as_posix()
        if relative_path.startswith(
            (
                "data/v3/source_text/",
                "data/v3/ground_truth/",
                "data/v4/source_text/",
                "data/v4/ground_truth/",
            )
        ):
            observed.append(relative_path)
        return original_sha256_file(Path(path))

    monkeypatch.setattr(evaluator, "sha256_file", recording_sha256)

    verified = evaluator._verify_v3_truth_side_reuse_aliases(lock)

    assert observed
    assert set(observed) == set(verified)
    assert any(path.startswith("data/v3/source_text/") for path in observed)
    assert any(path.startswith("data/v3/ground_truth/") for path in observed)
    assert any(path.startswith("data/v4/source_text/") for path in observed)
    assert any(path.startswith("data/v4/ground_truth/") for path in observed)
