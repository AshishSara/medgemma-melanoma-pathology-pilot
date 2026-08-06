from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import evaluate_v4 as evaluator  # noqa: E402


def test_generation_telemetry_recomputes_cap_consumption_from_token_count() -> None:
    generated_ids = list(range(evaluator.CANDIDATE_MAX_NEW_TOKENS))
    serialized_ids = json.dumps(
        generated_ids,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")
    raw_text = "{}"
    artifact = {
        "pass": "candidate",
        "raw_text": {
            "bytes": len(raw_text.encode("utf-8")),
            "sha256": evaluator.hashlib.sha256(raw_text.encode("utf-8")).hexdigest(),
        },
        "elapsed_seconds": 1.0,
        "prompt_token_count": 10,
        "generated_token_ids": generated_ids,
        "generated_token_ids_sha256": evaluator.hashlib.sha256(serialized_ids).hexdigest(),
        "eos_token_ids": [9999],
        "token_count": evaluator.CANDIDATE_MAX_NEW_TOKENS,
        "eos_observed": False,
        # This contradicts the mechanical token count and must fail closed.
        "cap_hit": False,
        "max_new_tokens": evaluator.CANDIDATE_MAX_NEW_TOKENS,
        "serializer": {
            "configuration_stage": "after_transformers_prefix_construction",
            "tokenizer_alphabet_preserved": True,
            "force_json_field_order": False,
            "max_consecutive_whitespaces": 12,
            "max_json_array_length": 20,
        },
        "decoding": {
            "do_sample": False,
            "num_beams": 1,
            "skip_special_tokens": True,
        },
    }

    issues = evaluator._generation_telemetry_issues(
        pass_name="candidate",
        raw_text=raw_text,
        artifact=artifact,
        record_value=artifact,
    )

    assert any("cap" in issue.lower() for issue in issues)


def _attempt_context(
    lock_path: Path,
    manifest_path: Path,
    *,
    attempt_id: str,
    repository_commit: str,
    protocol_source_commit: str,
) -> dict[str, Any]:
    return {
        "execution_attempt_id": attempt_id,
        "protocol_version": evaluator.PROTOCOL_VERSION,
        "protocol_lock_path": "data/v4/protocol_lock.json",
        "protocol_lock_sha256": evaluator.sha256_file(lock_path),
        "protocol_source_commit": protocol_source_commit,
        "repository_commit": repository_commit,
        "manifest_path": "data/v4/report_manifest.csv",
        "manifest_sha256": evaluator.sha256_file(manifest_path),
        "selected_row_count": evaluator.FORMAL_OUTPUT_COUNT,
        "condition": "all",
    }


def _write_event(
    attempts_dir: Path,
    attempt_id: str,
    event_name: str,
    payload: dict[str, Any],
) -> None:
    (attempts_dir / f"{attempt_id}.{event_name}.json").write_text(
        json.dumps(payload),
        encoding="utf-8",
    )


def test_execution_ledger_requires_one_completed_attempt_and_forbids_resume(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lock_path = tmp_path / "data" / "v4" / "protocol_lock.json"
    manifest_path = tmp_path / "data" / "v4" / "report_manifest.csv"
    attempts_dir = tmp_path / "results" / "v4" / "formal" / "execution_attempts"
    lock_path.parent.mkdir(parents=True)
    lock_path.write_text("{}\n", encoding="utf-8")
    manifest_path.write_text("manifest\n", encoding="utf-8")
    attempts_dir.mkdir(parents=True)
    monkeypatch.setattr(evaluator, "ROOT", tmp_path)
    monkeypatch.setattr(evaluator, "PROTOCOL_LOCK_PATH", lock_path)
    monkeypatch.setattr(evaluator, "V4_MANIFEST", manifest_path)

    repository_commit = "a" * 40
    protocol_lock = {"protocol_source_commit": "b" * 40}
    attempt_id = evaluator.FORMAL_EXECUTION_ATTEMPT_ID
    context = _attempt_context(
        lock_path,
        manifest_path,
        attempt_id=attempt_id,
        repository_commit=repository_commit,
        protocol_source_commit=protocol_lock["protocol_source_commit"],
    )
    _write_event(
        attempts_dir,
        attempt_id,
        "started",
        {
            **context,
            "status": "started",
            "timestamp_utc": "2026-08-04T12:00:00+00:00",
        },
    )
    _write_event(
        attempts_dir,
        attempt_id,
        "completed",
        {
            **context,
            "status": "completed",
            "timestamp_utc": "2026-08-04T12:30:00+00:00",
            "completed_new_row_count": evaluator.FORMAL_OUTPUT_COUNT,
            "verified_existing_row_count": 0,
        },
    )
    preflight = [
        {
            "record": {
                "attempt_status": "completed",
                "execution_attempt_id": attempt_id,
                "repository_commit": repository_commit,
            }
        }
        for _ in range(evaluator.FORMAL_OUTPUT_COUNT)
    ]

    ledger, issues = evaluator._validate_execution_attempt_ledger(
        preflight,
        protocol_lock,
    )
    assert issues == []
    assert ledger["attempt_count"] == 1
    assert ledger["completed_attempt_count"] == 1

    second_id = "attempt-two"
    second_context = _attempt_context(
        lock_path,
        manifest_path,
        attempt_id=second_id,
        repository_commit=repository_commit,
        protocol_source_commit=protocol_lock["protocol_source_commit"],
    )
    _write_event(
        attempts_dir,
        second_id,
        "started",
        {
            **second_context,
            "status": "started",
            "timestamp_utc": "2026-08-04T13:00:00+00:00",
        },
    )
    failed_path = attempts_dir / f"{second_id}.failed.json"
    failed_payload = {
        **second_context,
        "status": "failed",
        "timestamp_utc": "2026-08-04T13:01:00+00:00",
        "completed_new_row_count": 0,
        "verified_existing_row_count": 0,
        "resume_permitted_for_current_row": False,
    }
    _write_event(attempts_dir, second_id, "failed", failed_payload)

    _, repeated_attempt_issues = evaluator._validate_execution_attempt_ledger(
        preflight,
        protocol_lock,
    )
    assert any("exactly one attempt total" in issue for issue in repeated_attempt_issues)

    failed_payload["resume_permitted_for_current_row"] = True
    failed_path.write_text(json.dumps(failed_payload), encoding="utf-8")
    _, resume_issues = evaluator._validate_execution_attempt_ledger(
        preflight,
        protocol_lock,
    )
    assert any("violates the no-resume rule" in issue for issue in resume_issues)


@pytest.mark.parametrize(
    ("invalid_row_index", "ledger_issues"),
    [
        (19, []),
        (None, ["formal execution ledger must contain exactly one attempt total"]),
    ],
)
def test_evaluation_never_reads_formal_truth_before_complete_sealed_preflight(
    monkeypatch: pytest.MonkeyPatch,
    invalid_row_index: int | None,
    ledger_issues: list[str],
) -> None:
    manifest = [
        {
            "document_id": f"MEL-{index:03d}-A",
            "condition": "clean" if index % 2 == 0 else "ocr_degraded",
            "ground_truth_path": f"data/v4/ground_truth/MEL-{index:03d}-A.json",
        }
        for index in range(evaluator.FORMAL_OUTPUT_COUNT)
    ]
    protocol_lock = {
        "protocol_source_commit": "b" * 40,
        "runtime": {},
    }
    writes: list[dict[str, Any]] = []
    truth_reads: list[Path] = []

    monkeypatch.setattr(evaluator, "verify_protocol_lock", lambda: protocol_lock)
    monkeypatch.setattr(evaluator, "_load_formal_manifest", lambda: manifest)

    def preflight(row: dict[str, str], _lock: dict[str, Any]) -> dict[str, Any]:
        index = manifest.index(row)
        return {
            "row": dict(row),
            "record": {"repository_commit": "a" * 40},
            "issues": (
                ["missing sealed formal output"]
                if invalid_row_index is not None and index == invalid_row_index
                else []
            ),
        }

    monkeypatch.setattr(evaluator, "_preflight_row", preflight)
    monkeypatch.setattr(
        evaluator,
        "_validate_execution_attempt_ledger",
        lambda _preflight, _lock: (
            {"attempt_count": 1, "completed_attempt_count": 1, "events": []},
            ledger_issues,
        ),
    )

    def guarded_read_json(path: Path) -> Any:
        if "ground_truth" in Path(path).parts:
            truth_reads.append(Path(path))
            raise AssertionError("formal ground truth was read before preflight sealed")
        raise AssertionError(f"unexpected read_json call: {path}")

    monkeypatch.setattr(evaluator, "read_json", guarded_read_json)
    monkeypatch.setattr(
        evaluator,
        "_write_output_seal",
        lambda *_args: pytest.fail("output seal must not be written for incomplete provenance"),
    )
    monkeypatch.setattr(
        evaluator,
        "_verify_evaluation_only_corpus",
        lambda *_args: pytest.fail("truth-side hashes must not be opened before sealing"),
    )
    monkeypatch.setattr(
        evaluator,
        "write_json",
        lambda _path, value: writes.append(dict(value)),
    )

    payload = evaluator.evaluate()

    assert payload["pass"] is False
    assert payload["ground_truth_read"] is False
    assert payload["provenance_valid_count"] == evaluator.FORMAL_OUTPUT_COUNT - 1
    assert truth_reads == []
    assert writes == [payload]
