from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import evaluate_v3  # noqa: E402
from evaluate_v3 import _evidence_summary  # noqa: E402


def decision(
    *,
    accepted: bool = True,
    reason: str = "accepted",
    section: str = "current",
    normalization: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "candidate_value": "candidate",
        "candidate_omitted": False,
        "accepted": accepted,
        "reason": reason,
        "final_value": "final" if accepted else None,
        "evidence_line_ids": ["L001"],
        "evidence_sections": [section],
        "parsed_evidence_values": ["parsed"],
        "normalization": normalization,
    }


def summary_for(fields: dict[str, dict[str, Any]]) -> dict[str, Any]:
    return _evidence_summary(
        [
            {
                "compiler_log": {
                    "fields": fields,
                    "history_evidence_rejection_count": 0,
                }
            }
        ]
    )


def test_evidence_summary_accepts_all_prespecified_successful_decisions() -> None:
    fields = {
        "document_id": decision(
            reason="accepted_labeled_header_identity_fallback",
            section="header",
            normalization={
                "type": "labeled_header_identity_fallback",
                "source_field": "document_id",
            },
        ),
        "diagnosis": decision(),
        "specimen_site": decision(
            reason="accepted_bounded_site_ocr_substitution",
            normalization={
                "type": "single_confusable_glyph_substitution",
                "token_index": 1,
                "character_index": 4,
                "candidate_token": "shoulder",
                "ocr_token": "shou1der",
                "candidate_character": "l",
                "ocr_character": "1",
            },
        ),
        "laterality": decision(
            reason="accepted_from_specimen_site_evidence",
            normalization={
                "type": "linked_field_evidence",
                "source_field": "specimen_site",
            },
        ),
        "breslow_qualifier": decision(
            reason="derived_from_accepted_measurement_evidence",
            normalization={
                "type": "linked_measurement_qualifier",
                "source_field": "breslow_thickness_mm",
            },
        ),
        "mitotic_qualifier": decision(
            reason="derived_from_accepted_measurement_evidence",
            normalization={
                "type": "linked_measurement_qualifier",
                "source_field": "mitotic_rate_per_mm2",
            },
        ),
    }

    summary = summary_for(fields)

    # document_id is an artifact binding, not one of the 16 clinical fields.
    assert summary["accepted_non_null_count"] == 5
    assert summary["accepted_with_valid_non_history_evidence_count"] == 5
    assert summary["accepted_with_valid_non_history_evidence_rate"] == 1.0
    assert summary["history_carryover_count"] == 0


def test_evidence_summary_uses_acceptance_and_structure_not_reason_text() -> None:
    accepted_with_unrecognized_reason = decision(reason="future_compiler_success_reason")
    rejected_with_success_reason = decision(accepted=False, reason="accepted")

    summary = summary_for(
        {
            "diagnosis": accepted_with_unrecognized_reason,
            "ulceration": rejected_with_success_reason,
        }
    )

    assert summary["accepted_non_null_count"] == 1
    assert summary["accepted_with_valid_non_history_evidence_count"] == 1
    assert summary["accepted_with_valid_non_history_evidence_rate"] == 1.0
    assert summary["rejected_non_null_count"] == 1


def test_evidence_summary_rejects_malformed_or_history_evidence() -> None:
    duplicate_lines = decision()
    duplicate_lines["evidence_line_ids"] = ["L001", "L001"]
    duplicate_lines["evidence_sections"] = ["current", "current"]
    history = decision(section="history")
    empty_parse = decision()
    empty_parse["parsed_evidence_values"] = []
    invalid_link = decision(
        normalization={
            "type": "linked_field_evidence",
            "source_field": "diagnosis",
        }
    )
    invalid_substitution = decision(
        normalization={
            "type": "single_confusable_glyph_substitution",
            "token_index": 0,
            "character_index": 0,
            "candidate_token": "arm",
            "ocr_token": "xrm",
            "candidate_character": "a",
            "ocr_character": "x",
        }
    )

    summary = summary_for(
        {
            "diagnosis": duplicate_lines,
            "ulceration": history,
            "staging.pT": empty_parse,
            "laterality": invalid_link,
            "specimen_site": invalid_substitution,
        }
    )

    assert summary["accepted_non_null_count"] == 5
    assert summary["accepted_with_valid_non_history_evidence_count"] == 0
    assert summary["accepted_with_valid_non_history_evidence_rate"] == 0.0
    assert summary["history_carryover_count"] == 1


def test_formal_execution_ledger_is_complete_context_bound_and_hash_sealed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lock_path = tmp_path / "data" / "v3" / "protocol_lock.json"
    manifest_path = tmp_path / "data" / "v3" / "report_manifest.csv"
    attempts_dir = tmp_path / "results" / "v3" / "formal" / "execution_attempts"
    for path, content in (
        (lock_path, "{}\n"),
        (manifest_path, "manifest\n"),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    attempts_dir.mkdir(parents=True)
    monkeypatch.setattr(evaluate_v3, "ROOT", tmp_path)
    monkeypatch.setattr(evaluate_v3, "PROTOCOL_LOCK_PATH", lock_path)
    monkeypatch.setattr(evaluate_v3, "V3_MANIFEST", manifest_path)
    output_seal_path = tmp_path / "results" / "v3" / "formal" / "output_seal.json"
    monkeypatch.setattr(evaluate_v3, "OUTPUT_SEAL_PATH", output_seal_path)

    attempt_id = "20260804T120000.000000Z-pid1"
    repository_commit = "a" * 40
    protocol_lock = {"protocol_source_commit": "b" * 40}
    context = {
        "execution_attempt_id": attempt_id,
        "protocol_version": evaluate_v3.PROTOCOL_VERSION,
        "protocol_lock_path": "data/v3/protocol_lock.json",
        "protocol_lock_sha256": evaluate_v3.sha256_file(lock_path),
        "protocol_source_commit": protocol_lock["protocol_source_commit"],
        "repository_commit": repository_commit,
        "manifest_path": "data/v3/report_manifest.csv",
        "manifest_sha256": evaluate_v3.sha256_file(manifest_path),
        "selected_row_count": evaluate_v3.FORMAL_OUTPUT_COUNT,
        "condition": "all",
    }
    started = {
        **context,
        "status": "started",
        "timestamp_utc": "2026-08-04T12:00:00+00:00",
    }
    completed = {
        **context,
        "status": "completed",
        "timestamp_utc": "2026-08-04T12:30:00+00:00",
        "completed_new_row_count": evaluate_v3.FORMAL_OUTPUT_COUNT,
        "verified_existing_row_count": 0,
    }
    for event_name, event in (("started", started), ("completed", completed)):
        (attempts_dir / f"{attempt_id}.{event_name}.json").write_text(
            json.dumps(event),
            encoding="utf-8",
        )
    preflight = []
    for index in range(evaluate_v3.FORMAL_OUTPUT_COUNT):
        record_path = (
            tmp_path / "results" / "v3" / "formal" / "run_records" / f"record-{index:02d}.json"
        )
        record_path.parent.mkdir(parents=True, exist_ok=True)
        record_path.write_text("{}\n", encoding="utf-8")
        preflight.append(
            {
                "row": {
                    "document_id": f"MEL-{index:03d}-A",
                    "condition": "clean" if index % 2 == 0 else "ocr_degraded",
                },
                "paths": {"record": record_path},
                "record": {
                    "attempt_status": "completed",
                    "execution_attempt_id": attempt_id,
                    "repository_commit": repository_commit,
                    "artifacts": {},
                },
            }
        )

    ledger, issues = evaluate_v3._validate_execution_attempt_ledger(
        preflight,
        protocol_lock,
    )

    assert issues == []
    assert ledger["attempt_count"] == 1
    assert ledger["completed_attempt_count"] == 1
    assert {event["event"] for event in ledger["events"]} == {"started", "completed"}
    assert all(len(event["sha256"]) == 64 for event in ledger["events"])
    seal = evaluate_v3._write_output_seal(preflight, ledger)
    assert seal["execution_ledger"] == ledger
    assert json.loads(output_seal_path.read_text(encoding="utf-8"))["execution_ledger"] == ledger

    (attempts_dir / f"{attempt_id}.completed.json").unlink()
    _, missing_terminal_issues = evaluate_v3._validate_execution_attempt_ledger(
        preflight,
        protocol_lock,
    )
    assert any(
        "exactly one failed/completed terminal" in issue for issue in missing_terminal_issues
    )


def test_formal_execution_ledger_rejects_unbound_or_incomplete_records(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lock_path = tmp_path / "data" / "v3" / "protocol_lock.json"
    manifest_path = tmp_path / "data" / "v3" / "report_manifest.csv"
    attempts_dir = tmp_path / "results" / "v3" / "formal" / "execution_attempts"
    for path, content in ((lock_path, "{}\n"), (manifest_path, "manifest\n")):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    attempts_dir.mkdir(parents=True)
    monkeypatch.setattr(evaluate_v3, "ROOT", tmp_path)
    monkeypatch.setattr(evaluate_v3, "PROTOCOL_LOCK_PATH", lock_path)
    monkeypatch.setattr(evaluate_v3, "V3_MANIFEST", manifest_path)

    preflight = [
        {
            "record": {
                "attempt_status": "partial",
                "execution_attempt_id": "../unsafe",
                "repository_commit": "a" * 40,
            }
        }
        for _ in range(evaluate_v3.FORMAL_OUTPUT_COUNT)
    ]
    _, issues = evaluate_v3._validate_execution_attempt_ledger(
        preflight,
        {"protocol_source_commit": "b" * 40},
    )

    assert any("not marked completed" in issue for issue in issues)
    assert any("invalid execution_attempt_id" in issue for issue in issues)
    assert any("ledger is empty" in issue for issue in issues)
