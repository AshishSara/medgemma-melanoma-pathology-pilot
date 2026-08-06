from __future__ import annotations

import csv
import hashlib
import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import evaluate_v5 as evaluator  # noqa: E402, I001


EXPECTED_THRESHOLDS = {
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


def _valid_generation(pass_name: str, raw_text: str = "{}") -> dict[str, Any]:
    maximum = (
        evaluator.CANDIDATE_MAX_NEW_TOKENS
        if pass_name == "candidate"
        else evaluator.AUDIT_MAX_NEW_TOKENS
    )
    token_ids = [1, 2, 3]
    token_bytes = json.dumps(token_ids, separators=(",", ":"), ensure_ascii=True).encode("ascii")
    return {
        "pass": pass_name,
        "max_new_tokens": maximum,
        "token_count": len(token_ids),
        "cap_hit": False,
        "prompt_token_count": 100,
        "elapsed_seconds": 1.0,
        "generated_token_ids": token_ids,
        "generated_token_ids_sha256": hashlib.sha256(token_bytes).hexdigest(),
        "eos_observed": True,
        "raw_text": evaluator._expected_text_descriptor(raw_text),
        "decoding": {
            "do_sample": False,
            "num_beams": 1,
            "skip_special_tokens": True,
        },
        "serializer": {
            "configuration_stage": "after_transformers_prefix_construction",
            "tokenizer_alphabet_preserved": True,
            "force_json_field_order": (
                evaluator.CANDIDATE_SERIALIZER_FORCE_JSON_FIELD_ORDER
                if pass_name == "candidate"
                else evaluator.AUDIT_SERIALIZER_FORCE_JSON_FIELD_ORDER
            ),
            "max_consecutive_whitespaces": (
                evaluator.CANDIDATE_SERIALIZER_MAX_CONSECUTIVE_WHITESPACES
                if pass_name == "candidate"
                else evaluator.AUDIT_SERIALIZER_MAX_CONSECUTIVE_WHITESPACES
            ),
            "max_json_array_length": 20,
        },
    }


def _manifest_rows(split: str, count: int) -> list[dict[str, str]]:
    return [
        {
            "dataset_version": "pilot-v5",
            "generator_version": "5.0.0",
            "split": split,
            "semantic_case_id": f"MEL-{900 + index // 4}",
            "document_id": f"MEL-{900 + index // 2}-{'A' if index % 2 == 0 else 'B'}",
            "template_id": "A" if index % 2 == 0 else "B",
            "condition": "clean" if index < count // 2 else "ocr_degraded",
            "image_path": f"output/v5/rendered/{'clean' if index < count // 2 else 'ocr_degraded'}/"
            f"MEL-{900 + index // 2}-{'A' if index % 2 == 0 else 'B'}.png",
            "image_sha256": "a" * 64,
            "ground_truth_path": f"data/v5/ground_truth/fake-{index}.json",
        }
        for index in range(count)
    ]


def test_gate_thresholds_are_exact_protocol_counts() -> None:
    assert evaluator.GATE_THRESHOLDS == EXPECTED_THRESHOLDS
    assert len(evaluator.PRIMARY_FIELD_PATHS) == 16


def test_generation_validation_recomputes_cap_hit() -> None:
    generation = _valid_generation("candidate")
    maximum = evaluator.CANDIDATE_MAX_NEW_TOKENS
    generation["generated_token_ids"] = list(range(maximum))
    generation["token_count"] = maximum
    encoded = json.dumps(
        generation["generated_token_ids"], separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")
    generation["generated_token_ids_sha256"] = hashlib.sha256(encoded).hexdigest()
    generation["cap_hit"] = False

    issues = evaluator._generation_issues("candidate", "{}", generation)

    assert any("cap_hit contradicts" in issue for issue in issues)
    assert any("reached its token cap" in issue for issue in issues)


def test_bundle_validation_binds_prompt_schema_raw_hash_and_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        evaluator,
        "PROTOCOL_LOCK_PATH",
        ROOT / "docs" / "protocol_v5.md",
    )
    row = {
        "dataset_version": "pilot-v5",
        "split": "development",
        "semantic_case_id": "MEL-190",
        "document_id": "MEL-190-A",
        "condition": "clean",
        "image_path": "output/v5/rendered/clean/MEL-190-A.png",
        "image_sha256": "a" * 64,
    }
    rendered_prompt = "fixed prompt"
    event_path = (
        "results/v5/development/call_events/candidate/clean/MEL-190-A/001.response_persisted.json"
    )
    raw_text = "{}"
    bundle = {
        "protocol_version": evaluator.PROTOCOL_VERSION,
        "protocol_lock_path": evaluator._relative(evaluator.PROTOCOL_LOCK_PATH),
        "protocol_lock_sha256": evaluator.sha256_file(evaluator.PROTOCOL_LOCK_PATH),
        "protocol_source_commit": "b" * 40,
        "dataset_version": "pilot-v5",
        "split": "development",
        "semantic_case_id": "MEL-190",
        "document_id": "MEL-190-A",
        "condition": "clean",
        "image_path": row["image_path"],
        "image_sha256": row["image_sha256"],
        "pass": "candidate",
        "model_id": evaluator.MODEL_ID,
        "requested_revision": evaluator.MODEL_REVISION,
        "resolved_revision": evaluator.MODEL_REVISION,
        "repository_commit": "c" * 40,
        "execution_attempt_id": "qualification-attempt",
        "call_ordinal": 1,
        "prompt_template_path": evaluator._relative(evaluator.CANDIDATE_PROMPT_PATH),
        "prompt_template_sha256": evaluator.sha256_file(evaluator.CANDIDATE_PROMPT_PATH),
        "rendered_prompt": evaluator._expected_text_descriptor(rendered_prompt),
        "response_schema_path": evaluator._relative(evaluator.CANDIDATE_SCHEMA_PATH),
        "response_schema_sha256": evaluator.sha256_file(evaluator.CANDIDATE_SCHEMA_PATH),
        "raw_text": raw_text,
        "raw_text_descriptor": evaluator._expected_text_descriptor(raw_text),
        "generation": _valid_generation("candidate", raw_text),
        "response_persisted_event": {
            "path": event_path,
            "bytes": 1,
            "sha256": "d" * 64,
        },
    }
    monkeypatch.setattr(evaluator, "_artifact_descriptor_errors", lambda *_args: [])

    assert (
        evaluator._bundle_issues(
            pass_name="candidate",
            bundle=bundle,
            row=row,
            protocol_lock={"protocol_source_commit": "b" * 40},
            rendered_prompt=rendered_prompt,
            schema_path=evaluator.CANDIDATE_SCHEMA_PATH,
        )
        == []
    )

    bundle["rendered_prompt"]["sha256"] = "0" * 64
    bundle["raw_text_descriptor"]["bytes"] = 99
    issues = evaluator._bundle_issues(
        pass_name="candidate",
        bundle=bundle,
        row=row,
        protocol_lock={"protocol_source_commit": "b" * 40},
        rendered_prompt=rendered_prompt,
        schema_path=evaluator.CANDIDATE_SCHEMA_PATH,
    )
    assert any("rendered_prompt" in issue for issue in issues)
    assert any("raw_text_descriptor" in issue for issue in issues)


def test_incomplete_split_never_reads_truth_or_writes_seal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = _manifest_rows("development", 4)
    writes: list[dict[str, Any]] = []
    truth_reads: list[Path] = []
    monkeypatch.setattr(evaluator, "verify_protocol_lock", lambda: {})
    monkeypatch.setattr(evaluator, "_load_manifest", lambda _split: manifest)
    monkeypatch.setattr(evaluator, "_verify_operational_lock", lambda *_args: None)

    def preflight(row: dict[str, str], _lock: dict[str, Any]) -> dict[str, Any]:
        index = manifest.index(row)
        return {
            "row": row,
            "record": {"repository_commit": "c" * 40},
            "issues": ["missing response bundle"] if index == 3 else [],
        }

    monkeypatch.setattr(evaluator, "_preflight_row", preflight)
    monkeypatch.setattr(
        evaluator,
        "_validate_event_ledgers",
        lambda *_args: ({"events": []}, []),
    )
    monkeypatch.setattr(
        evaluator,
        "_write_output_seal",
        lambda *_args: pytest.fail("seal must not be written for incomplete provenance"),
    )
    monkeypatch.setattr(
        evaluator,
        "_verify_evaluation_only_corpus",
        lambda *_args: pytest.fail("truth-side bytes must remain unopened"),
    )

    original_read = evaluator.read_json

    def guarded_read(path: Path) -> Any:
        if "ground_truth" in Path(path).parts:
            truth_reads.append(Path(path))
            raise AssertionError("ground truth read before seal")
        return original_read(path)

    monkeypatch.setattr(evaluator, "read_json", guarded_read)
    monkeypatch.setattr(evaluator, "write_json", lambda _path, payload: writes.append(payload))

    payload = evaluator.evaluate_split("development")

    assert payload["status"] == "DEVELOPMENT_INCOMPLETE"
    assert payload["ground_truth_read"] is False
    assert payload["field_exact_total_denominator"] == 64
    assert payload["provenance_valid_count"] == 3
    assert payload["pass"] is False
    assert truth_reads == []
    assert writes == [payload]


def test_output_seal_precedes_first_ground_truth_deserialization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = _manifest_rows("development", 4)
    event_order: list[str] = []
    seal_path = tmp_path / "results" / "v5" / "development" / "output_seal.json"
    metrics_path = tmp_path / "results" / "v5" / "development" / "metrics.json"
    monkeypatch.setattr(evaluator, "ROOT", tmp_path)
    monkeypatch.setitem(evaluator.OUTPUT_SEAL_PATHS, "development", seal_path)
    monkeypatch.setitem(evaluator.METRICS_PATHS, "development", metrics_path)
    monkeypatch.setattr(evaluator, "verify_protocol_lock", lambda: {})
    monkeypatch.setattr(evaluator, "_load_manifest", lambda _split: manifest)
    monkeypatch.setattr(evaluator, "_verify_operational_lock", lambda *_args: None)
    preflight = [
        {
            "row": row,
            "record": {"repository_commit": "c" * 40},
            "candidate": {},
            "normalized": {},
            "compiler_log": {"fields": {}, "history_evidence_rejection_count": 0},
            "issues": [],
        }
        for row in manifest
    ]
    monkeypatch.setattr(
        evaluator, "_preflight_row", lambda row, _lock: preflight[manifest.index(row)]
    )
    monkeypatch.setattr(evaluator, "_validate_event_ledgers", lambda *_args: ({"events": []}, []))

    def write_seal(*_args: Any) -> dict[str, Any]:
        event_order.append("seal")
        payload = {
            "repository_commit": "c" * 40,
            "output_count": 4,
        }
        seal_path.parent.mkdir(parents=True)
        seal_path.write_text(json.dumps(payload), encoding="utf-8")
        return payload

    monkeypatch.setattr(evaluator, "_write_output_seal", write_seal)
    monkeypatch.setattr(
        evaluator,
        "_verify_evaluation_only_corpus",
        lambda *_args: event_order.append("truth_hash_verification"),
    )
    original_read = evaluator.read_json

    def guarded_read(path: Path) -> Any:
        if "ground_truth" in Path(path).parts:
            event_order.append("truth_deserialize")
            return {}
        return original_read(path)

    monkeypatch.setattr(evaluator, "read_json", guarded_read)
    exact_overall = {
        "output_count": 4,
        "schema_valid_count": 4,
        "schema_valid_rate": 1.0,
        "field_exact_correct": 64,
        "field_exact_total": 64,
        "field_exact_match": 1.0,
        "nonnull_recalled": 52,
        "nonnull_total": 52,
        "unsupported_count": 0,
        "unsupported_opportunities": 12,
    }
    scored_rows = [
        {
            "document_id": row["document_id"],
            "semantic_case_id": row["semantic_case_id"],
            "condition": row["condition"],
            "template_id": row["template_id"],
            "_prediction": {},
        }
        for row in manifest
    ]
    monkeypatch.setattr(
        evaluator,
        "_score_layer",
        lambda *_args: (deepcopy(exact_overall), deepcopy(scored_rows)),
    )
    monkeypatch.setattr(
        evaluator,
        "_group_metrics",
        lambda _rows, key: (
            [
                {"condition": "clean", "field_exact_correct": 32},
                {"condition": "ocr_degraded", "field_exact_correct": 32},
            ]
            if key == "condition"
            else []
        ),
    )
    monkeypatch.setattr(evaluator, "_per_field_metrics", lambda _rows: {})
    monkeypatch.setattr(
        evaluator,
        "_evidence_summary",
        lambda _items: {
            "accepted_with_valid_current_report_evidence_rate": 1.0,
            "history_carryover_count": 0,
        },
    )
    monkeypatch.setattr(evaluator, "_history_sentinel_carryover", lambda _rows: 0)
    monkeypatch.setattr(
        evaluator,
        "_telemetry_summary",
        lambda _items: {
            "candidate": {"cap_hit_count": 0},
            "audit": {"cap_hit_count": 0},
        },
    )
    monkeypatch.setattr(
        evaluator,
        "write_json",
        lambda path, payload: path.write_text(json.dumps(payload), encoding="utf-8"),
    )

    payload = evaluator.evaluate_split("development")

    assert event_order[:3] == ["seal", "truth_hash_verification", "truth_deserialize"]
    assert event_order.count("truth_deserialize") == 4
    assert payload["status"] == "DEVELOPMENT_PASS"
    assert payload["pass"] is True


def test_scoring_uses_all_sixteen_management_fields() -> None:
    manifest: list[dict[str, str]] = []
    preflight: list[dict[str, Any]] = []
    truth_by_document: dict[str, dict[str, Any]] = {}
    with evaluator.V5_MANIFEST.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["split"] != "development":
                continue
            truth = evaluator.read_json(ROOT / row["ground_truth_path"])
            manifest.append(row)
            truth_by_document[row["document_id"]] = truth
            preflight.append({"row": row, "normalized": deepcopy(truth)})

    overall, rows = evaluator._score_layer(preflight, truth_by_document, "normalized")

    assert len(rows) == 4
    assert overall["field_exact_total"] == 64
    assert overall["field_exact_correct"] == 64
    assert overall["nonnull_total"] == 52
    assert overall["unsupported_opportunities"] == 12


def test_formal_requires_hash_bound_development_pass(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metrics_path = tmp_path / "development" / "metrics.json"
    seal_path = tmp_path / "development" / "output_seal.json"
    lock_path = tmp_path / "protocol_lock.json"
    manifest_path = tmp_path / "report_manifest.csv"
    metrics_path.parent.mkdir(parents=True)
    lock_path.write_text("{}\n", encoding="utf-8")
    manifest_path.write_text("manifest\n", encoding="utf-8")
    seal = {
        "protocol_version": evaluator.PROTOCOL_VERSION,
        "split": "development",
        "output_count": 4,
        "protocol_lock_sha256": evaluator.sha256_file(lock_path),
        "manifest_sha256": evaluator.sha256_file(manifest_path),
    }
    seal_path.write_text(json.dumps(seal), encoding="utf-8")
    metrics = {
        "status": "DEVELOPMENT_FAIL",
        "protocol_version": evaluator.PROTOCOL_VERSION,
        "split": "development",
    }
    metrics_path.write_text(json.dumps(metrics), encoding="utf-8")
    monkeypatch.setitem(evaluator.METRICS_PATHS, "development", metrics_path)
    monkeypatch.setitem(evaluator.OUTPUT_SEAL_PATHS, "development", seal_path)
    monkeypatch.setattr(evaluator, "PROTOCOL_LOCK_PATH", lock_path)
    monkeypatch.setattr(evaluator, "V5_MANIFEST", manifest_path)

    with pytest.raises(ValueError, match="Development gate status"):
        evaluator._verify_development_gate_for_formal()


def test_blinded_checklist_contains_no_metrics_and_requires_human_attestation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path
    checklist = root / "results/v5/formal/manual_review_blinded.csv"
    attestation_path = root / "results/v5/formal/human_review_attestation.json"
    seal_path = root / "results/v5/formal/output_seal.json"
    seal_path.parent.mkdir(parents=True)
    seal_path.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(evaluator, "ROOT", root)
    monkeypatch.setattr(evaluator, "MANUAL_REVIEW_PATH", checklist)
    monkeypatch.setattr(evaluator, "HUMAN_ATTESTATION_PATH", attestation_path)
    monkeypatch.setitem(evaluator.OUTPUT_SEAL_PATHS, "formal", seal_path)
    manifest = _manifest_rows("formal", 20)

    def fake_paths(row: dict[str, str]) -> dict[str, Path]:
        base = root / "results/v5/formal"
        name = f"{row['document_id']}-{row['condition']}"
        return {
            "ocr": base / "ocr" / f"{name}.json",
            "candidate_bundle": base / "candidate" / f"{name}.json",
            "audit_bundle": base / "audit" / f"{name}.json",
            "normalized": base / "normalized" / f"{name}.json",
            "compiler_audit": base / "compiler" / f"{name}.json",
        }

    monkeypatch.setattr(evaluator, "output_paths", fake_paths)
    evaluator._ensure_blinded_review_checklist(manifest)
    checklist_text = checklist.read_text(encoding="utf-8").lower()
    for forbidden in ("exact_match", "score", "metric", "pass", "fail"):
        assert forbidden not in checklist_text

    pending = evaluator._human_review_status(manifest, {})
    assert pending["complete"] is False
    assert pending["attestation_valid"] is False

    with checklist.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        for key in (
            "reviewed",
            "artifact_bundle_complete",
            "evidence_checked",
            "prediction_not_corrected",
        ):
            row[key] = "yes"
    with checklist.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    attestation_path.write_text(
        json.dumps(
            {
                "protocol_version": evaluator.PROTOCOL_VERSION,
                "split": "formal",
                "reviewed_output_seal_sha256": evaluator.sha256_file(seal_path),
                "reviewed_row_count": 20,
                "checklist_path": evaluator._relative(checklist),
                "checklist_sha256": evaluator.sha256_file(checklist),
                "attestation": evaluator.HUMAN_ATTESTATION_TEXT,
                "reviewer_name": "Fixture Human",
                "review_completed_at_utc": "2026-08-06T12:00:00+00:00",
            }
        ),
        encoding="utf-8",
    )

    complete = evaluator._human_review_status(manifest, {})

    assert complete["checklist_complete"] is True
    assert complete["attestation_valid"] is True
    assert complete["complete"] is True
