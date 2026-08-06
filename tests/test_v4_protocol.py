from __future__ import annotations

import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import evaluate_v4  # noqa: E402
import freeze_v4_protocol as freezer  # noqa: E402
import run_v4_inference as inference  # noqa: E402


def _serializer() -> dict[str, Any]:
    return {
        "configuration_stage": "after_transformers_prefix_construction",
        "tokenizer_alphabet_preserved": True,
        "force_json_field_order": False,
        "max_consecutive_whitespaces": 12,
        "max_json_array_length": 20,
    }


def _generation(raw_text: str) -> inference.GenerationResult:
    token_ids = (10, 11, 12)
    encoded = json.dumps(list(token_ids), separators=(",", ":")).encode("ascii")
    return inference.GenerationResult(
        raw_text=raw_text,
        elapsed_seconds=0.25,
        prompt_token_count=100,
        generated_token_ids=token_ids,
        generated_token_ids_sha256=inference.sha256_bytes(encoded),
        eos_token_ids=(1, 2),
        token_count=len(token_ids),
        eos_observed=False,
        cap_hit=False,
        max_new_tokens=inference.CANDIDATE_MAX_NEW_TOKENS,
        serializer=_serializer(),
    )


def test_effective_lmfe_config_is_mutated_after_builder_reset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    class Parser:
        def __init__(self, schema: dict[str, Any]) -> None:
            self.schema = schema
            self.config = SimpleNamespace(
                alphabet="pre-builder",
                force_json_field_order=True,
                max_consecutive_whitespaces=99,
                max_json_array_length=20,
            )
            captured["parser"] = self

    def builder(tokenizer_data: Any, parser: Any) -> object:
        # This reproduces LMFE 0.11.3's tokenizer-adaptation reset.
        parser.config = SimpleNamespace(
            alphabet=tokenizer_data.tokenizer_alphabet,
            force_json_field_order=False,
            max_consecutive_whitespaces=12,
            max_json_array_length=20,
        )
        return object()

    package = ModuleType("lmformatenforcer")
    package.JsonSchemaParser = Parser  # type: ignore[attr-defined]
    integrations = ModuleType("lmformatenforcer.integrations")
    transformers = ModuleType("lmformatenforcer.integrations.transformers")
    transformers.build_transformers_prefix_allowed_tokens_fn = builder  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "lmformatenforcer", package)
    monkeypatch.setitem(sys.modules, "lmformatenforcer.integrations", integrations)
    monkeypatch.setitem(
        sys.modules,
        "lmformatenforcer.integrations.transformers",
        transformers,
    )

    tokenizer_data = SimpleNamespace(tokenizer_alphabet="tokenizer alphabet")
    _, settings = inference.build_effective_prefix_allowed_tokens_fn(
        tokenizer_data,
        {"type": "object"},
        force_json_field_order=True,
        max_consecutive_whitespaces=0,
    )

    assert captured["parser"].config.alphabet == "tokenizer alphabet"
    assert captured["parser"].config.force_json_field_order is True
    assert captured["parser"].config.max_consecutive_whitespaces == 0
    assert settings["tokenizer_alphabet_preserved"] is True
    assert settings["configuration_stage"] == "after_transformers_prefix_construction"


def test_raw_generation_and_telemetry_are_written_before_json_parse(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(inference, "ROOT", tmp_path)
    monkeypatch.setattr(inference, "validate_manifest_image", lambda _row: tmp_path / "x.png")
    monkeypatch.setattr(
        inference,
        "run_ocr",
        lambda _path: (
            [{"line_id": "L001", "text": "Document ID: MEL-104-A"}],
            {"text": ["Document ID: MEL-104-A"]},
            {"elapsed_seconds": 0.1},
        ),
    )
    monkeypatch.setattr(inference, "render_candidate_prompt", lambda _lines: "candidate")
    monkeypatch.setattr(inference, "render_audit_prompt", lambda _lines: "audit")
    monkeypatch.setattr(
        inference,
        "constrained_generate",
        lambda *_args, **_kwargs: _generation('{"value":'),
    )
    row = {
        "split": "development",
        "condition": "clean",
        "document_id": "MEL-104-A",
    }
    progress: dict[str, Any] = {}
    with pytest.raises(inference.StrictGeneratedJSONError):
        inference.run_row(
            row,
            SimpleNamespace(),
            "a" * 40,
            {
                "type": "object",
                "additionalProperties": False,
                "required": ["value"],
                "properties": {"value": {"type": "integer"}},
            },
            {},
            {},
            1,
            None,
            progress=progress,
        )
    paths = inference.output_paths(row, development_iteration=1)
    assert paths["candidate_raw"].read_text(encoding="utf-8") == '{"value":'
    metadata = json.loads(paths["candidate_generation"].read_text(encoding="utf-8"))
    assert metadata["generated_token_ids"] == [10, 11, 12]
    assert metadata["prompt_token_count"] == 100
    assert not paths["candidate_parsed"].exists()
    assert not paths["record"].exists()
    assert progress["model_call_count"] == 1
    assert progress["resume_permitted"] is False


def test_one_shot_guards_reject_existing_namespaces(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(inference, "ROOT", tmp_path)
    formal_file = tmp_path / "results" / "v4" / "formal" / "execution_attempts" / "a.started.json"
    formal_file.parent.mkdir(parents=True)
    formal_file.write_text("{}\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="one-shot"):
        inference.verify_formal_one_shot_safety([], "a" * 64)

    development_file = tmp_path / "results" / "v4" / "development" / "iteration-1" / "partial.txt"
    development_file.parent.mkdir(parents=True)
    development_file.write_text("partial\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="not pristine"):
        inference.verify_development_one_shot_safety()


def _development_pipeline(exact: float = 0.0) -> dict[str, Any]:
    overall = {
        "output_count": 4,
        "schema_valid_count": 4,
        "schema_valid_rate": 1.0,
        "field_exact_match": exact,
        "unsupported_count": 0,
        "unsupported_field_rate": 0.0,
    }
    return {
        "overall": overall,
        "by_condition": [
            {
                **overall,
                "condition": condition,
                "output_count": 2,
                "schema_valid_count": 2,
            }
            for condition in inference.CONDITIONS
        ],
    }


def test_development_gate_does_not_select_on_accuracy() -> None:
    gate = freezer.require_development_gate(
        _development_pipeline(exact=0.0),
        {
            "accepted_non_null_evidence_rate": 1.0,
            "history_carryover_count": 0,
        },
        {
            "candidate": {"cap_hit_count": 0},
            "audit": {"cap_hit_count": 0},
        },
    )
    assert gate["pooled_field_exact_match"] == 0.0
    assert all(gate["criteria"].values())


def test_development_gate_fails_on_cap_or_unsupported_value() -> None:
    with pytest.raises(freezer.ProtocolFreezeError, match="zero_candidate_cap_hits"):
        freezer.require_development_gate(
            _development_pipeline(),
            {
                "accepted_non_null_evidence_rate": 1.0,
                "history_carryover_count": 0,
            },
            {
                "candidate": {"cap_hit_count": 1},
                "audit": {"cap_hit_count": 0},
            },
        )
    pipeline = _development_pipeline()
    pipeline["overall"]["unsupported_count"] = 1
    with pytest.raises(freezer.ProtocolFreezeError, match="zero_unsupported_fields"):
        freezer.require_development_gate(
            pipeline,
            {
                "accepted_non_null_evidence_rate": 1.0,
                "history_carryover_count": 0,
            },
            {
                "candidate": {"cap_hit_count": 0},
                "audit": {"cap_hit_count": 0},
            },
        )


def test_only_development_iteration_one_is_permitted(tmp_path: Path) -> None:
    iteration_two = tmp_path / "results" / "v4" / "development" / "iteration-2"
    iteration_two.mkdir(parents=True)
    with pytest.raises(freezer.ProtocolFreezeError, match="forbidden additional iteration"):
        freezer.collect_documented_development_iterations(tmp_path, 1)
    with pytest.raises(freezer.ProtocolFreezeError, match="exactly one"):
        freezer.collect_documented_development_iterations(tmp_path, 2)


def test_generation_telemetry_detects_token_and_serializer_tampering() -> None:
    result = _generation("{}")
    artifact = inference.generation_metadata(result, pass_name="candidate")
    assert (
        evaluate_v4._generation_telemetry_issues(
            pass_name="candidate",
            raw_text="{}",
            artifact=artifact,
            record_value=artifact,
        )
        == []
    )
    tampered = json.loads(json.dumps(artifact))
    tampered["generated_token_ids"][0] += 1
    tampered["serializer"]["force_json_field_order"] = True
    issues = evaluate_v4._generation_telemetry_issues(
        pass_name="candidate",
        raw_text="{}",
        artifact=tampered,
        record_value=tampered,
    )
    assert any("token-ID digest" in issue for issue in issues)
    assert any("force_json_field_order" in issue for issue in issues)


def test_v3_reuse_boundary_is_machine_verified() -> None:
    rows = freezer.load_and_validate_manifest(ROOT)
    provenance, bound_paths = freezer.collect_v3_reuse_provenance(ROOT, rows)
    assert provenance["v3_formal_model_artifact_case_ids"] == ["MEL-104"]
    assert provenance["v3_unexecuted_reused_formal_case_ids"] == [
        "MEL-105",
        "MEL-106",
        "MEL-107",
        "MEL-108",
    ]
    assert "data/v3/protocol_lock.json" in bound_paths
