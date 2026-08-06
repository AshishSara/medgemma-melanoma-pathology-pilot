from __future__ import annotations

import argparse
import json
import sys
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import run_v4_inference as inference  # noqa: E402


class _FakeTokenVector:
    def __init__(self, values: list[int]) -> None:
        self.values = values

    def __getitem__(self, key: slice) -> "_FakeTokenVector":
        return _FakeTokenVector(self.values[key])

    def detach(self) -> "_FakeTokenVector":
        return self

    def cpu(self) -> "_FakeTokenVector":
        return self

    def tolist(self) -> list[int]:
        return list(self.values)


class _FakeInputs(dict[str, Any]):
    def to(self, *_args: Any, **_kwargs: Any) -> "_FakeInputs":
        return self


@pytest.mark.parametrize(
    ("generated_ids", "max_new_tokens", "expected_eos", "expected_cap_hit"),
    [
        ([8, 9, 99], 3, True, True),
        ([99, 8], 3, False, False),
    ],
)
def test_constrained_generation_records_mechanical_termination_telemetry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    generated_ids: list[int],
    max_new_tokens: int,
    expected_eos: bool,
    expected_cap_hit: bool,
) -> None:
    image_path = tmp_path / "input.png"
    Image.new("L", (2, 2), color=255).save(image_path)
    prompt_ids = [101, 102, 103, 104]
    generate_call: dict[str, Any] = {}

    class Processor:
        tokenizer = SimpleNamespace(eos_token_id=99)

        def apply_chat_template(self, *_args: Any, **_kwargs: Any) -> _FakeInputs:
            return _FakeInputs(
                input_ids=SimpleNamespace(shape=(1, len(prompt_ids))),
            )

        def decode(
            self,
            generated: _FakeTokenVector,
            *,
            skip_special_tokens: bool,
        ) -> str:
            assert generated.tolist() == generated_ids
            assert skip_special_tokens is True
            return '{"ok":true}'

    class Model:
        device = "cpu"
        generation_config = SimpleNamespace(eos_token_id=[1, 99])

        def generate(self, **kwargs: Any) -> list[_FakeTokenVector]:
            generate_call.update(kwargs)
            return [_FakeTokenVector(prompt_ids + generated_ids)]

    serializer = {
        "configuration_stage": "after_transformers_prefix_construction",
        "tokenizer_alphabet_preserved": True,
        "force_json_field_order": True,
        "max_consecutive_whitespaces": 0,
        "max_json_array_length": 20,
    }
    monkeypatch.setattr(
        inference,
        "build_effective_prefix_allowed_tokens_fn",
        lambda *_args, **_kwargs: ("prefix-callback", serializer),
    )
    backend = inference.Backend(
        model=Model(),
        processor=Processor(),
        tokenizer_data=object(),
        torch=SimpleNamespace(inference_mode=nullcontext),
        dtype="bf16",
        metadata={},
    )

    result = inference.constrained_generate(
        backend,
        image_path,
        "prompt",
        {"type": "object"},
        max_new_tokens,
        force_json_field_order=True,
        max_consecutive_whitespaces=0,
    )

    expected_digest = inference.sha256_bytes(
        json.dumps(generated_ids, separators=(",", ":")).encode("ascii")
    )
    assert result.prompt_token_count == len(prompt_ids)
    assert result.generated_token_ids == tuple(generated_ids)
    assert result.generated_token_ids_sha256 == expected_digest
    assert result.eos_token_ids == (1, 99)
    assert result.eos_observed is expected_eos
    assert result.cap_hit is expected_cap_hit
    assert result.token_count == len(generated_ids)
    assert generate_call["max_new_tokens"] == max_new_tokens
    assert generate_call["prefix_allowed_tokens_fn"] == "prefix-callback"
    assert generate_call["do_sample"] is False
    assert generate_call["num_beams"] == 1


def _result(
    raw_text: str,
    max_new_tokens: int,
    *,
    force_json_field_order: bool,
    max_consecutive_whitespaces: int,
) -> inference.GenerationResult:
    token_ids = (20, 21)
    digest = inference.sha256_bytes(
        json.dumps(list(token_ids), separators=(",", ":")).encode("ascii")
    )
    return inference.GenerationResult(
        raw_text=raw_text,
        elapsed_seconds=0.1,
        prompt_token_count=25,
        generated_token_ids=token_ids,
        generated_token_ids_sha256=digest,
        eos_token_ids=(1,),
        token_count=len(token_ids),
        eos_observed=False,
        cap_hit=False,
        max_new_tokens=max_new_tokens,
        serializer={
            "configuration_stage": "after_transformers_prefix_construction",
            "tokenizer_alphabet_preserved": True,
            "force_json_field_order": force_json_field_order,
            "max_consecutive_whitespaces": max_consecutive_whitespaces,
            "max_json_array_length": 20,
        },
    )


def test_each_pass_uses_frozen_settings_and_audit_bytes_precede_parse_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(inference, "ROOT", tmp_path)
    monkeypatch.setattr(
        inference,
        "validate_manifest_image",
        lambda _row: tmp_path / "unused.png",
    )
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
    calls: list[dict[str, Any]] = []

    def generate(
        _backend: Any,
        _image_path: Path,
        prompt: str,
        _schema: dict[str, Any],
        max_new_tokens: int,
        *,
        force_json_field_order: bool,
        max_consecutive_whitespaces: int,
    ) -> inference.GenerationResult:
        calls.append(
            {
                "prompt": prompt,
                "max_new_tokens": max_new_tokens,
                "force_json_field_order": force_json_field_order,
                "max_consecutive_whitespaces": max_consecutive_whitespaces,
            }
        )
        raw_text = '{"value":1}' if prompt == "candidate" else '{"value":'
        return _result(
            raw_text,
            max_new_tokens,
            force_json_field_order=force_json_field_order,
            max_consecutive_whitespaces=max_consecutive_whitespaces,
        )

    monkeypatch.setattr(inference, "constrained_generate", generate)
    schema = {
        "type": "object",
        "additionalProperties": False,
        "required": ["value"],
        "properties": {"value": {"type": "integer"}},
    }
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
            schema,
            schema,
            {},
            1,
            None,
            progress=progress,
        )

    assert calls == [
        {
            "prompt": "candidate",
            "max_new_tokens": inference.CANDIDATE_MAX_NEW_TOKENS,
            "force_json_field_order": False,
            "max_consecutive_whitespaces": 12,
        },
        {
            "prompt": "audit",
            "max_new_tokens": inference.AUDIT_MAX_NEW_TOKENS,
            "force_json_field_order": True,
            "max_consecutive_whitespaces": 0,
        },
    ]
    paths = inference.output_paths(row, development_iteration=1)
    assert paths["candidate_parsed"].is_file()
    assert paths["audit_raw"].read_text(encoding="utf-8") == '{"value":'
    audit_telemetry = json.loads(paths["audit_generation"].read_text(encoding="utf-8"))
    assert audit_telemetry["max_new_tokens"] == inference.AUDIT_MAX_NEW_TOKENS
    assert audit_telemetry["generated_token_ids"] == [20, 21]
    assert not paths["audit_parsed"].exists()
    assert progress == {
        "stage": "audit_validation",
        "model_call_count": 2,
        "valid_model_response_count": 1,
        "resume_permitted": False,
    }


def test_development_run_rejects_anything_other_than_all_four_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        inference,
        "load_manifest",
        lambda _split, _condition: [
            {
                "split": "development",
                "condition": "clean",
                "document_id": f"MEL-104-{index}",
            }
            for index in range(3)
        ],
    )
    args = argparse.Namespace(
        split="development",
        condition="all",
        limit=None,
        development_iteration=1,
        overwrite=False,
    )

    with pytest.raises(SystemExit, match="exactly 4 manifest rows"):
        inference.run(args)


def test_competing_formal_attempt_cannot_claim_the_fixed_budget_twice(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(inference, "ROOT", tmp_path)
    payload = {
        "execution_attempt_id": inference.FORMAL_EXECUTION_ATTEMPT_ID,
        "status": "started",
    }

    claimed = inference.claim_formal_attempt(payload)

    assert claimed.name == (f"{inference.FORMAL_EXECUTION_ATTEMPT_ID}.started.json")
    assert json.loads(claimed.read_text(encoding="utf-8")) == payload
    with pytest.raises(FileExistsError):
        inference.claim_formal_attempt(payload)
