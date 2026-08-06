from __future__ import annotations

import copy
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import freeze_v5_protocol as freezer  # noqa: E402


def _git(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=repo, text=True).strip()


def test_protocol_declares_same_lock_for_qualification_and_formal_gates() -> None:
    protocol = (ROOT / "docs" / "protocol_v5.md").read_text(encoding="utf-8")
    assert "These qualification criteria are part of the same protocol source inventory" in protocol
    assert "at least 58/64 fields are exact overall" in protocol
    assert "at least 45/52 ground-truth non-null fields are recalled" in protocol
    assert "At least 288/320 management-critical fields are exactly correct" in protocol
    assert "At least 167/196 ground-truth non-null fields are recalled" in protocol
    assert "draft `MEL-202` vector was identical to v2" in protocol
    assert "specimen site was changed from `shoulder`" in protocol
    assert "The source commit must not contain the lock" in protocol
    assert "both that prospective source commit and the execution" in protocol


def test_v5_fixed_runtime_and_requirement_pins_match_protocol() -> None:
    assert freezer.validate_runtime_profile(ROOT) == freezer.EXPECTED_RUNTIME_PROFILE
    assert freezer.EXPECTED_RUNTIME_PROFILE["cuda_device_count"] == 1
    assert freezer.EXPECTED_RUNTIME_PROFILE["cuda_device_name"] == "Tesla T4"
    assert freezer.EXPECTED_RUNTIME_PROFILE["python_version"] == "3.12.13"
    assert freezer.EXPECTED_RUNTIME_PROFILE["torch_version"] == "2.11.0+cu128"
    assert freezer.EXPECTED_RUNTIME_PROFILE["packages"] == {
        "accelerate": "1.14.0",
        "lm-format-enforcer": "0.11.3",
        "pytesseract": "0.3.13",
        "transformers": "4.57.6",
    }


def test_v5_thresholds_are_exact_integer_gates_not_recomputed_rates() -> None:
    assert freezer.GATE_THRESHOLDS == {
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


def test_v5_locked_source_inventory_includes_new_and_reused_dependencies() -> None:
    required = set(freezer.REQUIRED_LOCKED_SOURCE_PATHS)
    assert {
        "docs/protocol_v5.md",
        "requirements/colab-v5.txt",
        "data/v5/runtime_profile.json",
        "scripts/freeze_v5_protocol.py",
        "scripts/generate_reports.py",
        "scripts/generate_v5_reports.py",
        "scripts/run_v5_inference.py",
        "scripts/evaluate_v5.py",
        "scripts/pilot_utils.py",
        "scripts/v3_pipeline.py",
        "prompts/v3/candidate_prompt.txt",
        "prompts/v3/audit_prompt.txt",
        "schema/v3/candidate.schema.json",
        "schema/v3/audit.schema.json",
        "schema/extraction.schema.json",
        "tests/test_v5_corpus.py",
        "tests/test_v5_protocol.py",
        "tests/test_v5_inference.py",
        "tests/test_v5_evaluator.py",
    } <= required
    assert len(required) == len(freezer.REQUIRED_LOCKED_SOURCE_PATHS)


def test_runner_evaluator_and_freezer_configuration_handshake() -> None:
    freezer.validate_implementation_configuration()


def test_lock_payload_preserves_exact_hash_maps_inventory_and_scope() -> None:
    cases, freshness, prior_hashes = freezer.load_and_validate_cases(ROOT)
    rows = freezer.load_and_validate_manifest(ROOT, cases)
    operational_paths, evaluation_paths = freezer.expected_corpus_paths(rows)
    operational = {path: "a" * 64 for path in sorted(operational_paths)}
    evaluation = {path: "b" * 64 for path in sorted(evaluation_paths)}
    source = {path: "c" * 64 for path in freezer.REQUIRED_LOCKED_SOURCE_PATHS}
    denominators = freezer._validate_ground_truth_and_denominators(ROOT, rows, cases)

    payload = freezer.build_lock_payload(
        protocol_source_commit="d" * 40,
        runtime_profile=freezer.EXPECTED_RUNTIME_PROFILE,
        locked_source_files=source,
        corpus_artifacts=operational,
        evaluation_only_corpus_artifacts=evaluation,
        freshness_reference_artifacts=prior_hashes,
        semantic_freshness=freshness,
        denominators=denominators,
    )
    assert payload["status"] == "frozen_pre_inference"
    assert payload["protocol_source_commit"] == "d" * 40
    assert payload["gate_thresholds"] == freezer.GATE_THRESHOLDS
    assert payload["fixed_configuration"] == freezer.FIXED_CONFIGURATION
    assert payload["corpus_inventory"]["operational_count"] == 49
    assert payload["corpus_inventory"]["evaluation_only_count"] == 25
    assert payload["corpus_inventory"]["total_count"] == 74
    assert payload["freshness_reference_artifacts"] == prior_hashes


def test_runtime_and_threshold_drift_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    changed_runtime = copy.deepcopy(freezer.EXPECTED_RUNTIME_PROFILE)
    changed_runtime["cuda_device_count"] = 2
    runtime_path = ROOT / freezer.RUNTIME_PROFILE_RELATIVE_PATH
    original_reader = freezer._read_json

    def fake_reader(path: Path, label: str) -> dict[str, object]:
        if path == runtime_path:
            return changed_runtime
        return original_reader(path, label)

    monkeypatch.setattr(freezer, "_read_json", fake_reader)
    with pytest.raises(freezer.ProtocolFreezeError, match="runtime profile"):
        freezer.validate_runtime_profile(ROOT)


def _matching_runner() -> SimpleNamespace:
    values = {
        "PROTOCOL_VERSION": freezer.PROTOCOL_VERSION,
        "PROTOCOL_IDENTIFIER": freezer.PROTOCOL_IDENTIFIER,
        "MODEL_ID": freezer.MODEL_ID,
        "MODEL_REVISION": freezer.MODEL_REVISION,
        "DTYPE_NAME": freezer.DTYPE_NAME,
        "RANDOM_SEED": freezer.RANDOM_SEED,
        "CANDIDATE_MAX_NEW_TOKENS": freezer.CANDIDATE_MAX_NEW_TOKENS,
        "AUDIT_MAX_NEW_TOKENS": freezer.AUDIT_MAX_NEW_TOKENS,
        "CANDIDATE_SERIALIZER_FORCE_JSON_FIELD_ORDER": (
            freezer.CANDIDATE_SERIALIZER_FORCE_JSON_FIELD_ORDER
        ),
        "CANDIDATE_SERIALIZER_MAX_CONSECUTIVE_WHITESPACES": (
            freezer.CANDIDATE_SERIALIZER_MAX_CONSECUTIVE_WHITESPACES
        ),
        "AUDIT_SERIALIZER_FORCE_JSON_FIELD_ORDER": (
            freezer.AUDIT_SERIALIZER_FORCE_JSON_FIELD_ORDER
        ),
        "AUDIT_SERIALIZER_MAX_CONSECUTIVE_WHITESPACES": (
            freezer.AUDIT_SERIALIZER_MAX_CONSECUTIVE_WHITESPACES
        ),
        "TESSERACT_LANGUAGE": freezer.TESSERACT_LANGUAGE,
        "TESSERACT_CONFIG": freezer.TESSERACT_CONFIG,
        "OCR_TIMEOUT_SECONDS": freezer.TESSERACT_TIMEOUT_SECONDS,
        "CONDITIONS": freezer.CONDITIONS,
        "REQUIRED_LOCKED_SOURCE_PATHS": freezer.REQUIRED_LOCKED_SOURCE_PATHS,
    }
    return SimpleNamespace(**values)


def test_implementation_configuration_rejects_runner_or_threshold_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _matching_runner()
    changed_thresholds = copy.deepcopy(freezer.GATE_THRESHOLDS)
    changed_thresholds["formal"]["pooled_exact_minimum"] = 287
    evaluator = SimpleNamespace(GATE_THRESHOLDS=changed_thresholds)
    modules = {"run_v5_inference": runner, "evaluate_v5": evaluator}
    monkeypatch.setattr(
        freezer.importlib,
        "import_module",
        lambda name: modules[name],
    )
    with pytest.raises(freezer.ProtocolFreezeError, match="thresholds differ"):
        freezer.validate_implementation_configuration()

    runner.CANDIDATE_MAX_NEW_TOKENS = 513
    evaluator.GATE_THRESHOLDS = freezer.GATE_THRESHOLDS
    with pytest.raises(freezer.ProtocolFreezeError, match="runner configuration"):
        freezer.validate_implementation_configuration()


def test_exact_namespace_rejects_extra_files_and_symlinks(tmp_path: Path) -> None:
    namespace = tmp_path / "data" / "v5"
    expected = namespace / "cases.csv"
    expected.parent.mkdir(parents=True)
    expected.write_text("case_id\n", encoding="utf-8")
    freezer.validate_exact_namespace(tmp_path, "data/v5", ["data/v5/cases.csv"])

    extra = namespace / "cases 2.csv"
    extra.write_text("icloud duplicate\n", encoding="utf-8")
    with pytest.raises(freezer.ProtocolFreezeError, match="inventory is not exact"):
        freezer.validate_exact_namespace(tmp_path, "data/v5", ["data/v5/cases.csv"])
    extra.unlink()

    target = tmp_path / "outside.txt"
    target.write_text("outside\n", encoding="utf-8")
    link = namespace / "alias.csv"
    link.symlink_to(target)
    with pytest.raises(freezer.ProtocolFreezeError, match="symlink"):
        freezer.validate_exact_namespace(tmp_path, "data/v5", ["data/v5/cases.csv"])


def test_commit_check_ignores_unrelated_untracked_icloud_duplicates(
    tmp_path: Path,
) -> None:
    locked = tmp_path / "locked.txt"
    locked.write_text("locked bytes\n", encoding="utf-8")
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "config", "user.name", "V5 Protocol Test")
    _git(tmp_path, "config", "user.email", "v5-protocol@example.test")
    _git(tmp_path, "add", "locked.txt")
    _git(tmp_path, "commit", "-qm", "locked source")
    commit = _git(tmp_path, "rev-parse", "HEAD")

    unrelated = tmp_path / "results" / "old" / "artifact 2.json"
    unrelated.parent.mkdir(parents=True)
    unrelated.write_text("{}\n", encoding="utf-8")
    freezer.validate_protocol_source_commit(tmp_path, commit, ["locked.txt"])


def test_commit_check_rejects_dirty_or_untracked_locked_paths(tmp_path: Path) -> None:
    locked = tmp_path / "locked.txt"
    locked.write_text("locked bytes\n", encoding="utf-8")
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "config", "user.name", "V5 Protocol Test")
    _git(tmp_path, "config", "user.email", "v5-protocol@example.test")
    _git(tmp_path, "add", "locked.txt")
    _git(tmp_path, "commit", "-qm", "locked source")
    commit = _git(tmp_path, "rev-parse", "HEAD")

    locked.write_text("mutated bytes\n", encoding="utf-8")
    with pytest.raises(freezer.ProtocolFreezeError, match="dirty or untracked"):
        freezer.validate_protocol_source_commit(tmp_path, commit, ["locked.txt"])

    locked.write_text("locked bytes\n", encoding="utf-8")
    untracked = tmp_path / "new-locked.txt"
    untracked.write_text("not committed\n", encoding="utf-8")
    with pytest.raises(freezer.ProtocolFreezeError, match="dirty or untracked"):
        freezer.validate_protocol_source_commit(
            tmp_path,
            commit,
            ["locked.txt", "new-locked.txt"],
        )


def test_lock_writer_creates_only_the_lock_and_never_overwrites(tmp_path: Path) -> None:
    data = tmp_path / "data" / "v5"
    data.mkdir(parents=True)
    before = {path.relative_to(tmp_path) for path in tmp_path.rglob("*")}
    payload = {"status": "frozen_pre_inference", "protocol_source_commit": "a" * 40}
    lock_path = freezer._write_protocol_lock(tmp_path, payload)
    after = {path.relative_to(tmp_path) for path in tmp_path.rglob("*")}
    assert after - before == {Path("data/v5/protocol_lock.json")}
    assert json.loads(lock_path.read_text(encoding="utf-8")) == payload
    with pytest.raises(freezer.ProtocolFreezeError, match="overwrite"):
        freezer._write_protocol_lock(tmp_path, payload)


@pytest.mark.parametrize("bad_commit", ["a" * 39, "A" * 40, "g" * 40])
def test_commit_check_requires_exact_lowercase_40_character_sha(
    tmp_path: Path,
    bad_commit: str,
) -> None:
    with pytest.raises(freezer.ProtocolFreezeError, match="40-character lowercase"):
        freezer.validate_protocol_source_commit(tmp_path, bad_commit, ["x"])
