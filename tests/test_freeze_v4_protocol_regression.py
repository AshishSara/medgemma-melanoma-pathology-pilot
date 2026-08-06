from __future__ import annotations

import copy
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import freeze_v4_protocol as freezer  # noqa: E402


def _passing_development_gate_inputs() -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
]:
    overall = {
        "output_count": 4,
        "schema_valid_count": 4,
        "schema_valid_rate": 1.0,
        "field_exact_match": 0.0,
        "unsupported_count": 0,
        "unsupported_field_rate": 0.0,
    }
    pipeline = {
        "overall": overall,
        "by_condition": [
            {
                **overall,
                "condition": condition,
                "output_count": 2,
                "schema_valid_count": 2,
            }
            for condition in freezer.CONDITIONS
        ],
    }
    evidence = {
        "accepted_non_null_evidence_rate": 1.0,
        "history_carryover_count": 0,
    }
    telemetry = {
        "candidate": {"cap_hit_count": 0},
        "audit": {"cap_hit_count": 0},
    }
    return pipeline, evidence, telemetry


@pytest.mark.parametrize(
    ("scope", "key", "value", "message"),
    [
        ("overall", "output_count", 3, "output count is not 4"),
        ("overall", "schema_valid_rate", 0.75, "canonical_schema_100_percent"),
        (
            "evidence",
            "accepted_non_null_evidence_rate",
            0.99,
            "evidence_acceptance_100_percent",
        ),
        ("evidence", "history_carryover_count", 1, "zero_history_carryover"),
        ("audit", "cap_hit_count", 1, "zero_audit_cap_hits"),
    ],
)
def test_development_freeze_gate_requires_all_completion_and_safety_criteria(
    scope: str,
    key: str,
    value: float | int,
    message: str,
) -> None:
    pipeline, evidence, telemetry = copy.deepcopy(_passing_development_gate_inputs())
    if scope == "overall":
        pipeline["overall"][key] = value
    elif scope == "evidence":
        evidence[key] = value
    else:
        telemetry[scope][key] = value

    with pytest.raises(freezer.ProtocolFreezeError, match=message):
        freezer.require_development_gate(pipeline, evidence, telemetry)


def test_development_freeze_gate_requires_both_generation_telemetry_passes() -> None:
    pipeline, evidence, telemetry = _passing_development_gate_inputs()
    del telemetry["audit"]

    with pytest.raises(
        freezer.ProtocolFreezeError,
        match="lacks candidate or audit pass",
    ):
        freezer.require_development_gate(pipeline, evidence, telemetry)


def _git(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=repo, text=True).strip()


def test_freezer_rejects_source_mutation_before_and_after_development(
    tmp_path: Path,
) -> None:
    locked = tmp_path / "locked-source.py"
    locked.write_text("FROZEN = True\n", encoding="utf-8")
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "config", "user.name", "V4 Protocol Test")
    _git(tmp_path, "config", "user.email", "v4-protocol@example.test")
    _git(tmp_path, "add", "locked-source.py")
    _git(tmp_path, "commit", "-qm", "development source")
    development_commit = _git(tmp_path, "rev-parse", "HEAD")
    freezer.validate_protocol_source_commit(
        tmp_path,
        development_commit,
        ["locked-source.py"],
    )
    hardening = freezer.validate_post_development_immutability(
        tmp_path,
        development_commit,
        ["locked-source.py"],
    )
    assert hardening["classification"] == ("byte_identical_sources_corpus_and_v3_provenance")

    locked.write_text("FROZEN = False\n", encoding="utf-8")
    with pytest.raises(
        freezer.ProtocolFreezeError,
        match="must be committed before freezing",
    ):
        freezer.validate_protocol_source_commit(
            tmp_path,
            development_commit,
            ["locked-source.py"],
        )
    with pytest.raises(
        freezer.ProtocolFreezeError,
        match="changed after development execution",
    ):
        freezer.validate_post_development_immutability(
            tmp_path,
            development_commit,
            ["locked-source.py"],
        )


@pytest.mark.parametrize(
    "relative_path",
    [
        "data/v4/cases.csv",
        "results/v3/formal/execution_attempts/sole-attempt.failed.json",
    ],
)
def test_freezer_rejects_post_development_corpus_or_v3_boundary_mutation(
    tmp_path: Path,
    relative_path: str,
) -> None:
    artifact = tmp_path / relative_path
    artifact.parent.mkdir(parents=True)
    artifact.write_text("development bytes\n", encoding="utf-8")
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "config", "user.name", "V4 Protocol Test")
    _git(tmp_path, "config", "user.email", "v4-protocol@example.test")
    _git(tmp_path, "add", relative_path)
    _git(tmp_path, "commit", "-qm", "development execution state")
    development_commit = _git(tmp_path, "rev-parse", "HEAD")

    artifact.write_text("post-development mutation\n", encoding="utf-8")
    _git(tmp_path, "add", relative_path)
    _git(tmp_path, "commit", "-qm", "mutated protocol source")

    with pytest.raises(freezer.ProtocolFreezeError):
        freezer.validate_post_development_immutability(
            tmp_path,
            development_commit,
            [relative_path],
        )
