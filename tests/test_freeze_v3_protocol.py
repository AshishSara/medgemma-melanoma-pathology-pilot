from __future__ import annotations

import copy
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import freeze_v3_protocol as freezer  # noqa: E402
import run_v3_inference as inference  # noqa: E402
from pilot_utils import sha256_file  # noqa: E402


def _passing_final_pipeline() -> dict[str, Any]:
    aggregate = {
        "output_count": 12,
        "schema_valid_count": 12,
        "schema_valid_rate": 1.0,
        "field_exact_match": 0.9,
        "unsupported_field_rate": 0.02,
    }
    condition = {
        **aggregate,
        "output_count": 6,
        "schema_valid_count": 6,
        "field_exact_match": 0.85,
    }
    return {
        "overall": aggregate,
        "by_condition": [
            {"condition": "clean", **condition},
            {"condition": "ocr_degraded", **condition},
        ],
    }


def _passing_evidence() -> dict[str, Any]:
    return {
        "accepted_non_null_count": 30,
        "accepted_with_valid_non_history_evidence_count": 30,
        "accepted_non_null_evidence_rate": 1.0,
        "history_carryover_count": 0,
    }


def _runtime_record() -> dict[str, Any]:
    return {
        "resolved_revision": freezer.MODEL_REVISION,
        "device": "cuda:0",
        "dtype": "torch.bfloat16",
        "text_vocab_size": 262208,
        "cuda_version": "12.4",
        "cuda_device_name": "Tesla T4",
        "accelerate_version": "1.14.0",
        "torch_version": "2.6.0+cu124",
        "transformers_version": "4.57.6",
        "lm_format_enforcer_version": "0.11.3",
        "random_seed": 0,
        "backend": "transformers-direct",
        "requested_dtype": "bfloat16",
        "platform": "Linux-test",
        "python_version": "3.12.0",
        "ocr": {
            "engine": "tesseract",
            "tesseract_version": "5.3.4",
            "pytesseract_version": "0.3.13",
            "language": "eng",
            "config": "--oem 1 --psm 6",
            "image_preprocessing": "none",
            "output_type": "pytesseract.Output.DICT",
            "line_grouping": "ordered page_num/block_num/par_num/line_num word groups",
            "line_id_assignment": "L001..Lnnn in first-seen grouped OCR order",
            "timeout_seconds": 120,
        },
    }


def _git(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=repo, text=True).strip()


def _init_repo(repo: Path) -> str:
    _git(repo, "init", "-q")
    _git(repo, "config", "user.name", "Protocol Test")
    _git(repo, "config", "user.email", "protocol@example.test")
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "frozen inputs")
    return _git(repo, "rev-parse", "HEAD")


def _write_minimal_v3_corpus(repo: Path) -> tuple[dict[str, str], dict[str, str]]:
    manifest_path = repo / "data" / "v3" / "report_manifest.csv"
    cases_path = repo / "data" / "v3" / "cases.csv"
    source_path = repo / "data" / "v3" / "source_text" / "MEL-TEST-A.txt"
    truth_path = repo / "data" / "v3" / "ground_truth" / "MEL-TEST-A.json"
    pdf_path = repo / "output" / "v3" / "pdf" / "clean" / "MEL-TEST-A.pdf"
    image_path = repo / "output" / "v3" / "rendered" / "clean" / "MEL-TEST-A.png"
    for path, content in (
        (cases_path, "case_id\nMEL-TEST\n"),
        (source_path, "synthetic report\n"),
        (truth_path, "{}\n"),
        (pdf_path, "pdf\n"),
        (image_path, "image\n"),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "pdf_path",
                "image_path",
                "source_text_path",
                "ground_truth_path",
            ),
        )
        writer.writeheader()
        writer.writerow(
            {
                "pdf_path": "output/v3/pdf/clean/MEL-TEST-A.pdf",
                "image_path": "output/v3/rendered/clean/MEL-TEST-A.png",
                "source_text_path": "data/v3/source_text/MEL-TEST-A.txt",
                "ground_truth_path": "data/v3/ground_truth/MEL-TEST-A.json",
            }
        )
    return (
        {
            "data/v3/report_manifest.csv": sha256_file(manifest_path),
            "output/v3/pdf/clean/MEL-TEST-A.pdf": sha256_file(pdf_path),
            "output/v3/rendered/clean/MEL-TEST-A.png": sha256_file(image_path),
        },
        {
            "data/v3/cases.csv": sha256_file(cases_path),
            "data/v3/source_text/MEL-TEST-A.txt": sha256_file(source_path),
            "data/v3/ground_truth/MEL-TEST-A.json": sha256_file(truth_path),
        },
    )


def test_development_gate_accepts_exact_thresholds() -> None:
    gate = freezer.require_development_gate(
        _passing_final_pipeline(),
        _passing_evidence(),
    )
    assert all(gate["criteria"].values())
    assert gate["pooled_field_exact_match"] == 0.9
    assert gate["condition_field_exact_match"] == {
        "clean": 0.85,
        "ocr_degraded": 0.85,
    }


@pytest.mark.parametrize(
    ("mutation", "expected_failure"),
    [
        (("overall", "schema_valid_rate", 0.99), "schema_100_percent"),
        (("overall", "field_exact_match", 0.899), "pooled_exact"),
        (("overall", "unsupported_field_rate", 0.021), "unsupported"),
        (("clean", "field_exact_match", 0.849), "each_condition"),
        (("evidence", "accepted_non_null_evidence_rate", 0.99), "evidence_acceptance"),
        (("evidence", "history_carryover_count", 1), "zero_history"),
    ],
)
def test_development_gate_fails_closed(
    mutation: tuple[str, str, float | int],
    expected_failure: str,
) -> None:
    pipeline = _passing_final_pipeline()
    evidence = _passing_evidence()
    scope, key, value = mutation
    if scope == "overall":
        pipeline["overall"][key] = value
    elif scope == "evidence":
        evidence[key] = value
    else:
        condition = next(item for item in pipeline["by_condition"] if item["condition"] == scope)
        condition[key] = value
    with pytest.raises(freezer.ProtocolFreezeError, match=expected_failure):
        freezer.require_development_gate(pipeline, evidence)


def test_common_runtime_requires_all_twelve_rows_to_be_identical() -> None:
    records = [copy.deepcopy(_runtime_record()) for _ in range(12)]
    runtime = freezer.common_runtime(records)
    assert runtime["cuda_device_name"] == "Tesla T4"
    assert runtime["dtype"] == "torch.bfloat16"

    records[7]["ocr"]["tesseract_version"] = "5.4.0"
    with pytest.raises(freezer.ProtocolFreezeError, match="runtime metadata differs"):
        freezer.common_runtime(records)


def test_formal_lock_includes_shared_utils_and_the_freezer() -> None:
    assert "scripts/pilot_utils.py" in inference.REQUIRED_LOCKED_SOURCE_PATHS
    assert "scripts/freeze_v3_protocol.py" in inference.REQUIRED_LOCKED_SOURCE_PATHS


def test_corpus_inventory_is_closed_and_ground_truth_is_opaque(tmp_path: Path) -> None:
    paths = {
        "data/v3/cases.csv": b"case_id\nMEL-TEST\n",
        "data/v3/report_manifest.csv": b"manifest\n",
        "data/v3/source_text/MEL-TEST-A.txt": b"synthetic report",
        # Deliberately not JSON: corpus collection hashes this file but never deserializes it.
        "data/v3/ground_truth/MEL-TEST-A.json": b"opaque formal truth bytes",
        "output/v3/pdf/clean/MEL-TEST-A.pdf": b"pdf bytes",
        "output/v3/rendered/clean/MEL-TEST-A.png": b"png bytes",
    }
    for relative_path, content in paths.items():
        path = tmp_path / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    row = {
        "document_id": "MEL-TEST-A",
        "condition": "clean",
        "source_text_path": "data/v3/source_text/MEL-TEST-A.txt",
        "ground_truth_path": "data/v3/ground_truth/MEL-TEST-A.json",
        "pdf_path": "output/v3/pdf/clean/MEL-TEST-A.pdf",
        "image_path": "output/v3/rendered/clean/MEL-TEST-A.png",
        "pdf_sha256": sha256_file(tmp_path / "output/v3/pdf/clean/MEL-TEST-A.pdf"),
        "image_sha256": sha256_file(tmp_path / "output/v3/rendered/clean/MEL-TEST-A.png"),
    }
    hashes = freezer.collect_corpus_artifacts(tmp_path, [row])
    assert set(hashes) == set(paths)
    assert hashes[row["ground_truth_path"]] == sha256_file(tmp_path / row["ground_truth_path"])

    extra = tmp_path / "data" / "v3" / "source_text" / "unbound.txt"
    extra.write_text("not in manifest", encoding="utf-8")
    with pytest.raises(freezer.ProtocolFreezeError, match="inventory differs"):
        freezer.collect_corpus_artifacts(tmp_path, [row])


def test_manifest_rejects_formal_case_mislabeled_as_development() -> None:
    with (ROOT / "data" / "v3" / "report_manifest.csv").open(
        newline="",
        encoding="utf-8",
    ) as handle:
        rows = list(csv.DictReader(handle))
    development_row = next(row for row in rows if row["split"] == "development")
    formal_row = next(row for row in rows if row["split"] == "formal")
    development_row["split"] = "formal"
    formal_row["split"] = "development"
    with pytest.raises(freezer.ProtocolFreezeError, match="frozen case"):
        freezer.validate_manifest_rows(rows)


def test_documented_iterations_preserve_failure_and_cap_budget(tmp_path: Path) -> None:
    iteration_1 = tmp_path / "results" / "v3" / "development" / "iteration-1"
    iteration_2 = tmp_path / "results" / "v3" / "development" / "iteration-2"
    iteration_1.mkdir(parents=True)
    iteration_2.mkdir(parents=True)
    (iteration_1 / "iteration_attempt.json").write_text(
        json.dumps(
            {
                "status": "failed_development_smoke",
                "ground_truth_used_during_inference": False,
            }
        ),
        encoding="utf-8",
    )
    (iteration_2 / "iteration_record.json").write_text("{}\n", encoding="utf-8")

    documented = freezer.collect_documented_development_iterations(tmp_path, 2)
    assert [item["status"] for item in documented["summaries"]] == [
        "failed_preserved",
        "selected_complete",
    ]
    assert documented["artifacts"]

    conflicting = iteration_1 / "iteration_record.json"
    conflicting.write_text("{}\n", encoding="utf-8")
    with pytest.raises(freezer.ProtocolFreezeError, match="both failed and completed"):
        freezer.collect_documented_development_iterations(tmp_path, 2)
    conflicting.unlink()

    iteration_3 = tmp_path / "results" / "v3" / "development" / "iteration-3"
    iteration_3.mkdir()
    (iteration_3 / "iteration_attempt.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(freezer.ProtocolFreezeError, match="More than two"):
        freezer.collect_documented_development_iterations(tmp_path, 2)


def test_protocol_source_commit_must_be_exact_clean_current_head(tmp_path: Path) -> None:
    tracked = tmp_path / "tracked.txt"
    tracked.write_text("frozen\n", encoding="utf-8")
    head = _init_repo(tmp_path)
    (tmp_path / "unrelated-untracked.txt").write_text("allowed noise\n", encoding="utf-8")
    freezer.validate_protocol_source_commit(tmp_path, head, ["tracked.txt"])

    tracked.write_text("changed\n", encoding="utf-8")
    with pytest.raises(freezer.ProtocolFreezeError, match="must be committed"):
        freezer.validate_protocol_source_commit(tmp_path, head, ["tracked.txt"])

    tracked.write_text("frozen\n", encoding="utf-8")
    (tmp_path / "second.txt").write_text("new commit\n", encoding="utf-8")
    _git(tmp_path, "add", "second.txt")
    _git(tmp_path, "commit", "-qm", "new head")
    with pytest.raises(freezer.ProtocolFreezeError, match="must equal current HEAD"):
        freezer.validate_protocol_source_commit(tmp_path, head, ["tracked.txt"])


def test_post_development_change_may_only_harden_formal_lock_validation(
    tmp_path: Path,
) -> None:
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    inference_path = scripts / "run_v3_inference.py"
    inference_path.write_text(
        "REQUIRED_LOCKED_SOURCE_PATHS = ('old.py',)\n"
        "def verify_protocol_lock():\n"
        "    return 'old verifier'\n"
        "def run_row():\n"
        "    return 'unchanged inference'\n",
        encoding="utf-8",
    )
    (scripts / "pilot_utils.py").write_text("MODEL = 'unchanged'\n", encoding="utf-8")
    development_commit = _init_repo(tmp_path)

    inference_path.write_text(
        "REQUIRED_LOCKED_SOURCE_PATHS = ('old.py', 'new.py')\n"
        "def verify_protocol_lock():\n"
        "    return 'hardened verifier'\n"
        "def run_row():\n"
        "    return 'unchanged inference'\n",
        encoding="utf-8",
    )
    transition = freezer.validate_post_development_source_hardening(
        tmp_path,
        development_commit,
    )
    assert transition["classification"] == "formal_provenance_hardening_only"
    assert (
        transition["development_inference_script_sha256"]
        != transition["formal_inference_script_sha256"]
    )

    inference_path.write_text(
        "REQUIRED_LOCKED_SOURCE_PATHS = ('old.py', 'new.py')\n"
        "def verify_protocol_lock():\n"
        "    return 'hardened verifier'\n"
        "def run_row():\n"
        "    return 'changed inference semantics'\n",
        encoding="utf-8",
    )
    with pytest.raises(freezer.ProtocolFreezeError, match="Inference semantics"):
        freezer.validate_post_development_source_hardening(
            tmp_path,
            development_commit,
        )


def test_built_lock_is_accepted_by_formal_runner_validator(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for relative_path in inference.REQUIRED_LOCKED_SOURCE_PATHS:
        path = tmp_path / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"{relative_path}\n", encoding="utf-8")
    corpus_artifacts, evaluation_only_artifacts = _write_minimal_v3_corpus(tmp_path)
    source_commit = _init_repo(tmp_path)
    locked_sources = {
        relative_path: sha256_file(tmp_path / relative_path)
        for relative_path in inference.REQUIRED_LOCKED_SOURCE_PATHS
    }
    development = {
        "repository_commit": source_commit,
        "post_development_source_hardening": {"classification": "formal_provenance_hardening_only"},
        "runtime": freezer.common_runtime([copy.deepcopy(_runtime_record()) for _ in range(12)]),
        "gate": {
            "criteria": {"all": True},
            "provenance_valid_count": 12,
        },
        "metrics_sha256": "a" * 64,
        "iteration_record_sha256": "b" * 64,
    }
    payload = freezer.build_lock_payload(
        protocol_source_commit=source_commit,
        selected_iteration=2,
        development=development,
        documented_iterations={
            "summaries": [
                {"iteration": 1, "status": "failed_preserved"},
                {"iteration": 2, "status": "selected_complete"},
            ],
            "artifacts": {},
        },
        locked_source_files=locked_sources,
        corpus_artifacts=corpus_artifacts,
        evaluation_only_corpus_artifacts=evaluation_only_artifacts,
    )
    lock_path = tmp_path / "data" / "v3" / "protocol_lock.json"
    lock_path.write_text(json.dumps(payload), encoding="utf-8")
    _git(tmp_path, "add", "data/v3/protocol_lock.json")
    _git(tmp_path, "commit", "-qm", "commit protocol lock")

    monkeypatch.setattr(inference, "ROOT", tmp_path)
    monkeypatch.setattr(inference, "PROTOCOL_LOCK_PATH", lock_path)
    real_sha256_file = inference.sha256_file
    opened_paths: list[str] = []

    def guarded_operational_hash(path: Path) -> str:
        relative_path = path.relative_to(tmp_path).as_posix()
        opened_paths.append(relative_path)
        assert not (
            relative_path == "data/v3/cases.csv"
            or relative_path.startswith(("data/v3/source_text/", "data/v3/ground_truth/"))
        )
        return real_sha256_file(path)

    monkeypatch.setattr(inference, "sha256_file", guarded_operational_hash)
    verified = inference.verify_protocol_lock()
    assert verified["protocol_source_commit"] == source_commit
    assert verified["development_iterations_used"] == 2
    assert verified["pass_criteria"] == freezer.PASS_CRITERIA
    assert opened_paths

    valid_payload = copy.deepcopy(payload)
    traversal_path = "output/v3/rendered/clean/../../../../data/v3/ground_truth/MEL-TEST-A.json"
    payload["corpus_artifacts"][traversal_path] = evaluation_only_artifacts[
        "data/v3/ground_truth/MEL-TEST-A.json"
    ]
    lock_path.write_text(json.dumps(payload), encoding="utf-8")
    _git(tmp_path, "add", "data/v3/protocol_lock.json")
    _git(tmp_path, "commit", "-qm", "malformed traversal lock")
    opened_paths.clear()
    with pytest.raises(SystemExit, match="unsafe or invalid"):
        inference.verify_protocol_lock()
    assert traversal_path not in opened_paths

    lock_path.write_text(json.dumps(valid_payload), encoding="utf-8")
    _git(tmp_path, "add", "data/v3/protocol_lock.json")
    _git(tmp_path, "commit", "-qm", "restore valid lock")
    manifest_path = tmp_path / "data" / "v3" / "report_manifest.csv"
    manifest_path.unlink()
    manifest_path.symlink_to("ground_truth/MEL-TEST-A.json")
    opened_paths.clear()
    with pytest.raises(SystemExit, match="unsafe locked source"):
        inference.verify_protocol_lock()
    assert "data/v3/report_manifest.csv" not in opened_paths


def test_formal_lock_validator_rejects_untracked_or_modified_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for relative_path in inference.REQUIRED_LOCKED_SOURCE_PATHS:
        path = tmp_path / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"{relative_path}\n", encoding="utf-8")
    corpus_artifacts, evaluation_only_artifacts = _write_minimal_v3_corpus(tmp_path)
    source_commit = _init_repo(tmp_path)
    payload = freezer.build_lock_payload(
        protocol_source_commit=source_commit,
        selected_iteration=2,
        development={
            "repository_commit": source_commit,
            "post_development_source_hardening": {
                "classification": "formal_provenance_hardening_only"
            },
            "runtime": freezer.common_runtime(
                [copy.deepcopy(_runtime_record()) for _ in range(12)]
            ),
            "gate": {"criteria": {"all": True}, "provenance_valid_count": 12},
            "metrics_sha256": "a" * 64,
            "iteration_record_sha256": "b" * 64,
        },
        documented_iterations={"summaries": [], "artifacts": {}},
        locked_source_files={
            relative_path: sha256_file(tmp_path / relative_path)
            for relative_path in inference.REQUIRED_LOCKED_SOURCE_PATHS
        },
        corpus_artifacts=corpus_artifacts,
        evaluation_only_corpus_artifacts=evaluation_only_artifacts,
    )
    lock_path = tmp_path / "data" / "v3" / "protocol_lock.json"
    lock_path.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(inference, "ROOT", tmp_path)
    monkeypatch.setattr(inference, "PROTOCOL_LOCK_PATH", lock_path)

    with pytest.raises(SystemExit, match="not tracked"):
        inference.verify_protocol_lock()

    _git(tmp_path, "add", "data/v3/protocol_lock.json")
    _git(tmp_path, "commit", "-qm", "commit protocol lock")
    lock_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    with pytest.raises(SystemExit, match="differs from the execution"):
        inference.verify_protocol_lock()


def test_freezer_never_overwrites_a_lock(tmp_path: Path) -> None:
    lock_path = tmp_path / "data" / "v3" / "protocol_lock.json"
    lock_path.parent.mkdir(parents=True)
    lock_path.write_text('{"existing": true}\n', encoding="utf-8")
    with pytest.raises(freezer.ProtocolFreezeError, match="Refusing to overwrite"):
        freezer.freeze_protocol(
            2,
            "a" * 40,
            root=tmp_path,
            lock_path=lock_path,
        )


def test_metric_recomputation_refuses_formal_rows_before_truth_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reads: list[Path] = []

    def record_read(path: Path) -> Any:
        reads.append(path)
        raise AssertionError("ground truth should not be read")

    monkeypatch.setattr(freezer, "read_json", record_read)
    with pytest.raises(freezer.ProtocolFreezeError, match="outside the development split"):
        freezer._recomputed_metrics(
            tmp_path,
            [],
            [
                {
                    "split": "formal",
                    "document_id": "MEL-104-A",
                    "ground_truth_path": "data/v3/ground_truth/MEL-104-A.json",
                }
            ],
        )
    assert reads == []


def test_committed_repository_lock_was_not_changed_during_freezer_tests() -> None:
    lock_path = ROOT / "data" / "v3" / "protocol_lock.json"
    committed = subprocess.check_output(
        ["git", "cat-file", "blob", "HEAD:data/v3/protocol_lock.json"],
        cwd=ROOT,
    )
    assert lock_path.read_bytes() == committed
