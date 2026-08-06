from __future__ import annotations

import copy
import csv
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import evaluate_v3  # noqa: E402
from pilot_utils import sha256_file  # noqa: E402
from v3_pipeline import (  # noqa: E402
    V3CompilationError,
    compile_prediction,
    sectionize_ocr_lines,
)


def candidate() -> dict:
    return {
        "document_id": "MEL-104-A",
        "specimen_site": "temple",
        "laterality": "right",
        "diagnosis": "invasive_melanoma",
        "breslow_thickness_mm": 0.7,
        "breslow_qualifier": "approximate",
        "ulceration": "not_identified",
        "mitotic_rate_per_mm2": 1,
        "mitotic_qualifier": "exact",
        "margins": {
            "invasive_peripheral": "not_involved",
            "invasive_deep": "not_involved",
            "in_situ_peripheral": "involved",
            "in_situ_deep": "not_involved",
        },
        "staging": {
            "pT": "pT1a",
            "pN": None,
            "pM": None,
            "stage_group": None,
        },
    }


def ocr_lines() -> list[dict]:
    text = [
        "SYNTHETIC RESEARCH REPORT - NOT A REAL PATIENT",
        "Document ID: MEL-104-A",
        "Procedure: Excision",
        "Specimen site: Right temple",
        "Diagnosis: Invasive melanoma, lentigo maligna melanoma",
        "Breslow thickness: Approximately 0.7 mm",
        "Ulceration: Not identified",
        "Mitotic rate: 1 per mm2",
        "Invasive melanoma - peripheral margin: Not involved",
        "Invasive melanoma - deep margin: Not involved",
        "Melanoma in situ - peripheral margin: Involved",
        "Melanoma in situ - deep margin: Not involved",
        "Reported pT: pT1a",
    ]
    return [{"line_id": f"L{index:03d}", "text": line} for index, line in enumerate(text, 1)]


def audit() -> dict:
    def present(*line_ids: str) -> dict:
        return {"status": "present", "evidence_line_ids": list(line_ids)}

    def absent() -> dict:
        return {"status": "absent", "evidence_line_ids": []}

    return {
        "document_id": present("L002"),
        "specimen_site": present("L004"),
        "laterality": present("L004"),
        "diagnosis": present("L005"),
        "breslow_thickness_mm": present("L006"),
        "breslow_qualifier": present("L006"),
        "ulceration": present("L007"),
        "mitotic_rate_per_mm2": present("L008"),
        "mitotic_qualifier": present("L008"),
        "margins": {
            "invasive_peripheral": present("L009"),
            "invasive_deep": present("L010"),
            "in_situ_peripheral": present("L011"),
            "in_situ_deep": present("L012"),
        },
        "staging": {
            "pT": present("L013"),
            "pN": absent(),
            "pM": absent(),
            "stage_group": absent(),
        },
    }


def test_sectionize_preserves_metadata_and_rejects_history() -> None:
    lines = [
        {"line_id": "L001", "text": "Document ID: MEL-108-B", "mean_confidence": 99},
        {"line_id": "L002", "text": "CLINICAL HISTORY - PRIOR REPORT ONLY"},
        {
            "line_id": "L003",
            "text": "Prior outside biopsy reported Breslow thickness 1.3 mm and pT2a.",
        },
        {"line_id": "L004", "text": "Forearm, Re-excision:"},
        {
            "line_id": "L005",
            "text": "Residual melanoma in situ; no residual invasive melanoma identified.",
        },
        {"line_id": "L006", "text": "COMMENT"},
        {"line_id": "L007", "text": "No residual invasive melanoma is identified."},
    ]
    sectioned = sectionize_ocr_lines(lines)
    assert [line["section"] for line in sectioned] == [
        "header",
        "history",
        "history",
        "current",
        "current",
        "footer",
        "footer",
    ]
    assert sectioned[0]["mean_confidence"] == 99


def test_sectionize_rejects_duplicate_or_invalid_ids() -> None:
    with pytest.raises(ValueError, match="Duplicate"):
        sectionize_ocr_lines(
            [
                {"line_id": "L001", "text": "one"},
                {"line_id": "L001", "text": "two"},
            ]
        )
    with pytest.raises(ValueError, match="Invalid"):
        sectionize_ocr_lines([{"line_id": "../bad", "text": "one"}])


def test_sectionize_explicit_synoptic_heading_and_colonless_labels() -> None:
    lines = [
        {"line_id": "L001", "text": "Document ID MEL-101-A"},
        {"line_id": "L002", "text": "CLINICAL HISTORY - PRIOR REPORT ONLY"},
        {"line_id": "L003", "text": "Specimen site Left ear"},
        {"line_id": "L004", "text": "CURRENT SPECIMEN - SYNOPTIC SUMMARY"},
        {"line_id": "L005", "text": "Procedure Shave biopsy"},
        {"line_id": "L006", "text": "Specimen site Right lower leq"},
    ]

    sectioned = sectionize_ocr_lines(lines)

    assert [line["section"] for line in sectioned] == [
        "header",
        "history",
        "history",
        "current",
        "current",
        "current",
    ]


def test_compile_accepts_only_three_way_agreement() -> None:
    final, log = compile_prediction(candidate(), audit(), ocr_lines())
    assert final == candidate()
    assert log["schema_valid"] is True
    assert log["accepted_non_null_count"] == 14
    assert log["rejected_non_null_count"] == 0


@pytest.mark.parametrize("unit", ["per mm2", "/mm2"])
def test_mitotic_rate_accepts_per_and_slash_units(unit: str) -> None:
    lines = ocr_lines()
    lines[7]["text"] = f"Mitotic rate: 1 {unit}"

    final, _ = compile_prediction(candidate(), audit(), lines)

    assert final["mitotic_rate_per_mm2"] == 1
    assert final["mitotic_qualifier"] == "exact"


def test_candidate_omission_stays_null_even_when_audit_says_present() -> None:
    value = candidate()
    value["ulceration"] = None
    final, log = compile_prediction(value, audit(), ocr_lines())
    assert final["ulceration"] is None
    assert log["fields"]["ulceration"]["reason"] == "candidate_null_or_omitted"


def test_candidate_evidence_disagreement_rejects_non_null_value() -> None:
    value = candidate()
    value["margins"]["in_situ_peripheral"] = "not_involved"
    final, log = compile_prediction(value, audit(), ocr_lines())
    assert final["margins"]["in_situ_peripheral"] is None
    assert log["fields"]["margins.in_situ_peripheral"]["reason"] == (
        "candidate_evidence_disagreement"
    )


def test_audit_absent_or_unknown_line_rejects_non_null_value() -> None:
    absent_audit = audit()
    absent_audit["ulceration"] = {"status": "absent", "evidence_line_ids": []}
    final, log = compile_prediction(candidate(), absent_audit, ocr_lines())
    assert final["ulceration"] is None
    assert log["fields"]["ulceration"]["reason"] == "audit_status_absent"

    bad_line_audit = audit()
    bad_line_audit["ulceration"]["evidence_line_ids"] = ["L999"]
    final, log = compile_prediction(candidate(), bad_line_audit, ocr_lines())
    assert final["ulceration"] is None
    assert log["fields"]["ulceration"]["reason"] == "unknown_evidence_line_id"


def test_history_evidence_is_never_accepted() -> None:
    value = candidate()
    value["breslow_thickness_mm"] = 1.3
    value["breslow_qualifier"] = "exact"
    lines = ocr_lines() + [
        {"line_id": "L014", "text": "CLINICAL HISTORY - PRIOR REPORT ONLY"},
        {"line_id": "L015", "text": "Prior Breslow thickness: 1.3 mm"},
    ]
    history_audit = audit()
    history_audit["breslow_thickness_mm"]["evidence_line_ids"] = ["L015"]
    history_audit["breslow_qualifier"]["evidence_line_ids"] = ["L015"]

    final, log = compile_prediction(value, history_audit, lines)
    assert final["breslow_thickness_mm"] is None
    assert final["breslow_qualifier"] is None
    assert log["history_evidence_rejection_count"] == 2


def test_qualifier_coherence_nulls_measurement_when_qualifier_fails() -> None:
    broken_audit = audit()
    broken_audit["breslow_qualifier"] = {
        "status": "unclear",
        "evidence_line_ids": ["L006"],
    }
    final, log = compile_prediction(candidate(), broken_audit, ocr_lines())
    assert final["breslow_thickness_mm"] is None
    assert final["breslow_qualifier"] is None
    assert log["fields"]["breslow_thickness_mm"]["reason"] == "coherence_qualifier_is_null"


def test_colonless_synoptic_compilation_uses_only_bounded_linked_normalization() -> None:
    value = candidate()
    value["document_id"] = "MEL-101-A"
    value["specimen_site"] = "lower leg"
    value["breslow_qualifier"] = None
    value["mitotic_qualifier"] = None

    lines = [
        {"line_id": "L001", "text": "SYNTHETIC RESEARCH REPORT - NOT A REAL PATIENT"},
        {"line_id": "L002", "text": "Document ID MEL-101-A"},
        {"line_id": "L003", "text": "CURRENT SPECIMEN - SYNOPTIC SUMMARY"},
        {"line_id": "L004", "text": "Procedure Excision"},
        {"line_id": "L005", "text": "Specimen site Right lower leq"},
        {
            "line_id": "L006",
            "text": "Diagnosis Invasive melanoma, lentigo maligna melanoma",
        },
        {"line_id": "L007", "text": "Breslow thickness Approximately 0.7 mm"},
        {"line_id": "L008", "text": "Ulceration Not identified"},
        {"line_id": "L009", "text": "Mitotic rate 1 per mm2"},
        {
            "line_id": "L010",
            "text": "Invasive melanoma - peripheral margin Not involved",
        },
        {
            "line_id": "L011",
            "text": "Invasive melanoma - deep margin Not involved",
        },
        {
            "line_id": "L012",
            "text": "Melanoma in situ - peripheral margin Involved",
        },
        {
            "line_id": "L013",
            "text": "Melanoma in situ - deep margin Not involved",
        },
        {"line_id": "L014", "text": "Reported pT pT1a"},
    ]

    colonless_audit = copy.deepcopy(audit())
    colonless_audit["document_id"] = {"status": "absent", "evidence_line_ids": []}
    colonless_audit["specimen_site"]["evidence_line_ids"] = ["L005"]
    colonless_audit["laterality"] = {"status": "absent", "evidence_line_ids": []}
    colonless_audit["diagnosis"]["evidence_line_ids"] = ["L006"]
    colonless_audit["breslow_thickness_mm"]["evidence_line_ids"] = ["L007"]
    colonless_audit["breslow_qualifier"] = {"status": "absent", "evidence_line_ids": []}
    colonless_audit["ulceration"]["evidence_line_ids"] = ["L008"]
    colonless_audit["mitotic_rate_per_mm2"]["evidence_line_ids"] = ["L009"]
    colonless_audit["mitotic_qualifier"] = {"status": "absent", "evidence_line_ids": []}
    colonless_audit["margins"]["invasive_peripheral"]["evidence_line_ids"] = ["L010"]
    colonless_audit["margins"]["invasive_deep"]["evidence_line_ids"] = ["L011"]
    colonless_audit["margins"]["in_situ_peripheral"]["evidence_line_ids"] = ["L012"]
    colonless_audit["margins"]["in_situ_deep"]["evidence_line_ids"] = ["L013"]
    colonless_audit["staging"]["pT"]["evidence_line_ids"] = ["L014"]

    final, log = compile_prediction(value, colonless_audit, lines)

    expected = copy.deepcopy(value)
    expected["breslow_qualifier"] = "approximate"
    expected["mitotic_qualifier"] = "exact"
    assert final == expected
    assert log["fields"]["document_id"]["reason"] == ("accepted_labeled_header_identity_fallback")
    assert log["fields"]["specimen_site"]["reason"] == ("accepted_bounded_site_ocr_substitution")
    assert log["fields"]["specimen_site"]["normalization"]["candidate_token"] == "leg"
    assert log["fields"]["specimen_site"]["normalization"]["ocr_token"] == "leq"
    assert log["fields"]["laterality"]["reason"] == ("accepted_from_specimen_site_evidence")
    assert log["fields"]["breslow_qualifier"]["reason"] == (
        "derived_from_accepted_measurement_evidence"
    )
    assert log["fields"]["mitotic_qualifier"]["reason"] == (
        "derived_from_accepted_measurement_evidence"
    )


def test_site_ocr_normalization_rejects_nonconfusable_or_multiple_substitutions() -> None:
    value = candidate()
    value["specimen_site"] = "lower leg"
    site_audit = audit()

    one_nonconfusable = ocr_lines()
    one_nonconfusable[3]["text"] = "Specimen site: Right lower led"
    final, log = compile_prediction(value, site_audit, one_nonconfusable)
    assert final["specimen_site"] is None
    assert log["fields"]["specimen_site"]["reason"] == "candidate_evidence_disagreement"
    assert log["fields"]["specimen_site"]["normalization"] is None

    two_confusable = ocr_lines()
    two_confusable[3]["text"] = "Specimen site: Right iower leq"
    final, log = compile_prediction(value, site_audit, two_confusable)
    assert final["specimen_site"] is None
    assert log["fields"]["specimen_site"]["reason"] == "candidate_evidence_disagreement"
    assert log["fields"]["specimen_site"]["normalization"] is None


def test_missing_qualifier_is_not_derived_without_an_accepted_measurement() -> None:
    value = candidate()
    value["breslow_thickness_mm"] = None
    value["breslow_qualifier"] = None

    final, log = compile_prediction(value, audit(), ocr_lines())

    assert final["breslow_thickness_mm"] is None
    assert final["breslow_qualifier"] is None
    assert log["fields"]["breslow_qualifier"]["reason"] == "candidate_null_or_omitted"
    assert log["fields"]["breslow_qualifier"]["normalization"] is None


def test_missing_document_id_cannot_be_fabricated() -> None:
    value = candidate()
    del value["document_id"]
    with pytest.raises(V3CompilationError) as exc_info:
        compile_prediction(value, audit(), ocr_lines())
    assert exc_info.value.audit_log["schema_valid"] is False


def test_narrative_combined_lines_parse_each_atomic_field() -> None:
    value = candidate()
    lines = [
        {"line_id": "L001", "text": "Document ID: MEL-104-A"},
        {"line_id": "L002", "text": "Right temple, Excision:"},
        {
            "line_id": "L003",
            "text": "Invasive melanoma, lentigo maligna melanoma.",
        },
        {
            "line_id": "L004",
            "text": "Breslow thickness is Approximately 0.7 mm.",
        },
        {"line_id": "L005", "text": "Ulceration: not identified."},
        {"line_id": "L006", "text": "Mitotic rate is 1 per mm2."},
        {
            "line_id": "L007",
            "text": (
                "invasive melanoma peripheral margin: not involved; "
                "invasive melanoma deep margin: not involved; "
                "melanoma in situ peripheral margin: involved; "
                "melanoma in situ deep margin: not involved."
            ),
        },
        {
            "line_id": "L008",
            "text": "Reported pathologic classification - pT: pT1a.",
        },
    ]

    narrative_audit = copy.deepcopy(audit())
    evidence = {
        "document_id": "L001",
        "specimen_site": "L002",
        "laterality": "L002",
        "diagnosis": "L003",
        "breslow_thickness_mm": "L004",
        "breslow_qualifier": "L004",
        "ulceration": "L005",
        "mitotic_rate_per_mm2": "L006",
        "mitotic_qualifier": "L006",
        "margins.invasive_peripheral": "L007",
        "margins.invasive_deep": "L007",
        "margins.in_situ_peripheral": "L007",
        "margins.in_situ_deep": "L007",
        "staging.pT": "L008",
    }
    for path, line_id in evidence.items():
        node = narrative_audit
        parts = path.split(".")
        for part in parts[:-1]:
            node = node[part]
        node[parts[-1]]["evidence_line_ids"] = [line_id]

    final, _ = compile_prediction(value, narrative_audit, lines)
    assert final == value


def test_evaluator_does_not_open_ground_truth_before_complete_preflight(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = [
        {
            "document_id": f"MEL-{104 + index // 4:03d}-{'A' if index % 4 < 2 else 'B'}",
            "condition": "clean" if index % 2 == 0 else "ocr_degraded",
        }
        for index in range(20)
    ]
    monkeypatch.setattr(
        evaluate_v3,
        "verify_protocol_lock",
        lambda: {"protocol_source_commit": "0" * 40},
    )
    monkeypatch.setattr(evaluate_v3, "_load_formal_manifest", lambda: manifest)
    monkeypatch.setattr(
        evaluate_v3,
        "_preflight_row",
        lambda row, protocol_lock: {"row": row, "issues": ["missing formal artifact"]},
    )

    def forbidden_ground_truth_read(*args: object, **kwargs: object) -> object:
        raise AssertionError("ground truth must remain closed during incomplete preflight")

    written: dict = {}
    monkeypatch.setattr(evaluate_v3, "read_json", forbidden_ground_truth_read)
    monkeypatch.setattr(
        evaluate_v3,
        "_verify_evaluation_only_corpus",
        forbidden_ground_truth_read,
    )
    monkeypatch.setattr(
        evaluate_v3,
        "write_json",
        lambda path, payload: written.update({"path": path, "payload": payload}),
    )

    payload = evaluate_v3.evaluate()
    assert payload["ground_truth_read"] is False
    assert payload["provenance_valid_count"] == 0
    assert written["payload"] == payload


def test_evaluation_only_hash_mismatch_precedes_semantic_truth_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_path = tmp_path / "data" / "v3" / "report_manifest.csv"
    cases_path = tmp_path / "data" / "v3" / "cases.csv"
    source_path = tmp_path / "data" / "v3" / "source_text" / "MEL-104-A.txt"
    truth_path = tmp_path / "data" / "v3" / "ground_truth" / "MEL-104-A.json"
    for path in (manifest_path, cases_path, source_path, truth_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    cases_path.write_text("case_id\nMEL-104\n", encoding="utf-8")
    source_path.write_text("report\n", encoding="utf-8")
    truth_path.write_text('{"document_id":"MEL-104-A"}\n', encoding="utf-8")
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("source_text_path", "ground_truth_path"),
        )
        writer.writeheader()
        writer.writerow(
            {
                "source_text_path": "data/v3/source_text/MEL-104-A.txt",
                "ground_truth_path": "data/v3/ground_truth/MEL-104-A.json",
            }
        )
    lock = {
        "evaluation_only_corpus_artifacts": {
            "data/v3/cases.csv": sha256_file(cases_path),
            "data/v3/source_text/MEL-104-A.txt": sha256_file(source_path),
            "data/v3/ground_truth/MEL-104-A.json": sha256_file(truth_path),
        }
    }
    monkeypatch.setattr(evaluate_v3, "ROOT", tmp_path)
    monkeypatch.setattr(evaluate_v3, "V3_MANIFEST", manifest_path)
    assert len(evaluate_v3._verify_evaluation_only_corpus(lock)) == 3

    truth_path.write_text('{"tampered":true}\n', encoding="utf-8")
    semantic_reads: list[Path] = []

    def forbidden_semantic_read(path: Path) -> object:
        semantic_reads.append(path)
        raise AssertionError("semantic truth read must not occur")

    monkeypatch.setattr(evaluate_v3, "read_json", forbidden_semantic_read)
    with pytest.raises(ValueError, match="hash mismatch"):
        evaluate_v3._verify_evaluation_only_corpus(lock)
    assert semantic_reads == []
