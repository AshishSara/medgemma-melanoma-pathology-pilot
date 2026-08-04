from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

ROOT = Path(__file__).resolve().parents[1]
MODEL_IDS = ("google/medgemma-4b-it", "google/medgemma-1.5-4b-it")
TEMPLATE_IDS = ("A", "B")
CONDITIONS = ("clean", "ocr_degraded")

PRIMARY_FIELD_PATHS = (
    "specimen_site",
    "laterality",
    "diagnosis",
    "breslow_thickness_mm",
    "breslow_qualifier",
    "ulceration",
    "mitotic_rate_per_mm2",
    "mitotic_qualifier",
    "margins.invasive_peripheral",
    "margins.invasive_deep",
    "margins.in_situ_peripheral",
    "margins.in_situ_deep",
    "staging.pT",
    "staging.pN",
    "staging.pM",
    "staging.stage_group",
)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def sha256_text(value: str) -> str:
    return sha256_bytes(value.encode("utf-8"))


def null_if_blank(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    stripped = value.strip()
    return stripped if stripped else None


def number_if_present(value: Optional[str]) -> Optional[float]:
    normalized = null_if_blank(value)
    return None if normalized is None else float(normalized)


def load_cases() -> List[Dict[str, str]]:
    with (ROOT / "data" / "cases.csv").open(newline="", encoding="utf-8") as handle:
        cases = list(csv.DictReader(handle))
    if any(None in row for row in cases):
        raise ValueError("data/cases.csv contains extra columns")
    return cases


def document_id(case_id: str, template_id: str) -> str:
    return f"{case_id}-{template_id}"


def ground_truth_from_case(case: Mapping[str, str], template_id: str) -> Dict[str, Any]:
    return {
        "document_id": document_id(case["case_id"], template_id),
        "specimen_site": null_if_blank(case["specimen_site"]),
        "laterality": null_if_blank(case["laterality"]),
        "diagnosis": null_if_blank(case["diagnosis"]),
        "breslow_thickness_mm": number_if_present(case["breslow_thickness_mm"]),
        "breslow_qualifier": null_if_blank(case["breslow_qualifier"]),
        "ulceration": null_if_blank(case["ulceration"]),
        "mitotic_rate_per_mm2": number_if_present(case["mitotic_rate_per_mm2"]),
        "mitotic_qualifier": null_if_blank(case["mitotic_qualifier"]),
        "margins": {
            "invasive_peripheral": null_if_blank(case["invasive_peripheral"]),
            "invasive_deep": null_if_blank(case["invasive_deep"]),
            "in_situ_peripheral": null_if_blank(case["in_situ_peripheral"]),
            "in_situ_deep": null_if_blank(case["in_situ_deep"]),
        },
        "staging": {
            "pT": null_if_blank(case["pT"]),
            "pN": null_if_blank(case["pN"]),
            "pM": null_if_blank(case["pM"]),
            "stage_group": null_if_blank(case["stage_group"]),
        },
    }


def flatten_json(value: Mapping[str, Any], prefix: str = "") -> Dict[str, Any]:
    flattened: Dict[str, Any] = {}
    for key, child in value.items():
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(child, Mapping):
            flattened.update(flatten_json(child, path))
        else:
            flattened[path] = child
    return flattened


def select_primary_fields(value: Mapping[str, Any]) -> Dict[str, Any]:
    flattened = flatten_json(value)
    return {path: flattened.get(path) for path in PRIMARY_FIELD_PATHS}


def compare_prediction(truth: Mapping[str, Any], prediction: Mapping[str, Any]) -> Dict[str, Any]:
    truth_fields = select_primary_fields(truth)
    prediction_fields = select_primary_fields(prediction)
    document_id_correct = truth.get("document_id") == prediction.get("document_id")
    exact_correct = 0
    true_positive = 0
    false_positive = 0
    false_negative = 0
    unsupported_count = 0
    unsupported_opportunities = 0
    mismatched_fields: List[str] = []
    unsupported_fields: List[str] = []

    if not document_id_correct:
        mismatched_fields.append("document_id")

    for path in PRIMARY_FIELD_PATHS:
        expected = truth_fields[path]
        observed = prediction_fields[path]
        if expected == observed:
            exact_correct += 1
            if expected is not None:
                true_positive += 1
        else:
            mismatched_fields.append(path)
            if observed is not None:
                false_positive += 1
            if expected is not None:
                false_negative += 1

        if expected is None:
            unsupported_opportunities += 1
            if observed is not None:
                unsupported_count += 1
                unsupported_fields.append(path)

    precision = safe_divide(true_positive, true_positive + false_positive)
    recall = safe_divide(true_positive, true_positive + false_negative)
    f1 = safe_divide(2 * precision * recall, precision + recall)

    return {
        "field_exact_correct": exact_correct,
        "field_exact_total": len(PRIMARY_FIELD_PATHS),
        "field_exact_match": safe_divide(exact_correct, len(PRIMARY_FIELD_PATHS)),
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "unsupported_count": unsupported_count,
        "unsupported_opportunities": unsupported_opportunities,
        "unsupported_field_rate": safe_divide(unsupported_count, unsupported_opportunities),
        "document_id_correct": document_id_correct,
        "complete_report": document_id_correct and exact_correct == len(PRIMARY_FIELD_PATHS),
        "mismatched_fields": mismatched_fields,
        "unsupported_fields": unsupported_fields,
    }


def safe_divide(numerator: float, denominator: float) -> float:
    return 0.0 if denominator == 0 else numerator / denominator


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, fieldnames: Iterable[str], rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(fieldnames),
            extrasaction="ignore",
            lineterminator="\n",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def slugify_model_id(model_id: str) -> str:
    return model_id.replace("/", "__")


def relative(path: Path) -> str:
    return str(path.relative_to(ROOT))
