#!/usr/bin/env python3
from __future__ import annotations

import csv
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from output_parser import parse_file
from pilot_utils import (
    MODEL_IDS,
    PRIMARY_FIELD_PATHS,
    ROOT,
    compare_prediction,
    read_json,
    safe_divide,
    select_primary_fields,
    slugify_model_id,
    write_csv,
    write_json,
)

PILOT_FIELDS = (
    "model_id",
    "semantic_case_id",
    "document_id",
    "template_id",
    "condition",
    "run_status",
    "raw_present",
    "parse_valid",
    "schema_valid",
    "document_id_correct",
    "field_exact_correct",
    "field_exact_total",
    "field_exact_match",
    "true_positive",
    "false_positive",
    "false_negative",
    "precision",
    "recall",
    "f1",
    "unsupported_count",
    "unsupported_opportunities",
    "unsupported_field_rate",
    "complete_report",
)

REVIEW_FIELDS = (
    "model_id",
    "semantic_case_id",
    "document_id",
    "template_id",
    "condition",
    "review_status",
    "parse_valid",
    "schema_valid",
    "document_id_correct",
    "mismatched_fields",
    "unsupported_fields",
    "source_ambiguity",
    "adjudication",
    "reviewer",
    "review_date",
)


def load_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def row_key(row: Mapping[str, Any]) -> Tuple[str, str, str]:
    return row["model_id"], row["document_id"], row["condition"]


def raw_output_path(model_id: str, condition: str, document_id: str) -> Path:
    return ROOT / "results" / "raw" / slugify_model_id(model_id) / condition / f"{document_id}.txt"


def parse_failure_metrics(truth: Mapping[str, Any]) -> Dict[str, Any]:
    truth_fields = select_primary_fields(truth)
    non_null = sum(value is not None for value in truth_fields.values())
    null_count = len(PRIMARY_FIELD_PATHS) - non_null
    return {
        "field_exact_correct": 0,
        "field_exact_total": len(PRIMARY_FIELD_PATHS),
        "field_exact_match": 0.0,
        "true_positive": 0,
        "false_positive": 0,
        "false_negative": non_null,
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0,
        "unsupported_count": 0,
        "unsupported_opportunities": null_count,
        "unsupported_field_rate": 0.0,
        "document_id_correct": False,
        "complete_report": False,
        "mismatched_fields": ["document_id", *PRIMARY_FIELD_PATHS],
        "unsupported_fields": [],
    }


def evaluate_rows() -> List[Dict[str, Any]]:
    manifest = load_csv(ROOT / "data" / "report_manifest.csv")
    rows: List[Dict[str, Any]] = []
    for model_id in MODEL_IDS:
        for item in manifest:
            truth = read_json(ROOT / item["ground_truth_path"])
            raw_path = raw_output_path(model_id, item["condition"], item["document_id"])
            base = {
                "model_id": model_id,
                "semantic_case_id": item["semantic_case_id"],
                "document_id": item["document_id"],
                "template_id": item["template_id"],
                "condition": item["condition"],
                "_truth": truth,
            }
            if not raw_path.exists():
                rows.append(
                    {
                        **base,
                        "run_status": "NOT_RUN",
                        "raw_present": False,
                        "parse_valid": None,
                        "schema_valid": None,
                        "_metrics": None,
                        "_parse_metadata": None,
                    }
                )
                continue

            prediction, parse_metadata = parse_file(raw_path)
            metrics = (
                compare_prediction(truth, prediction)
                if prediction is not None
                else parse_failure_metrics(truth)
            )
            metrics["complete_report"] = bool(
                metrics["complete_report"] and parse_metadata["schema_valid"]
            )
            rows.append(
                {
                    **base,
                    "run_status": "COMPLETED",
                    "raw_present": True,
                    "parse_valid": parse_metadata["parse_valid"],
                    "schema_valid": parse_metadata["schema_valid"],
                    "_metrics": metrics,
                    "_parse_metadata": parse_metadata,
                }
            )
    return rows


def pilot_row(row: Mapping[str, Any]) -> Dict[str, Any]:
    result = {field: row.get(field, "") for field in PILOT_FIELDS}
    metrics = row["_metrics"]
    if metrics is not None:
        for field in PILOT_FIELDS:
            if field in metrics:
                result[field] = metrics[field]
    return result


def aggregate(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    completed = [row for row in rows if row["raw_present"]]
    parse_valid = sum(bool(row["parse_valid"]) for row in completed)
    schema_valid = sum(bool(row["schema_valid"]) for row in completed)
    metric_rows = [row["_metrics"] for row in completed]
    exact_correct = sum(row["field_exact_correct"] for row in metric_rows)
    exact_total = sum(row["field_exact_total"] for row in metric_rows)
    true_positive = sum(row["true_positive"] for row in metric_rows)
    false_positive = sum(row["false_positive"] for row in metric_rows)
    false_negative = sum(row["false_negative"] for row in metric_rows)
    unsupported_count = sum(row["unsupported_count"] for row in metric_rows)
    unsupported_opportunities = sum(row["unsupported_opportunities"] for row in metric_rows)
    document_id_correct = sum(bool(row["document_id_correct"]) for row in metric_rows)
    precision = safe_divide(true_positive, true_positive + false_positive)
    recall = safe_divide(true_positive, true_positive + false_negative)
    return {
        "intended_outputs": len(rows),
        "completed_outputs": len(completed),
        "raw_json_parse_count": parse_valid,
        "raw_json_parse_rate": safe_divide(parse_valid, len(completed)),
        "schema_valid_json_count": schema_valid,
        "schema_valid_json_rate": safe_divide(schema_valid, len(completed)),
        "document_id_correct_count": document_id_correct,
        "document_id_accuracy": safe_divide(document_id_correct, len(completed)),
        "field_exact_correct": exact_correct,
        "field_exact_total": exact_total,
        "field_exact_match": safe_divide(exact_correct, exact_total),
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "micro_precision": precision,
        "micro_recall": recall,
        "micro_f1": safe_divide(2 * precision * recall, precision + recall),
        "unsupported_count": unsupported_count,
        "unsupported_opportunities": unsupported_opportunities,
        "unsupported_field_rate": safe_divide(unsupported_count, unsupported_opportunities),
        "complete_report_count": sum(bool(row["complete_report"]) for row in metric_rows),
        "complete_report_accuracy": safe_divide(
            sum(bool(row["complete_report"]) for row in metric_rows),
            len(completed),
        ),
    }


def grouped_metrics(
    rows: Sequence[Mapping[str, Any]], key_fields: Sequence[str]
) -> List[Dict[str, Any]]:
    groups: Dict[Tuple[str, ...], List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        key = tuple(str(row[field]) for field in key_fields)
        groups[key].append(row)
    result = []
    for key in sorted(groups):
        result.append(
            {
                **dict(zip(key_fields, key)),
                **aggregate(groups[key]),
            }
        )
    return result


def per_field_metrics(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    stats = {
        path: {
            "exact_correct": 0,
            "total": 0,
            "true_positive": 0,
            "false_positive": 0,
            "false_negative": 0,
            "unsupported_count": 0,
            "unsupported_opportunities": 0,
        }
        for path in PRIMARY_FIELD_PATHS
    }
    for row in rows:
        if not row["raw_present"]:
            continue
        truth_fields = select_primary_fields(row["_truth"])
        prediction = None
        raw_path = raw_output_path(row["model_id"], row["condition"], row["document_id"])
        prediction, _ = parse_file(raw_path)
        prediction_fields = select_primary_fields(prediction) if prediction is not None else {}
        for path in PRIMARY_FIELD_PATHS:
            expected = truth_fields[path]
            observed = prediction_fields.get(path)
            item = stats[path]
            item["total"] += 1
            if prediction is not None and expected == observed:
                item["exact_correct"] += 1
            if expected is not None and prediction is not None and expected == observed:
                item["true_positive"] += 1
            elif prediction is not None and observed is not None:
                item["false_positive"] += 1
                if expected is not None:
                    item["false_negative"] += 1
            elif expected is not None:
                item["false_negative"] += 1
            if expected is None:
                item["unsupported_opportunities"] += 1
                if prediction is not None and observed is not None:
                    item["unsupported_count"] += 1

    output: Dict[str, Any] = {}
    for path, item in stats.items():
        precision = safe_divide(
            item["true_positive"], item["true_positive"] + item["false_positive"]
        )
        recall = safe_divide(item["true_positive"], item["true_positive"] + item["false_negative"])
        output[path] = {
            **item,
            "exact_match": safe_divide(item["exact_correct"], item["total"]),
            "precision": precision,
            "recall": recall,
            "f1": safe_divide(2 * precision * recall, precision + recall),
            "unsupported_field_rate": safe_divide(
                item["unsupported_count"], item["unsupported_opportunities"]
            ),
        }
    return output


def update_manual_review(rows: Sequence[Mapping[str, Any]]) -> None:
    path = ROOT / "results" / "manual_review.csv"
    existing = {row_key(row): row for row in load_csv(path)}
    output = []
    for row in rows:
        key = row_key(row)
        review = existing.get(
            key,
            {
                "model_id": row["model_id"],
                "semantic_case_id": row["semantic_case_id"],
                "document_id": row["document_id"],
                "template_id": row["template_id"],
                "condition": row["condition"],
                "review_status": "pending",
            },
        )
        if row["raw_present"]:
            metrics = row["_metrics"]
            review["parse_valid"] = row["parse_valid"]
            review["schema_valid"] = row["schema_valid"]
            review["document_id_correct"] = metrics["document_id_correct"]
            review["mismatched_fields"] = ";".join(metrics["mismatched_fields"])
            review["unsupported_fields"] = ";".join(metrics["unsupported_fields"])
        output.append(review)
    write_csv(path, REVIEW_FIELDS, output)


def update_run_manifest(status: str, completed: int) -> None:
    path = ROOT / "results" / "run_manifest.json"
    payload = read_json(path)
    records = (
        sorted(
            str(path.relative_to(ROOT))
            for path in (ROOT / "results" / "run_records").glob("**/*.json")
        )
        if (ROOT / "results" / "run_records").exists()
        else []
    )
    payload["status"] = status
    payload["completed_output_count"] = completed
    payload["run_record_paths"] = records
    payload["last_evaluated_utc"] = datetime.now(timezone.utc).isoformat()
    write_json(path, payload)


def evaluate() -> Dict[str, Any]:
    rows = evaluate_rows()
    pilot_rows = [pilot_row(row) for row in rows]
    write_csv(ROOT / "results" / "pilot_table.csv", PILOT_FIELDS, pilot_rows)
    update_manual_review(rows)

    completed = sum(bool(row["raw_present"]) for row in rows)
    if completed == 0:
        status = "NOT_RUN"
    elif completed < len(rows):
        status = "INCOMPLETE"
    else:
        status = "COMPLETE"

    metrics = {
        "status": status,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "intended_output_count": len(rows),
        "completed_output_count": completed,
        "overall": aggregate(rows),
        "overall_by_model": grouped_metrics(rows, ("model_id",)),
        "by_model_and_condition": grouped_metrics(rows, ("model_id", "condition")),
        "by_model_and_template": grouped_metrics(rows, ("model_id", "template_id")),
        "by_model_template_condition": grouped_metrics(
            rows, ("model_id", "template_id", "condition")
        ),
        "per_field": per_field_metrics(rows),
    }
    write_json(ROOT / "results" / "metrics.json", metrics)
    update_run_manifest(status, completed)
    print(f"Evaluation status: {status} ({completed}/{len(rows)} outputs present).")
    return metrics


if __name__ == "__main__":
    evaluate()
