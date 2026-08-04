#!/usr/bin/env python3
from __future__ import annotations

import csv
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from output_parser import parse_file
from pilot_utils import (
    CONDITIONS,
    DEVELOPMENT_CASE_IDS,
    FORMAL_BASE_REPORT_COUNT,
    FORMAL_DOCUMENT_CONDITION_COUNT,
    FORMAL_DTYPE,
    FORMAL_MAX_NEW_TOKENS,
    FORMAL_OUTPUT_COUNT,
    FORMAL_RESPONSE_MODE,
    FORMAL_SEMANTIC_CASE_COUNT,
    HASH_BOUND_REVIEW_STATUSES,
    JSON_ASSISTANT_PREFIX,
    MODEL_IDS,
    MODEL_REVISIONS,
    PRIMARY_FIELD_PATHS,
    PROTOCOL_VERSION,
    REVIEW_STATUS_PENDING,
    ROOT,
    compare_prediction,
    flatten_json,
    formal_manifest_rows,
    read_json,
    relative,
    safe_divide,
    select_primary_fields,
    sha256_bytes,
    sha256_file,
    slugify_model_id,
    values_exactly_equal,
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
    "provenance_valid",
    "provenance_errors",
    "raw_output_sha256",
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
    "raw_output_sha256",
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


def result_path(
    category: str,
    model_id: str,
    condition: str,
    document_id: str,
    extension: str,
) -> Path:
    return (
        ROOT
        / "results"
        / category
        / slugify_model_id(model_id)
        / condition
        / f"{document_id}.{extension}"
    )


def raw_output_path(model_id: str, condition: str, document_id: str) -> Path:
    return result_path("raw", model_id, condition, document_id, "txt")


def continuation_path(model_id: str, condition: str, document_id: str) -> Path:
    return result_path("generated_continuations", model_id, condition, document_id, "txt")


def run_record_path(model_id: str, condition: str, document_id: str) -> Path:
    return result_path("run_records", model_id, condition, document_id, "json")


def normalized_path(model_id: str, condition: str, document_id: str) -> Path:
    return result_path("normalized", model_id, condition, document_id, "json")


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


def provenance_errors(
    model_id: str,
    item: Mapping[str, str],
    raw_path: Path,
    continuation: Path,
    record_path: Path,
) -> tuple[List[str], Dict[str, Any] | None]:
    if not record_path.exists():
        return [f"missing run record: {relative(record_path)}"], None
    if not continuation.exists():
        return [f"missing generated continuation: {relative(continuation)}"], None
    try:
        record = read_json(record_path)
    except (OSError, ValueError) as exc:
        return [f"unreadable run record: {exc}"], None

    raw_sha = sha256_file(raw_path)
    continuation_sha = sha256_file(continuation)
    prompt_sha = sha256_file(ROOT / "prompts" / "extraction_prompt.txt")
    schema_sha = sha256_file(ROOT / "schema" / "extraction.schema.json")
    prefix_sha = sha256_bytes(JSON_ASSISTANT_PREFIX.encode("utf-8"))
    expected = {
        "protocol_version": PROTOCOL_VERSION,
        "scope": "formal",
        "model_id": model_id,
        "backend": "transformers",
        "requested_revision": MODEL_REVISIONS[model_id],
        "resolved_revision": MODEL_REVISIONS[model_id],
        "response_mode": FORMAL_RESPONSE_MODE,
        "assistant_prefix": JSON_ASSISTANT_PREFIX,
        "assistant_prefix_sha256": prefix_sha,
        "max_new_tokens": FORMAL_MAX_NEW_TOKENS,
        "do_sample": False,
        "requested_dtype": FORMAL_DTYPE,
        "dtype": f"torch.{FORMAL_DTYPE}",
        "document_id": item["document_id"],
        "semantic_case_id": item["semantic_case_id"],
        "template_id": item["template_id"],
        "condition": item["condition"],
        "image_path": item["image_path"],
        "image_sha256": item["image_sha256"],
        "prompt_sha256": prompt_sha,
        "schema_sha256": schema_sha,
        "raw_output_path": relative(raw_path),
        "raw_output_bytes": len(raw_path.read_bytes()),
        "raw_output_sha256": raw_sha,
        "generated_continuation_path": relative(continuation),
        "generated_continuation_bytes": len(continuation.read_bytes()),
        "generated_continuation_sha256": continuation_sha,
    }
    errors = [
        f"{key}: expected {value!r}, found {record.get(key)!r}"
        for key, value in expected.items()
        if record.get(key) != value
    ]
    if not record.get("repository_commit"):
        errors.append("repository_commit is missing")
    return errors, record


def evaluate_rows() -> List[Dict[str, Any]]:
    full_manifest = load_csv(ROOT / "data" / "report_manifest.csv")
    manifest = formal_manifest_rows(full_manifest)
    rows: List[Dict[str, Any]] = []
    for model_id in MODEL_IDS:
        for item in manifest:
            truth = read_json(ROOT / item["ground_truth_path"])
            raw_path = raw_output_path(model_id, item["condition"], item["document_id"])
            continuation = continuation_path(model_id, item["condition"], item["document_id"])
            record_path = run_record_path(model_id, item["condition"], item["document_id"])
            base = {
                "model_id": model_id,
                "semantic_case_id": item["semantic_case_id"],
                "document_id": item["document_id"],
                "template_id": item["template_id"],
                "condition": item["condition"],
                "_truth": truth,
                "_record_path": record_path,
            }
            if not raw_path.exists():
                rows.append(
                    {
                        **base,
                        "run_status": "NOT_RUN",
                        "raw_present": False,
                        "provenance_valid": None,
                        "provenance_errors": "",
                        "raw_output_sha256": "",
                        "parse_valid": None,
                        "schema_valid": None,
                        "_metrics": None,
                        "_parse_metadata": None,
                    }
                )
                continue

            errors, record = provenance_errors(
                model_id,
                item,
                raw_path,
                continuation,
                record_path,
            )
            raw_sha = sha256_file(raw_path)
            if errors:
                rows.append(
                    {
                        **base,
                        "run_status": "INVALID_PROVENANCE",
                        "raw_present": True,
                        "provenance_valid": False,
                        "provenance_errors": "; ".join(errors),
                        "raw_output_sha256": raw_sha,
                        "parse_valid": None,
                        "schema_valid": None,
                        "_metrics": None,
                        "_parse_metadata": None,
                        "_record": record,
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
            normalized = normalized_path(model_id, item["condition"], item["document_id"])
            if prediction is None and normalized.exists():
                raise ValueError(f"Parse-invalid output has stale normalized JSON: {normalized}")
            if prediction is not None and not normalized.exists():
                raise ValueError(f"Parse-valid output lacks normalized JSON: {normalized}")
            rows.append(
                {
                    **base,
                    "run_status": "COMPLETED",
                    "raw_present": True,
                    "provenance_valid": True,
                    "provenance_errors": "",
                    "raw_output_sha256": raw_sha,
                    "parse_valid": parse_metadata["parse_valid"],
                    "schema_valid": parse_metadata["schema_valid"],
                    "_metrics": metrics,
                    "_parse_metadata": parse_metadata,
                    "_record": record,
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


def completed_rows(rows: Sequence[Mapping[str, Any]]) -> List[Mapping[str, Any]]:
    return [row for row in rows if row["run_status"] == "COMPLETED"]


def aggregate(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    completed = completed_rows(rows)
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
        "invalid_provenance_outputs": sum(
            row["run_status"] == "INVALID_PROVENANCE" for row in rows
        ),
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
    return [
        {
            **dict(zip(key_fields, key)),
            **aggregate(groups[key]),
        }
        for key in sorted(groups)
    ]


_MISSING = object()


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
    for row in completed_rows(rows):
        truth_fields = select_primary_fields(row["_truth"])
        raw_path = raw_output_path(row["model_id"], row["condition"], row["document_id"])
        prediction, _ = parse_file(raw_path)
        flattened_prediction = flatten_json(prediction) if prediction is not None else {}
        for path in PRIMARY_FIELD_PATHS:
            expected = truth_fields[path]
            observed = flattened_prediction.get(path, _MISSING)
            item = stats[path]
            item["total"] += 1
            exact = prediction is not None and values_exactly_equal(expected, observed)
            if exact:
                item["exact_correct"] += 1
            if expected is not None and exact:
                item["true_positive"] += 1
            elif observed is not _MISSING and observed is not None:
                item["false_positive"] += 1
                if expected is not None:
                    item["false_negative"] += 1
            elif expected is not None:
                item["false_negative"] += 1
            if expected is None:
                item["unsupported_opportunities"] += 1
                if observed is not _MISSING and observed is not None:
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


def paired_clean_to_degraded(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    output: List[Dict[str, Any]] = []
    for model_id in MODEL_IDS:
        model_rows = [row for row in rows if row["model_id"] == model_id]
        pairs: Dict[str, Dict[str, Mapping[str, Any]]] = defaultdict(dict)
        for row in model_rows:
            pairs[row["document_id"]][row["condition"]] = row
        completed_pairs = []
        for pair in pairs.values():
            if set(pair) == set(CONDITIONS) and all(
                row["run_status"] == "COMPLETED" for row in pair.values()
            ):
                completed_pairs.append(pair)

        deltas = [
            pair["ocr_degraded"]["_metrics"]["field_exact_match"]
            - pair["clean"]["_metrics"]["field_exact_match"]
            for pair in completed_pairs
        ]
        output.append(
            {
                "model_id": model_id,
                "intended_pairs": len(pairs),
                "completed_pairs": len(completed_pairs),
                "mean_field_exact_match_delta": safe_divide(sum(deltas), len(deltas)),
                "degraded_improved_pairs": sum(delta > 0 for delta in deltas),
                "degraded_tied_pairs": sum(delta == 0 for delta in deltas),
                "degraded_worse_pairs": sum(delta < 0 for delta in deltas),
                "schema_valid_gain_pairs": sum(
                    not pair["clean"]["schema_valid"] and pair["ocr_degraded"]["schema_valid"]
                    for pair in completed_pairs
                ),
                "schema_valid_loss_pairs": sum(
                    pair["clean"]["schema_valid"] and not pair["ocr_degraded"]["schema_valid"]
                    for pair in completed_pairs
                ),
                "complete_report_gain_pairs": sum(
                    not pair["clean"]["_metrics"]["complete_report"]
                    and pair["ocr_degraded"]["_metrics"]["complete_report"]
                    for pair in completed_pairs
                ),
                "complete_report_loss_pairs": sum(
                    pair["clean"]["_metrics"]["complete_report"]
                    and not pair["ocr_degraded"]["_metrics"]["complete_report"]
                    for pair in completed_pairs
                ),
            }
        )
    return output


def update_manual_review(rows: Sequence[Mapping[str, Any]]) -> None:
    path = ROOT / "results" / "manual_review.csv"
    existing = {row_key(row): row for row in load_csv(path)} if path.exists() else {}
    output = []
    for row in rows:
        key = row_key(row)
        review = existing.get(key, {}).copy()
        review.update(
            {
                "model_id": row["model_id"],
                "semantic_case_id": row["semantic_case_id"],
                "document_id": row["document_id"],
                "template_id": row["template_id"],
                "condition": row["condition"],
            }
        )
        review.setdefault("review_status", REVIEW_STATUS_PENDING)
        current_raw_sha = row["raw_output_sha256"]
        if (
            review.get("review_status") in HASH_BOUND_REVIEW_STATUSES
            and review.get("raw_output_sha256") != current_raw_sha
        ):
            review.update(
                {
                    "review_status": REVIEW_STATUS_PENDING,
                    "source_ambiguity": "",
                    "adjudication": "",
                    "reviewer": "",
                    "review_date": "",
                }
            )
        review["raw_output_sha256"] = current_raw_sha
        if row["run_status"] == "COMPLETED":
            metrics = row["_metrics"]
            review["parse_valid"] = row["parse_valid"]
            review["schema_valid"] = row["schema_valid"]
            review["document_id_correct"] = metrics["document_id_correct"]
            review["mismatched_fields"] = ";".join(metrics["mismatched_fields"])
            review["unsupported_fields"] = ";".join(metrics["unsupported_fields"])
        else:
            for field in (
                "parse_valid",
                "schema_valid",
                "document_id_correct",
                "mismatched_fields",
                "unsupported_fields",
            ):
                review[field] = ""
        output.append(review)
    write_csv(path, REVIEW_FIELDS, output)


def update_run_manifest(
    rows: Sequence[Mapping[str, Any]],
    status: str,
    completed: int,
) -> None:
    path = ROOT / "results" / "run_manifest.json"
    existing = read_json(path) if path.exists() else {}
    formal_case_ids = sorted({row["semantic_case_id"] for row in rows})
    records = sorted(relative(row["_record_path"]) for row in rows if row["_record_path"].exists())
    payload = {
        "dataset_version": existing.get("dataset_version", "pilot-v1"),
        "protocol_version": PROTOCOL_VERSION,
        "status": status,
        "models": list(MODEL_IDS),
        "model_revisions": MODEL_REVISIONS,
        "conditions": list(CONDITIONS),
        "development_case_ids": list(DEVELOPMENT_CASE_IDS),
        "formal_case_ids": formal_case_ids,
        "formal_semantic_case_count": FORMAL_SEMANTIC_CASE_COUNT,
        "formal_base_report_count": FORMAL_BASE_REPORT_COUNT,
        "formal_document_condition_count": FORMAL_DOCUMENT_CONDITION_COUNT,
        "intended_output_count": FORMAL_OUTPUT_COUNT,
        "completed_output_count": completed,
        "invalid_provenance_output_count": sum(
            row["run_status"] == "INVALID_PROVENANCE" for row in rows
        ),
        "response_mode": FORMAL_RESPONSE_MODE,
        "assistant_prefix": JSON_ASSISTANT_PREFIX,
        "assistant_prefix_sha256": sha256_bytes(JSON_ASSISTANT_PREFIX.encode("utf-8")),
        "dtype": FORMAL_DTYPE,
        "max_new_tokens": FORMAL_MAX_NEW_TOKENS,
        "do_sample": False,
        "prompt_sha256": sha256_file(ROOT / "prompts" / "extraction_prompt.txt"),
        "schema_sha256": sha256_file(ROOT / "schema" / "extraction.schema.json"),
        "run_record_paths": records,
        "last_evaluated_utc": datetime.now(timezone.utc).isoformat(),
    }
    write_json(path, payload)


def evaluate() -> Dict[str, Any]:
    rows = evaluate_rows()
    if len(rows) != FORMAL_OUTPUT_COUNT:
        raise ValueError(f"Expected {FORMAL_OUTPUT_COUNT} formal rows, found {len(rows)}")
    write_csv(
        ROOT / "results" / "pilot_table.csv",
        PILOT_FIELDS,
        [pilot_row(row) for row in rows],
    )
    update_manual_review(rows)

    completed = len(completed_rows(rows))
    invalid = sum(row["run_status"] == "INVALID_PROVENANCE" for row in rows)
    if invalid:
        status = "INVALID_PROVENANCE"
    elif completed == 0:
        status = "NOT_RUN"
    elif completed < len(rows):
        status = "INCOMPLETE"
    else:
        status = "COMPLETE"

    metrics = {
        "status": status,
        "protocol_version": PROTOCOL_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "intended_output_count": len(rows),
        "completed_output_count": completed,
        "invalid_provenance_output_count": invalid,
        "overall": aggregate(rows),
        "overall_by_model": grouped_metrics(rows, ("model_id",)),
        "by_model_and_condition": grouped_metrics(rows, ("model_id", "condition")),
        "by_model_and_template": grouped_metrics(rows, ("model_id", "template_id")),
        "by_model_template_condition": grouped_metrics(
            rows, ("model_id", "template_id", "condition")
        ),
        "paired_clean_to_degraded": paired_clean_to_degraded(rows),
        "per_field": per_field_metrics(rows),
    }
    write_json(ROOT / "results" / "metrics.json", metrics)
    update_run_manifest(rows, status, completed)
    print(f"Evaluation status: {status} ({completed}/{len(rows)} outputs complete).")
    return metrics


if __name__ == "__main__":
    evaluate()
