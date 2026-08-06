#!/usr/bin/env python3
"""Evaluate one immutable v3 development iteration."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from typing import Any, Dict, List

from evaluate_v3 import (
    _condition_metrics,
    _paired_clean_to_degraded,
    _per_field_metrics,
    _score_layer,
    _template_metrics,
)
from pilot_utils import ROOT, read_json, sha256_file, write_json
from run_v3_inference import (
    CANDIDATE_SCHEMA_PATH,
    CANONICAL_SCHEMA_PATH,
    MODEL_ID,
    MODEL_REVISION,
    PROTOCOL_VERSION,
    V3_MANIFEST,
    output_paths,
    parse_generated_json,
    verify_existing,
)

DEVELOPMENT_OUTPUT_COUNT = 12


def load_development_manifest() -> List[Dict[str, str]]:
    with V3_MANIFEST.open(newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle) if row["split"] == "development"]
    if len(rows) != DEVELOPMENT_OUTPUT_COUNT:
        raise ValueError(f"Expected {DEVELOPMENT_OUTPUT_COUNT} development rows, found {len(rows)}")
    return rows


def evaluate_iteration(
    iteration: int,
    change_note: str,
    selection_rule: str,
) -> Dict[str, Any]:
    rows = load_development_manifest()
    candidate_schema = read_json(CANDIDATE_SCHEMA_PATH)
    canonical_schema = read_json(CANONICAL_SCHEMA_PATH)
    preflight = []

    for row in rows:
        paths = output_paths(row, development_iteration=iteration)
        verify_existing(paths, row, iteration, None)
        candidate_raw = paths["candidate_raw"].read_text(encoding="utf-8")
        candidate = parse_generated_json(candidate_raw, candidate_schema)
        normalized = read_json(paths["normalized"])
        preflight.append(
            {
                "row": row,
                "candidate": candidate,
                "normalized": normalized,
            }
        )

    truth_by_document = {
        row["document_id"]: read_json(ROOT / row["ground_truth_path"]) for row in rows
    }
    candidate_overall, candidate_rows = _score_layer(
        preflight,
        truth_by_document,
        "candidate",
        canonical_schema,
    )
    final_overall, final_rows = _score_layer(
        preflight,
        truth_by_document,
        "normalized",
        canonical_schema,
    )
    base = ROOT / "results" / "v3" / "development" / f"iteration-{iteration}"
    metrics_path = base / "metrics.json"
    metrics = {
        "status": "DEVELOPMENT_ONLY",
        "protocol_version": PROTOCOL_VERSION,
        "iteration": iteration,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "output_count": len(rows),
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "candidate": {
            "overall": candidate_overall,
            "by_condition": _condition_metrics(candidate_rows),
            "by_template": _template_metrics(candidate_rows),
            "per_field": _per_field_metrics(candidate_rows),
            "paired_clean_to_degraded": _paired_clean_to_degraded(candidate_rows),
        },
        "final_pipeline": {
            "overall": final_overall,
            "by_condition": _condition_metrics(final_rows),
            "by_template": _template_metrics(final_rows),
            "per_field": _per_field_metrics(final_rows),
            "paired_clean_to_degraded": _paired_clean_to_degraded(final_rows),
        },
    }
    write_json(metrics_path, metrics)

    run_record_hashes = {
        f"{row['document_id']}:{row['condition']}": sha256_file(
            output_paths(row, development_iteration=iteration)["record"]
        )
        for row in rows
    }
    write_json(
        base / "iteration_record.json",
        {
            "status": "DEVELOPMENT_ONLY",
            "protocol_version": PROTOCOL_VERSION,
            "iteration": iteration,
            "change_note": change_note,
            "selection_rule": selection_rule,
            "manifest_sha256": sha256_file(V3_MANIFEST),
            "candidate_schema_sha256": sha256_file(CANDIDATE_SCHEMA_PATH),
            "canonical_schema_sha256": sha256_file(CANONICAL_SCHEMA_PATH),
            "metrics_path": metrics_path.relative_to(ROOT).as_posix(),
            "metrics_sha256": sha256_file(metrics_path),
            "run_record_sha256": run_record_hashes,
        },
    )
    print(
        f"Development iteration {iteration}: "
        f"final exact={final_overall['field_exact_match']:.1%}, "
        f"unsupported={final_overall['unsupported_field_rate']:.1%}."
    )
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iteration", type=int, choices=(1, 2), required=True)
    parser.add_argument("--change-note", required=True)
    parser.add_argument("--selection-rule", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate_iteration(args.iteration, args.change_note, args.selection_rule)
