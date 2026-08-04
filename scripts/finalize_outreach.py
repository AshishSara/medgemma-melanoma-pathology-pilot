#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import re
from typing import Any, Dict

from pilot_utils import (
    FORMAL_DOCUMENT_CONDITION_COUNT,
    FORMAL_OUTPUT_COUNT,
    PROTOCOL_VERSION,
    ROOT,
    read_json,
    sha256_file,
    slugify_model_id,
)

PLACEHOLDER = re.compile(
    r"\[PILOT RESULT SENTENCE - replace only after all 56 outputs are scored "
    r"and manually verified\.\]"
)


def model_metrics(metrics: Dict[str, Any], model_id: str) -> Dict[str, Any]:
    for row in metrics["overall_by_model"]:
        if row["model_id"] == model_id:
            return row
    raise SystemExit(f"Metrics missing for {model_id}")


def verified_review_count() -> tuple[int, int, int]:
    path = ROOT / "results" / "manual_review.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    verified = 0
    stale = 0
    for row in rows:
        if row["review_status"] != "verified":
            continue
        raw_path = (
            ROOT
            / "results"
            / "raw"
            / slugify_model_id(row["model_id"])
            / row["condition"]
            / f"{row['document_id']}.txt"
        )
        if raw_path.exists() and row.get("raw_output_sha256") == sha256_file(raw_path):
            verified += 1
        else:
            stale += 1
    return verified, len(rows), stale


def finalize(repository_url: str) -> str:
    metrics = read_json(ROOT / "results" / "metrics.json")
    if (
        metrics.get("protocol_version") != PROTOCOL_VERSION
        or metrics.get("status") != "COMPLETE"
        or metrics.get("completed_output_count") != FORMAL_OUTPUT_COUNT
    ):
        raise SystemExit(
            f"Cannot finalize outreach until all {FORMAL_OUTPUT_COUNT} held-out outputs "
            "are complete under the frozen protocol."
        )
    verified, total, stale = verified_review_count()
    if (verified, total, stale) != (FORMAL_OUTPUT_COUNT, FORMAL_OUTPUT_COUNT, 0):
        raise SystemExit(
            f"Cannot finalize outreach until manual review is "
            f"{FORMAL_OUTPUT_COUNT}/{FORMAL_OUTPUT_COUNT} with matching raw-output hashes; "
            f"found {verified}/{total} verified and {stale} stale."
        )

    row = model_metrics(metrics, "google/medgemma-1.5-4b-it")
    if row["intended_outputs"] != FORMAL_DOCUMENT_CONDITION_COUNT:
        raise SystemExit("MedGemma 1.5 metrics have an unexpected denominator.")
    sentence = (
        "In a held-out synthetic feasibility pilot (7 semantic cases; 14 "
        "report-layout documents; 28 clean/degraded rendered inputs forming "
        "14 matched pairs), MedGemma 1.5 4B IT, run with greedy BF16 decoding "
        "using a MEL-001-development-selected no-example prompt and a one-character "
        "JSON assistant prefix, produced strict parse-valid JSON in "
        f"{row['raw_json_parse_count']}/{row['intended_outputs']} outputs and "
        f"schema-valid JSON in {row['schema_valid_json_count']}/"
        f"{row['intended_outputs']}; field exact match was "
        f"{row['field_exact_correct']}/{row['field_exact_total']} "
        f"({row['field_exact_match'] * 100:.1f}%), non-null micro-F1 was "
        f"{row['micro_f1'] * 100:.1f}%, and unsupported non-null values occurred in "
        f"{row['unsupported_count']}/{row['unsupported_opportunities']} "
        "ground-truth-null opportunities ("
        f"{row['unsupported_field_rate'] * 100:.1f}% "
        f"unsupported-field rate); materials: {repository_url}"
    )

    draft_path = ROOT / "local-notes" / "Outreach Draft.md"
    draft = draft_path.read_text(encoding="utf-8")
    updated, count = PLACEHOLDER.subn(sentence, draft)
    if count != 1:
        raise SystemExit("Expected exactly one pilot-sentence placeholder.")
    draft_path.write_text(updated, encoding="utf-8")
    print(sentence)
    return sentence


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replace the outreach placeholder only after the full gate passes."
    )
    parser.add_argument("--repository-url", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    finalize(args.repository_url)
