#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import re
from typing import Any, Dict

from pilot_utils import ROOT, read_json

PLACEHOLDER = re.compile(
    r"\[PILOT RESULT SENTENCE - replace only after all 64 outputs are scored "
    r"and manually verified\.\]"
)


def model_metrics(metrics: Dict[str, Any], model_id: str) -> Dict[str, Any]:
    for row in metrics["overall_by_model"]:
        if row["model_id"] == model_id:
            return row
    raise SystemExit(f"Metrics missing for {model_id}")


def verified_review_count() -> tuple[int, int]:
    path = ROOT / "results" / "manual_review.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return sum(row["review_status"] == "verified" for row in rows), len(rows)


def finalize(repository_url: str) -> str:
    metrics = read_json(ROOT / "results" / "metrics.json")
    if metrics.get("status") != "COMPLETE" or metrics.get("completed_output_count") != 64:
        raise SystemExit("Cannot finalize outreach until all 64 outputs are complete.")
    verified, total = verified_review_count()
    if (verified, total) != (64, 64):
        raise SystemExit(
            f"Cannot finalize outreach until manual review is 64/64; found {verified}/{total}."
        )

    row = model_metrics(metrics, "google/medgemma-1.5-4b-it")
    sentence = (
        "In a frozen synthetic feasibility pilot of 16 base reports evaluated "
        "as 32 paired clean/degraded document conditions, MedGemma 1.5 4B "
        f"produced schema-valid JSON for {row['schema_valid_json_count']}/32 "
        f"documents, achieved {row['field_exact_match'] * 100:.1f}% atomic-field "
        "exact match, and had an unsupported-field rate of "
        f"{row['unsupported_field_rate'] * 100:.1f}% "
        f"({row['unsupported_count']}/{row['unsupported_opportunities']} "
        f"null-field opportunities); materials: {repository_url}"
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
