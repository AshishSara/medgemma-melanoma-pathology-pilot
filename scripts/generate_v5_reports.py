#!/usr/bin/env python3
"""Generate the fresh v5 confirmatory corpus without touching earlier versions."""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from hashlib import sha256
from importlib.metadata import version as distribution_version
from pathlib import Path
from typing import Any, Dict, List, Mapping

from generate_reports import (
    DPI,
    degrade_image,
    generate_clean_pdf,
    image_only_pdf,
    render_pdf,
    row_hash,
)
from pilot_utils import (
    ROOT,
    document_id,
    ground_truth_from_case,
    sha256_file,
    write_csv,
    write_json,
)

DATASET_VERSION = "pilot-v5"
GENERATOR_VERSION = "5.0.0"
TEMPLATE_IDS = ("A", "B")
CONDITIONS = ("clean", "ocr_degraded")
DEVELOPMENT_CASE_IDS = ("MEL-190",)
FORMAL_CASE_IDS = ("MEL-201", "MEL-202", "MEL-203", "MEL-204", "MEL-205")
ALL_CASE_IDS = (*DEVELOPMENT_CASE_IDS, *FORMAL_CASE_IDS)
EXPECTED_REPORT_DATES = {
    "MEL-190": "2026-08-21",
    "MEL-201": "2026-08-22",
    "MEL-202": "2026-08-23",
    "MEL-203": "2026-08-24",
    "MEL-204": "2026-08-25",
    "MEL-205": "2026-08-26",
}
PRIOR_CASE_PATHS = (
    Path("data/cases.csv"),
    Path("data/v3/cases.csv"),
    Path("data/v4/cases.csv"),
)


def relative(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def load_v5_cases(root: Path = ROOT) -> List[Dict[str, str]]:
    case_path = root / "data" / "v5" / "cases.csv"
    with case_path.open(newline="", encoding="utf-8") as handle:
        cases = list(csv.DictReader(handle))
    if any(None in row for row in cases):
        raise ValueError(f"{relative(case_path, root)} contains extra columns")
    observed_ids = tuple(row["case_id"] for row in cases)
    if observed_ids != ALL_CASE_IDS:
        raise ValueError(f"V5 case IDs must be exactly {ALL_CASE_IDS!r}, found {observed_ids!r}")
    expected_splits = {
        **{case_id: "development" for case_id in DEVELOPMENT_CASE_IDS},
        **{case_id: "formal" for case_id in FORMAL_CASE_IDS},
    }
    if {row["case_id"]: row["split"] for row in cases} != expected_splits:
        raise ValueError("V5 split assignments do not match the prespecified case matrix")
    if any(row.get("case_origin") != "fresh-v5" for row in cases):
        raise ValueError("Every v5 case must declare case_origin=fresh-v5")
    return cases


def degradation_parameters(doc_id: str) -> Dict[str, Any]:
    digest = sha256(f"{doc_id}|degradation-v5".encode("utf-8")).digest()
    sign = -1 if digest[0] % 2 else 1
    return {
        "perturbation_id": "scan-v5",
        "rotation_degrees": round(sign * (0.35 + (digest[1] / 255) * 0.30), 3),
        "blur_radius": round(0.55 + (digest[2] / 255) * 0.30, 3),
        "contrast_factor": round(0.82 + (digest[3] / 255) * 0.08, 3),
        "jpeg_quality": 58 + digest[4] % 15,
        "seed": int.from_bytes(digest[5:13], "big"),
    }


def output_paths(root: Path, doc_id: str) -> Dict[str, Path]:
    return {
        "clean_pdf": root / "output" / "v5" / "pdf" / "clean" / f"{doc_id}.pdf",
        "degraded_pdf": root / "output" / "v5" / "pdf" / "ocr_degraded" / f"{doc_id}.pdf",
        "clean_image": root / "output" / "v5" / "rendered" / "clean" / f"{doc_id}.png",
        "degraded_image": (root / "output" / "v5" / "rendered" / "ocr_degraded" / f"{doc_id}.png"),
        "source_text": root / "data" / "v5" / "source_text" / f"{doc_id}.txt",
        "ground_truth": root / "data" / "v5" / "ground_truth" / f"{doc_id}.json",
    }


def semantic_fingerprint(case: Mapping[str, str]) -> tuple[str, ...]:
    fields = (
        "specimen_site",
        "laterality",
        "diagnosis",
        "breslow_thickness_mm",
        "breslow_qualifier",
        "ulceration",
        "mitotic_rate_per_mm2",
        "mitotic_qualifier",
        "invasive_peripheral",
        "invasive_deep",
        "in_situ_peripheral",
        "in_situ_deep",
        "pT",
        "pN",
        "pM",
        "stage_group",
    )
    return tuple(case[field] for field in fields)


def validate_freshness(cases: List[Dict[str, str]], root: Path = ROOT) -> None:
    """Apply the prospectively fixed, exact-match freshness rule."""

    prior_cases: List[Dict[str, str]] = []
    for relative_path in PRIOR_CASE_PATHS:
        with (root / relative_path).open(newline="", encoding="utf-8") as handle:
            prior_cases.extend(csv.DictReader(handle))

    prior_fingerprints = {semantic_fingerprint(case): case["case_id"] for case in prior_cases}
    collisions = [
        (case["case_id"], prior_fingerprints[semantic_fingerprint(case)])
        for case in cases
        if semantic_fingerprint(case) in prior_fingerprints
    ]
    if collisions:
        raise ValueError(f"V5 semantic fingerprint overlaps prior corpus: {collisions!r}")

    prior_ids = {case["case_id"] for case in prior_cases}
    overlapping_ids = sorted({case["case_id"] for case in cases} & prior_ids)
    if overlapping_ids:
        raise ValueError(f"V5 case IDs overlap prior corpus: {overlapping_ids!r}")

    prior_dates = {case["report_date"] for case in prior_cases}
    overlapping_dates = sorted({case["report_date"] for case in cases} & prior_dates)
    if overlapping_dates:
        raise ValueError(f"V5 report dates overlap prior corpus: {overlapping_dates!r}")

    observed_dates = {case["case_id"]: case["report_date"] for case in cases}
    if observed_dates != EXPECTED_REPORT_DATES:
        raise ValueError(
            f"V5 report dates must be exactly {EXPECTED_REPORT_DATES!r}, found {observed_dates!r}"
        )


def generate(root: Path = ROOT) -> List[Dict[str, Any]]:
    cases = load_v5_cases(root)
    fingerprints = [semantic_fingerprint(case) for case in cases]
    if len(set(fingerprints)) != len(fingerprints):
        raise ValueError("V5 semantic fact vectors must be unique")
    validate_freshness(cases, root)

    manifest_rows: List[Dict[str, Any]] = []
    for case in cases:
        for template_id in TEMPLATE_IDS:
            doc_id = document_id(case["case_id"], template_id)
            paths = output_paths(root, doc_id)
            source_text = generate_clean_pdf(paths["clean_pdf"], doc_id, template_id, case)
            paths["source_text"].parent.mkdir(parents=True, exist_ok=True)
            paths["source_text"].write_text(source_text, encoding="utf-8")
            write_json(paths["ground_truth"], ground_truth_from_case(case, template_id))

            clean_image = render_pdf(paths["clean_pdf"])
            paths["clean_image"].parent.mkdir(parents=True, exist_ok=True)
            clean_image.save(paths["clean_image"], format="PNG", optimize=False)

            params = degradation_parameters(doc_id)
            degraded_image = degrade_image(clean_image, params)
            paths["degraded_image"].parent.mkdir(parents=True, exist_ok=True)
            degraded_image.save(paths["degraded_image"], format="PNG", optimize=False)
            image_only_pdf(paths["degraded_pdf"], degraded_image, doc_id)
            clean_image.close()
            degraded_image.close()

            common = {
                "dataset_version": DATASET_VERSION,
                "generator_version": GENERATOR_VERSION,
                "split": case["split"],
                "case_origin": case["case_origin"],
                "semantic_case_id": case["case_id"],
                "document_id": doc_id,
                "template_id": template_id,
                "layout_scope": "known_fixed_layout_family",
                "dpi": DPI,
                "renderer": f"pypdfium2-{distribution_version('pypdfium2')}",
                "source_row_sha256": row_hash(case),
                "source_text_path": relative(paths["source_text"], root),
                "ground_truth_path": relative(paths["ground_truth"], root),
            }
            for condition, pdf_key, image_key, condition_values in (
                (
                    "clean",
                    "clean_pdf",
                    "clean_image",
                    {
                        "perturbation_id": "none",
                        "rotation_degrees": 0,
                        "blur_radius": 0,
                        "contrast_factor": 1,
                        "jpeg_quality": "",
                    },
                ),
                (
                    "ocr_degraded",
                    "degraded_pdf",
                    "degraded_image",
                    {
                        "perturbation_id": params["perturbation_id"],
                        "rotation_degrees": params["rotation_degrees"],
                        "blur_radius": params["blur_radius"],
                        "contrast_factor": params["contrast_factor"],
                        "jpeg_quality": params["jpeg_quality"],
                    },
                ),
            ):
                manifest_rows.append(
                    {
                        **common,
                        "condition": condition,
                        **condition_values,
                        "pdf_path": relative(paths[pdf_key], root),
                        "image_path": relative(paths[image_key], root),
                        "pdf_sha256": sha256_file(paths[pdf_key]),
                        "image_sha256": sha256_file(paths[image_key]),
                    }
                )

    fields = (
        "dataset_version",
        "generator_version",
        "split",
        "case_origin",
        "semantic_case_id",
        "document_id",
        "template_id",
        "layout_scope",
        "condition",
        "perturbation_id",
        "dpi",
        "renderer",
        "rotation_degrees",
        "blur_radius",
        "contrast_factor",
        "jpeg_quality",
        "source_row_sha256",
        "source_text_path",
        "ground_truth_path",
        "pdf_path",
        "image_path",
        "pdf_sha256",
        "image_sha256",
    )
    write_csv(root / "data" / "v5" / "report_manifest.csv", fields, manifest_rows)
    counts = Counter(row["split"] for row in manifest_rows)
    print(
        "Generated v5 corpus: "
        f"{len(cases)} cases, {len(cases) * len(TEMPLATE_IDS)} report layouts, "
        f"{len(manifest_rows)} rendered inputs "
        f"({counts['development']} development, {counts['formal']} formal)."
    )
    return manifest_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    return parser.parse_args()


if __name__ == "__main__":
    generate(parse_args().root.resolve())
