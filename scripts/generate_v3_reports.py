#!/usr/bin/env python3
"""Generate the fresh, versioned v3 synthetic report corpus."""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from hashlib import sha256
from importlib.metadata import version as distribution_version
from pathlib import Path
from typing import Any, Dict, List

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

DATASET_VERSION = "pilot-v3"
GENERATOR_VERSION = "3.0.0"
TEMPLATE_IDS = ("A", "B")
CONDITIONS = ("clean", "ocr_degraded")
DEVELOPMENT_CASE_IDS = ("MEL-101", "MEL-102", "MEL-103")
FORMAL_CASE_IDS = ("MEL-104", "MEL-105", "MEL-106", "MEL-107", "MEL-108")
ALL_CASE_IDS = (*DEVELOPMENT_CASE_IDS, *FORMAL_CASE_IDS)


def relative(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def load_v3_cases(root: Path = ROOT) -> List[Dict[str, str]]:
    case_path = root / "data" / "v3" / "cases.csv"
    with case_path.open(newline="", encoding="utf-8") as handle:
        cases = list(csv.DictReader(handle))
    if any(None in row for row in cases):
        raise ValueError(f"{relative(case_path, root)} contains extra columns")

    ids = tuple(case["case_id"] for case in cases)
    if ids != ALL_CASE_IDS:
        raise ValueError(f"V3 case IDs must be exactly {ALL_CASE_IDS!r}, found {ids!r}")
    if len(set(ids)) != len(ids):
        raise ValueError("V3 case IDs must be unique")

    expected_splits = {
        **{case_id: "development" for case_id in DEVELOPMENT_CASE_IDS},
        **{case_id: "formal" for case_id in FORMAL_CASE_IDS},
    }
    observed_splits = {case["case_id"]: case["split"] for case in cases}
    if observed_splits != expected_splits:
        raise ValueError(
            f"V3 split assignments must be {expected_splits!r}, found {observed_splits!r}"
        )
    return cases


def degradation_parameters(doc_id: str) -> Dict[str, Any]:
    digest = sha256(f"{doc_id}|degradation-v3".encode("utf-8")).digest()
    sign = -1 if digest[0] % 2 else 1
    return {
        "perturbation_id": "scan-v3",
        "rotation_degrees": round(sign * (0.35 + (digest[1] / 255) * 0.30), 3),
        "blur_radius": round(0.55 + (digest[2] / 255) * 0.30, 3),
        "contrast_factor": round(0.82 + (digest[3] / 255) * 0.08, 3),
        "jpeg_quality": 58 + digest[4] % 15,
        "seed": int.from_bytes(digest[5:13], "big"),
    }


def output_paths(root: Path, doc_id: str) -> Dict[str, Path]:
    return {
        "clean_pdf": root / "output" / "v3" / "pdf" / "clean" / f"{doc_id}.pdf",
        "degraded_pdf": (root / "output" / "v3" / "pdf" / "ocr_degraded" / f"{doc_id}.pdf"),
        "clean_image": (root / "output" / "v3" / "rendered" / "clean" / f"{doc_id}.png"),
        "degraded_image": (root / "output" / "v3" / "rendered" / "ocr_degraded" / f"{doc_id}.png"),
        "source_text": root / "data" / "v3" / "source_text" / f"{doc_id}.txt",
        "ground_truth": root / "data" / "v3" / "ground_truth" / f"{doc_id}.json",
    }


def generate(root: Path = ROOT) -> List[Dict[str, Any]]:
    cases = load_v3_cases(root)
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
                "semantic_case_id": case["case_id"],
                "document_id": doc_id,
                "template_id": template_id,
                "dpi": DPI,
                "renderer": f"pypdfium2-{distribution_version('pypdfium2')}",
                "source_row_sha256": row_hash(case),
                "source_text_path": relative(paths["source_text"], root),
                "ground_truth_path": relative(paths["ground_truth"], root),
            }
            manifest_rows.append(
                {
                    **common,
                    "condition": "clean",
                    "perturbation_id": "none",
                    "rotation_degrees": 0,
                    "blur_radius": 0,
                    "contrast_factor": 1,
                    "jpeg_quality": "",
                    "pdf_path": relative(paths["clean_pdf"], root),
                    "image_path": relative(paths["clean_image"], root),
                    "pdf_sha256": sha256_file(paths["clean_pdf"]),
                    "image_sha256": sha256_file(paths["clean_image"]),
                }
            )
            manifest_rows.append(
                {
                    **common,
                    "condition": "ocr_degraded",
                    "perturbation_id": params["perturbation_id"],
                    "rotation_degrees": params["rotation_degrees"],
                    "blur_radius": params["blur_radius"],
                    "contrast_factor": params["contrast_factor"],
                    "jpeg_quality": params["jpeg_quality"],
                    "pdf_path": relative(paths["degraded_pdf"], root),
                    "image_path": relative(paths["degraded_image"], root),
                    "pdf_sha256": sha256_file(paths["degraded_pdf"]),
                    "image_sha256": sha256_file(paths["degraded_image"]),
                }
            )

    manifest_fields = (
        "dataset_version",
        "generator_version",
        "split",
        "semantic_case_id",
        "document_id",
        "template_id",
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
    write_csv(
        root / "data" / "v3" / "report_manifest.csv",
        manifest_fields,
        manifest_rows,
    )

    split_counts = Counter(row["split"] for row in manifest_rows)
    print(
        "Generated v3 corpus: "
        f"{len(cases)} cases, {len(cases) * len(TEMPLATE_IDS)} report layouts, "
        f"{len(manifest_rows)} rendered inputs "
        f"({split_counts['development']} development, "
        f"{split_counts['formal']} formal)."
    )
    return manifest_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=ROOT,
        help="Repository root; primarily useful for isolated tests.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    generate(args.root.resolve())
