#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import io
import json
import random
from hashlib import sha256
from importlib.metadata import version as distribution_version
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import pypdfium2 as pdfium
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter
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
    PROTOCOL_VERSION,
    REVIEW_STATUS_PENDING,
    ROOT,
    TEMPLATE_IDS,
    document_id,
    formal_manifest_rows,
    ground_truth_from_case,
    load_cases,
    null_if_blank,
    read_json,
    relative,
    sha256_bytes,
    sha256_file,
    sha256_text,
    write_csv,
    write_json,
)
from reportlab.lib.colors import HexColor, black
from reportlab.lib.pagesizes import letter
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.pdfgen import canvas

DPI = 160
DATASET_VERSION = "pilot-v1"
GENERATOR_VERSION = "1.0.0"
PAGE_WIDTH, PAGE_HEIGHT = letter


def title_case(value: str) -> str:
    return value[:1].upper() + value[1:]


def displayed_site(case: Mapping[str, str]) -> str:
    laterality = null_if_blank(case["laterality"])
    site = case["specimen_site"]
    if laterality == "midline":
        return f"midline {site}"
    if laterality:
        return f"{laterality} {site}"
    return site


def diagnosis_text(case: Mapping[str, str]) -> str:
    diagnosis = case["diagnosis"]
    histologic_type = case["histologic_type"]
    if diagnosis == "invasive_melanoma":
        return f"Invasive melanoma, {histologic_type}"
    if diagnosis == "melanoma_in_situ":
        return f"Melanoma in situ, {histologic_type}"
    if diagnosis == "residual_melanoma_in_situ_no_invasive":
        return (
            f"Residual melanoma in situ, {histologic_type}; "
            "no residual invasive melanoma identified"
        )
    raise ValueError(f"Unsupported diagnosis: {diagnosis}")


def breslow_text(case: Mapping[str, str]) -> str:
    value = null_if_blank(case["breslow_thickness_mm"])
    if value is None:
        return ""
    qualifier = null_if_blank(case["breslow_qualifier"])
    prefixes = {"exact": "", "at_least": "At least ", "approximate": "Approximately "}
    return f"{prefixes[qualifier]}{value} mm"


def mitotic_text(case: Mapping[str, str]) -> str:
    value = null_if_blank(case["mitotic_rate_per_mm2"])
    if value is None:
        return ""
    qualifier = null_if_blank(case["mitotic_qualifier"])
    prefix = "At least " if qualifier == "at_least" else ""
    return f"{prefix}{value} per mm2"


def display_status(value: str) -> str:
    return {
        "present": "Present",
        "not_identified": "Not identified",
        "cannot_be_assessed": "Cannot be assessed",
        "involved": "Involved",
        "not_involved": "Not involved",
    }[value]


def fact_rows(case: Mapping[str, str]) -> List[Tuple[str, str]]:
    rows: List[Tuple[str, str]] = [
        ("Procedure", title_case(case["procedure"])),
        ("Specimen site", title_case(displayed_site(case))),
        ("Diagnosis", diagnosis_text(case)),
    ]
    if breslow_text(case):
        rows.append(("Breslow thickness", breslow_text(case)))
    if null_if_blank(case["ulceration"]):
        rows.append(("Ulceration", display_status(case["ulceration"])))
    if mitotic_text(case):
        rows.append(("Mitotic rate", mitotic_text(case)))

    margin_labels = (
        ("Invasive melanoma - peripheral margin", "invasive_peripheral"),
        ("Invasive melanoma - deep margin", "invasive_deep"),
        ("Melanoma in situ - peripheral margin", "in_situ_peripheral"),
        ("Melanoma in situ - deep margin", "in_situ_deep"),
    )
    for label, key in margin_labels:
        value = null_if_blank(case[key])
        if value:
            rows.append((label, display_status(value)))

    for label, key in (
        ("Reported pT", "pT"),
        ("Reported pN", "pN"),
        ("Reported pM", "pM"),
        ("Reported stage group", "stage_group"),
    ):
        value = null_if_blank(case[key])
        if value:
            rows.append((label, value))
    return rows


def wrap_text(text: str, font_name: str, font_size: float, width: float) -> List[str]:
    words = text.split()
    if not words:
        return [""]
    lines: List[str] = []
    current = words[0]
    for word in words[1:]:
        candidate = f"{current} {word}"
        if stringWidth(candidate, font_name, font_size) <= width:
            current = candidate
        else:
            lines.append(current)
            current = word
    lines.append(current)
    return lines


def draw_common_header(
    pdf: canvas.Canvas, doc_id: str, case: Mapping[str, str], accent: HexColor
) -> float:
    pdf.setFillColor(HexColor("#F5F7FA"))
    pdf.rect(0, PAGE_HEIGHT - 102, PAGE_WIDTH, 102, fill=1, stroke=0)
    pdf.setFillColor(accent)
    pdf.rect(0, PAGE_HEIGHT - 7, PAGE_WIDTH, 7, fill=1, stroke=0)
    pdf.setFillColor(black)
    pdf.setFont("Helvetica-Bold", 15)
    pdf.drawString(48, PAGE_HEIGHT - 38, "SYNTHETIC MELANOMA PATHOLOGY REPORT")
    pdf.setFont("Helvetica-Bold", 9)
    pdf.setFillColor(HexColor("#A32020"))
    pdf.drawString(
        48,
        PAGE_HEIGHT - 57,
        "RESEARCH FIXTURE - NOT A REAL PATIENT - NOT FOR CLINICAL USE",
    )
    pdf.setFillColor(black)
    pdf.setFont("Helvetica", 8.5)
    pdf.drawString(48, PAGE_HEIGHT - 79, f"Document ID: {doc_id}")
    pdf.drawRightString(PAGE_WIDTH - 48, PAGE_HEIGHT - 79, f"Report date: {case['report_date']}")
    return PAGE_HEIGHT - 124


def draw_footer(pdf: canvas.Canvas, doc_id: str) -> None:
    pdf.setStrokeColor(HexColor("#C8CDD4"))
    pdf.line(48, 42, PAGE_WIDTH - 48, 42)
    pdf.setFillColor(HexColor("#5E6670"))
    pdf.setFont("Helvetica", 7.5)
    pdf.drawString(48, 29, f"{doc_id} | deterministic synthetic benchmark artifact")
    pdf.drawRightString(PAGE_WIDTH - 48, 29, "Page 1 of 1")


def draw_template_a(pdf: canvas.Canvas, doc_id: str, case: Mapping[str, str]) -> str:
    accent = HexColor("#1E5D78")
    y = draw_common_header(pdf, doc_id, case, accent)
    pdf.setFont("Helvetica-Bold", 11)
    pdf.setFillColor(accent)
    pdf.drawString(48, y, "CURRENT SPECIMEN - SYNOPTIC SUMMARY")
    y -= 17

    label_width = 205
    value_width = PAGE_WIDTH - 96 - label_width
    row_index = 0
    source_lines = [
        "SYNTHETIC RESEARCH REPORT - NOT A REAL PATIENT",
        f"Document ID: {doc_id}",
    ]
    for label, value in fact_rows(case):
        label_lines = wrap_text(label, "Helvetica-Bold", 8.2, label_width - 14)
        value_lines = wrap_text(value, "Helvetica", 8.4, value_width - 14)
        line_count = max(len(label_lines), len(value_lines))
        height = 13 + (line_count - 1) * 10
        if row_index % 2 == 0:
            pdf.setFillColor(HexColor("#F2F6F8"))
            pdf.rect(48, y - height + 4, PAGE_WIDTH - 96, height, fill=1, stroke=0)
        pdf.setFillColor(black)
        pdf.setFont("Helvetica-Bold", 8.2)
        for offset, line in enumerate(label_lines):
            pdf.drawString(55, y - 8 - offset * 10, line)
        pdf.setFont("Helvetica", 8.4)
        for offset, line in enumerate(value_lines):
            pdf.drawString(55 + label_width, y - 8 - offset * 10, line)
        y -= height
        row_index += 1
        source_lines.append(f"{label}: {value}")

    historical = null_if_blank(case["historical_context"])
    if historical:
        y -= 12
        pdf.setFillColor(HexColor("#FFF5D8"))
        history_lines = wrap_text(historical, "Helvetica", 8.4, PAGE_WIDTH - 116)
        box_height = 31 + 10 * len(history_lines)
        pdf.rect(48, y - box_height + 4, PAGE_WIDTH - 96, box_height, fill=1, stroke=0)
        pdf.setFillColor(black)
        pdf.setFont("Helvetica-Bold", 8.5)
        pdf.drawString(56, y - 9, "CLINICAL HISTORY - PRIOR REPORT ONLY")
        pdf.setFont("Helvetica", 8.4)
        for offset, line in enumerate(history_lines):
            pdf.drawString(56, y - 23 - offset * 10, line)
        y -= box_height
        source_lines.extend(["CLINICAL HISTORY - PRIOR REPORT ONLY", historical])

    additional = null_if_blank(case["additional_findings"])
    if additional:
        y -= 14
        pdf.setFillColor(accent)
        pdf.setFont("Helvetica-Bold", 9.5)
        pdf.drawString(48, y, "ADDITIONAL CURRENT-SPECIMEN FINDINGS")
        y -= 14
        pdf.setFillColor(black)
        pdf.setFont("Helvetica", 8.5)
        for line in wrap_text(additional, "Helvetica", 8.5, PAGE_WIDTH - 96):
            pdf.drawString(48, y, line)
            y -= 11
        source_lines.extend(["ADDITIONAL CURRENT-SPECIMEN FINDINGS", additional])

    draw_footer(pdf, doc_id)
    return "\n".join(source_lines) + "\n"


def narrative_sentences(case: Mapping[str, str]) -> List[str]:
    sentences = [
        f"The current specimen is a {case['procedure']} from the {displayed_site(case)}.",
        f"The diagnosis is {diagnosis_text(case).lower()}.",
    ]
    if breslow_text(case):
        sentences.append(f"Breslow thickness is {breslow_text(case)}.")
    if null_if_blank(case["ulceration"]):
        sentences.append(f"Ulceration: {display_status(case['ulceration']).lower()}.")
    if mitotic_text(case):
        sentences.append(f"Mitotic rate is {mitotic_text(case)}.")

    margin_phrases = []
    for component, location, key in (
        ("invasive melanoma", "peripheral", "invasive_peripheral"),
        ("invasive melanoma", "deep", "invasive_deep"),
        ("melanoma in situ", "peripheral", "in_situ_peripheral"),
        ("melanoma in situ", "deep", "in_situ_deep"),
    ):
        value = null_if_blank(case[key])
        if value:
            margin_phrases.append(f"{component} {location} margin: {display_status(value).lower()}")
    if margin_phrases:
        sentences.append("; ".join(margin_phrases) + ".")

    staging_phrases = []
    for label, key in (
        ("pT", "pT"),
        ("pN", "pN"),
        ("pM", "pM"),
        ("stage group", "stage_group"),
    ):
        value = null_if_blank(case[key])
        if value:
            staging_phrases.append(f"{label}: {value}")
    if staging_phrases:
        sentences.append("Reported pathologic classification - " + "; ".join(staging_phrases) + ".")
    return sentences


def draw_paragraph(
    pdf: canvas.Canvas,
    text: str,
    x: float,
    y: float,
    width: float,
    font_name: str = "Helvetica",
    font_size: float = 9.5,
    leading: float = 13,
) -> float:
    pdf.setFont(font_name, font_size)
    for line in wrap_text(text, font_name, font_size, width):
        pdf.drawString(x, y, line)
        y -= leading
    return y


def draw_template_b(pdf: canvas.Canvas, doc_id: str, case: Mapping[str, str]) -> str:
    accent = HexColor("#5A3D78")
    y = draw_common_header(pdf, doc_id, case, accent)
    source_lines = [
        "SYNTHETIC RESEARCH REPORT - NOT A REAL PATIENT",
        f"Document ID: {doc_id}",
    ]

    historical = null_if_blank(case["historical_context"])
    if historical:
        pdf.setFillColor(accent)
        pdf.setFont("Helvetica-Bold", 10)
        pdf.drawString(48, y, "CLINICAL HISTORY - PRIOR REPORT ONLY")
        y -= 15
        pdf.setFillColor(black)
        y = draw_paragraph(pdf, historical, 48, y, PAGE_WIDTH - 96, font_size=9)
        y -= 12
        source_lines.extend(["CLINICAL HISTORY - PRIOR REPORT ONLY", historical])

    pdf.setFillColor(accent)
    pdf.setFont("Helvetica-Bold", 11)
    pdf.drawString(48, y, "FINAL DIAGNOSIS - CURRENT SPECIMEN")
    y -= 18
    pdf.setFillColor(black)
    pdf.setFont("Helvetica-Bold", 10)
    diagnosis_line = f"{title_case(displayed_site(case))}, {title_case(case['procedure'])}:"
    pdf.drawString(48, y, diagnosis_line)
    y -= 16
    y = draw_paragraph(pdf, diagnosis_text(case) + ".", 64, y, PAGE_WIDTH - 128, font_size=10)
    source_lines.extend([diagnosis_line, diagnosis_text(case) + "."])

    y -= 12
    pdf.setFillColor(accent)
    pdf.setFont("Helvetica-Bold", 10)
    pdf.drawString(48, y, "MICROSCOPIC SUMMARY")
    y -= 17
    pdf.setFillColor(black)
    for sentence in narrative_sentences(case)[2:]:
        y = draw_paragraph(pdf, sentence, 48, y, PAGE_WIDTH - 96, font_size=9.2)
        y -= 5
        source_lines.append(sentence)

    additional = null_if_blank(case["additional_findings"])
    if additional:
        y -= 8
        pdf.setFillColor(accent)
        pdf.setFont("Helvetica-Bold", 10)
        pdf.drawString(48, y, "COMMENT")
        y -= 17
        pdf.setFillColor(black)
        y = draw_paragraph(pdf, additional, 48, y, PAGE_WIDTH - 96, font_size=9.2)
        source_lines.extend(["COMMENT", additional])

    draw_footer(pdf, doc_id)
    return "\n".join(source_lines) + "\n"


def generate_clean_pdf(path: Path, doc_id: str, template_id: str, case: Mapping[str, str]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    pdf = canvas.Canvas(
        str(path),
        pagesize=letter,
        bottomup=1,
        pageCompression=1,
        invariant=1,
    )
    pdf.setTitle(f"{doc_id} synthetic melanoma pathology report")
    pdf.setAuthor("Synthetic benchmark generator")
    pdf.setSubject("Research fixture; not for clinical use")
    source_text = (
        draw_template_a(pdf, doc_id, case)
        if template_id == "A"
        else draw_template_b(pdf, doc_id, case)
    )
    pdf.showPage()
    pdf.save()
    return source_text


def render_pdf(path: Path, dpi: int = DPI) -> Image.Image:
    document = pdfium.PdfDocument(str(path))
    if len(document) != 1:
        raise ValueError(f"Expected one page in {path}, found {len(document)}")
    page = document[0]
    bitmap = page.render(scale=dpi / 72.0)
    image = bitmap.to_pil().convert("RGB")
    page.close()
    document.close()
    return image


def degradation_parameters(doc_id: str) -> Dict[str, Any]:
    digest = sha256(f"{doc_id}|degradation-v1".encode("utf-8")).digest()
    sign = -1 if digest[0] % 2 else 1
    return {
        "perturbation_id": "scan-v1",
        "rotation_degrees": round(sign * (0.35 + (digest[1] / 255) * 0.30), 3),
        "blur_radius": round(0.55 + (digest[2] / 255) * 0.30, 3),
        "contrast_factor": round(0.82 + (digest[3] / 255) * 0.08, 3),
        "jpeg_quality": 58 + digest[4] % 15,
        "seed": int.from_bytes(digest[5:13], "big"),
    }


def degrade_image(image: Image.Image, params: Mapping[str, Any]) -> Image.Image:
    degraded = image.convert("L").convert("RGB")
    degraded = ImageEnhance.Contrast(degraded).enhance(params["contrast_factor"])
    degraded = degraded.rotate(
        params["rotation_degrees"],
        resample=Image.Resampling.BICUBIC,
        expand=False,
        fillcolor=(255, 255, 255),
    )
    degraded = degraded.filter(ImageFilter.GaussianBlur(params["blur_radius"]))

    buffer = io.BytesIO()
    degraded.save(
        buffer,
        format="JPEG",
        quality=params["jpeg_quality"],
        optimize=False,
        progressive=False,
        subsampling=2,
    )
    buffer.seek(0)
    degraded = Image.open(buffer).convert("RGB")

    rng = random.Random(params["seed"])
    draw = ImageDraw.Draw(degraded)
    width, height = degraded.size
    for _ in range(420):
        x = rng.randrange(width)
        y = rng.randrange(height)
        shade = rng.choice((115, 140, 165, 205))
        draw.point((x, y), fill=(shade, shade, shade))
    for _ in range(5):
        y = rng.randrange(80, height - 80)
        shade = rng.choice((224, 230, 235))
        draw.line((0, y, width, y), fill=(shade, shade, shade), width=1)
    return degraded


def image_only_pdf(path: Path, image: Image.Image, doc_id: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pdf = canvas.Canvas(
        str(path),
        pagesize=letter,
        bottomup=1,
        pageCompression=1,
        invariant=1,
    )
    pdf.setTitle(f"{doc_id} degraded synthetic report")
    pdf.setAuthor("Synthetic benchmark generator")
    pdf.drawImage(
        ImageReader(image),
        0,
        0,
        width=PAGE_WIDTH,
        height=PAGE_HEIGHT,
        preserveAspectRatio=False,
        mask=None,
    )
    pdf.showPage()
    pdf.save()


def row_hash(case: Mapping[str, str]) -> str:
    canonical = json.dumps(dict(case), sort_keys=True, separators=(",", ":"))
    return sha256_text(canonical)


def output_paths(doc_id: str) -> Dict[str, Path]:
    return {
        "clean_pdf": ROOT / "output" / "pdf" / "clean" / f"{doc_id}.pdf",
        "degraded_pdf": ROOT / "output" / "pdf" / "ocr_degraded" / f"{doc_id}.pdf",
        "clean_image": ROOT / "output" / "rendered" / "clean" / f"{doc_id}.png",
        "degraded_image": ROOT / "output" / "rendered" / "ocr_degraded" / f"{doc_id}.png",
        "source_text": ROOT / "data" / "source_text" / f"{doc_id}.txt",
        "ground_truth": ROOT / "data" / "ground_truth" / f"{doc_id}.json",
    }


def generate(refresh_empty_ledgers: bool = False) -> None:
    cases = load_cases()
    if len(cases) != 8:
        raise ValueError(f"Pilot must contain exactly 8 semantic cases, found {len(cases)}")
    if len({case["case_id"] for case in cases}) != len(cases):
        raise ValueError("Case IDs must be unique")

    manifest_rows: List[Dict[str, Any]] = []
    for case in cases:
        for template_id in TEMPLATE_IDS:
            doc_id = document_id(case["case_id"], template_id)
            paths = output_paths(doc_id)
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

            common = {
                "dataset_version": DATASET_VERSION,
                "generator_version": GENERATOR_VERSION,
                "semantic_case_id": case["case_id"],
                "document_id": doc_id,
                "template_id": template_id,
                "dpi": DPI,
                "renderer": f"pypdfium2-{distribution_version('pypdfium2')}",
                "source_row_sha256": row_hash(case),
                "source_text_path": relative(paths["source_text"]),
                "ground_truth_path": relative(paths["ground_truth"]),
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
                    "pdf_path": relative(paths["clean_pdf"]),
                    "image_path": relative(paths["clean_image"]),
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
                    "pdf_path": relative(paths["degraded_pdf"]),
                    "image_path": relative(paths["degraded_image"]),
                    "pdf_sha256": sha256_file(paths["degraded_pdf"]),
                    "image_sha256": sha256_file(paths["degraded_image"]),
                }
            )

    manifest_fields = (
        "dataset_version",
        "generator_version",
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
    write_csv(ROOT / "data" / "report_manifest.csv", manifest_fields, manifest_rows)
    initialize_result_ledgers(manifest_rows, refresh_empty_ledgers)
    print(
        f"Generated {len(cases)} cases, {len(cases) * len(TEMPLATE_IDS)} base reports, "
        f"and {len(manifest_rows)} document conditions."
    )


def initialize_result_ledgers(
    manifest_rows: Sequence[Mapping[str, Any]],
    refresh_empty_ledgers: bool = False,
) -> None:
    prompt_sha = sha256_file(ROOT / "prompts" / "extraction_prompt.txt")
    schema_sha = sha256_file(ROOT / "schema" / "extraction.schema.json")
    formal_rows = formal_manifest_rows(manifest_rows)
    intended_rows = []
    review_rows = []
    for model_id in MODEL_IDS:
        for item in formal_rows:
            base = {
                "model_id": model_id,
                "semantic_case_id": item["semantic_case_id"],
                "document_id": item["document_id"],
                "template_id": item["template_id"],
                "condition": item["condition"],
            }
            intended_rows.append(
                {
                    **base,
                    "run_status": "NOT_RUN",
                    "raw_present": "",
                    "provenance_valid": "",
                    "provenance_errors": "",
                    "raw_output_sha256": "",
                    "parse_valid": "",
                    "schema_valid": "",
                    "document_id_correct": "",
                    "field_exact_correct": "",
                    "field_exact_total": "",
                    "field_exact_match": "",
                    "true_positive": "",
                    "false_positive": "",
                    "false_negative": "",
                    "precision": "",
                    "recall": "",
                    "f1": "",
                    "unsupported_count": "",
                    "unsupported_opportunities": "",
                    "unsupported_field_rate": "",
                    "complete_report": "",
                }
            )
            review_rows.append(
                {
                    **base,
                    "review_status": REVIEW_STATUS_PENDING,
                    "raw_output_sha256": "",
                    "parse_valid": "",
                    "schema_valid": "",
                    "document_id_correct": "",
                    "mismatched_fields": "",
                    "unsupported_fields": "",
                    "source_ambiguity": "",
                    "adjudication": "",
                    "reviewer": "",
                    "review_date": "",
                }
            )

    pilot_path = ROOT / "results" / "pilot_table.csv"
    review_path = ROOT / "results" / "manual_review.csv"
    if refresh_empty_ledgers:
        run_manifest_path = ROOT / "results" / "run_manifest.json"
        current = read_json(run_manifest_path) if run_manifest_path.exists() else {}
        raw_files = list((ROOT / "results" / "raw").glob("**/*.txt"))
        hash_bound_reviews = False
        if review_path.exists():
            with review_path.open(newline="", encoding="utf-8") as handle:
                hash_bound_reviews = any(
                    row.get("review_status") in HASH_BOUND_REVIEW_STATUSES
                    for row in csv.DictReader(handle)
                )
        if (
            current.get("status", "NOT_RUN") != "NOT_RUN"
            or current.get("completed_output_count", 0) != 0
            or raw_files
            or hash_bound_reviews
        ):
            raise ValueError("Refusing to refresh non-empty result ledgers")

    if refresh_empty_ledgers or not pilot_path.exists():
        write_csv(pilot_path, intended_rows[0].keys(), intended_rows)
    if refresh_empty_ledgers or not review_path.exists():
        write_csv(review_path, review_rows[0].keys(), review_rows)

    run_manifest_path = ROOT / "results" / "run_manifest.json"
    if refresh_empty_ledgers or not run_manifest_path.exists():
        write_json(
            run_manifest_path,
            {
                "dataset_version": DATASET_VERSION,
                "protocol_version": PROTOCOL_VERSION,
                "status": "NOT_RUN",
                "models": list(MODEL_IDS),
                "model_revisions": MODEL_REVISIONS,
                "conditions": list(CONDITIONS),
                "development_case_ids": list(DEVELOPMENT_CASE_IDS),
                "formal_semantic_case_count": FORMAL_SEMANTIC_CASE_COUNT,
                "formal_base_report_count": FORMAL_BASE_REPORT_COUNT,
                "formal_document_condition_count": FORMAL_DOCUMENT_CONDITION_COUNT,
                "intended_output_count": FORMAL_OUTPUT_COUNT,
                "completed_output_count": 0,
                "response_mode": FORMAL_RESPONSE_MODE,
                "assistant_prefix": JSON_ASSISTANT_PREFIX,
                "assistant_prefix_sha256": sha256_bytes(JSON_ASSISTANT_PREFIX.encode("utf-8")),
                "dtype": FORMAL_DTYPE,
                "max_new_tokens": FORMAL_MAX_NEW_TOKENS,
                "do_sample": False,
                "prompt_sha256": prompt_sha,
                "schema_sha256": schema_sha,
                "run_record_paths": [],
            },
        )
    metrics_path = ROOT / "results" / "metrics.json"
    if refresh_empty_ledgers or not metrics_path.exists():
        write_json(
            metrics_path,
            {
                "status": "NOT_RUN",
                "protocol_version": PROTOCOL_VERSION,
                "intended_output_count": FORMAL_OUTPUT_COUNT,
                "completed_output_count": 0,
                "message": "No MedGemma outputs have been scored.",
            },
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--refresh-empty-ledgers",
        action="store_true",
        help="Rewrite ledgers only when no outputs or hash-bound reviews exist.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    generate(args.refresh_empty_ledgers)
