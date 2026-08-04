# MedGemma melanoma pathology extraction pilot

This repository is a small, deterministic technical gate for testing whether
MedGemma can extract management-critical melanoma fields from heterogeneous
pathology-report images without inventing unstated values.

> Status: **materials ready; scored MedGemma inference not yet run.** The
> synthetic corpus, paired render conditions, JSON ground truth, frozen prompt,
> validator, inference harness, metric code, and manual-review ledger are
> reproducible. Model access and the 64-output scored run remain gating items.
> Do not use the outreach draft yet.

## Research question

Can MedGemma 1.5 accurately extract specimen site and laterality, Breslow
thickness, ulceration, mitotic rate, margin status, and reported staging
elements from heterogeneous pathology-report PDFs without inventing unstated
values, and how robust is extraction to bounded raster/OCR degradation?

The defensible novelty is the combination of narrative melanoma management
fields, explicit abstention targets, unsupported-field measurement, and paired
layout/degradation testing. Synthetic PDFs or layout variation alone are not
novel: the MedGemma 1.5 report describes a synthetic custom-PDF training and
evaluation dataset.

## Pilot design

- Eight semantic cases rendered through two original report templates: 16
  wholly synthetic, single-page base reports.
- A clean text PDF and a paired image-only degraded PDF for every case.
- Deterministic JSON ground truth generated from `data/cases.csv`.
- Identical prompt and greedy decoding for:
  - `google/medgemma-4b-it`
  - `google/medgemma-1.5-4b-it`
- 16 base reports x 2 render conditions x 2 models = 64 planned outputs.
- Primary metrics: per-field exact match/F1 and unsupported-field rate.
- Secondary metrics: complete-report accuracy, JSON validity, and performance
  by template and render condition.
- Manual review of all 64 pilot outputs before any outreach.

PDF files are rendered to images before inference because the released model
accepts text plus images, not PDF bytes. This mirrors the document-understanding
method described in the technical report.

## Reproduce the materials

Python 3.9+ and [`uv`](https://docs.astral.sh/uv/) are recommended.

```bash
uv sync --extra dev
uv run python scripts/generate_reports.py
uv run python scripts/validate_artifacts.py
uv run pytest
```

The generator uses fixed data, fixed dates, and deterministic perturbation
parameters. Clean PDFs are under `output/pdf/clean/`; degraded image-only PDFs
are under `output/pdf/ocr_degraded/`; model-ready PNGs are under
`output/rendered/`.

## Run inference

The local backend requires accepted HAI-DEF terms, authenticated Hugging Face
access, and the optional inference dependencies:

```bash
uv sync --extra inference
uv run python scripts/run_inference.py \
  --backend transformers \
  --model google/medgemma-1.5-4b-it \
  --condition clean
```

Run all four model/condition cells, then evaluate:

```bash
uv run python scripts/run_inference.py \
  --backend transformers \
  --model google/medgemma-4b-it \
  --condition all
uv run python scripts/run_inference.py \
  --backend transformers \
  --model google/medgemma-1.5-4b-it \
  --condition all
uv run python scripts/evaluate.py
```

An OpenAI-compatible vision endpoint backend is also supported for a
self-deployed vLLM/Vertex-compatible service. No model weights, tokens, or
patient data belong in this repository.

## Gate status

See [`docs/gate_status.md`](docs/gate_status.md). Publication is useful only as
an honest reproducibility artifact; it is not evidence that model inference
passed. The HAI-DEF feedback form, developer forum, and only then individual
research outreach are the intended sequence.

## Safety and scope

All reports are synthetic and visibly watermarked. No real patient data are
used. This is a research evaluation, not a medical device, diagnostic system,
or clinical recommendation. Schema and ambiguous cases require review by a
qualified dermatopathologist before expansion.

## Evidence

- [MedGemma 1.5 technical report](https://arxiv.org/abs/2604.05081)
- [MedGemma 1.5 model card](https://developers.google.com/health-ai-developer-foundations/medgemma/model-card)
- [Official MedGemma get-started and engagement paths](https://developers.google.com/health-ai-developer-foundations/medgemma/get-started)
- [Current CAP cancer protocol index](https://www.cap.org/protocols-and-guidelines/cancer-protocols/current-cancer-protocols/)

MedGemma model weights remain governed by the HAI-DEF terms. This repository's
original code and synthetic artifacts are MIT-licensed; it does not redistribute
model weights.
