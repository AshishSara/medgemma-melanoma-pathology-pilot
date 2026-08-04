# Pilot protocol

## Objective

Estimate zero-shot extraction accuracy and unsupported-value generation for
management-critical melanoma pathology fields under two report layouts and two
render conditions.

## Corpus

The pilot contains eight unique semantic cases:

- Every case is rendered once as compact synoptic Template A and once as
  narrative Template B, yielding 16 base reports.
- Every base report has a clean text PDF and a paired image-only degraded PDF,
  yielding 32 document conditions.
- All identifiers are explicitly synthetic; no names or real accessions appear.

The paired design isolates degradation effects because case content is identical
within a pair. Single-page reports avoid confounding the first gate with
multi-page aggregation.

## Target fields

- document ID
- specimen site and laterality
- diagnosis context
- Breslow thickness and qualifier
- ulceration
- mitotic rate and qualifier
- invasive and in-situ peripheral/deep margin status
- reported pT, pN, pM, and stage group

The target is extraction, not derivation. If a staging value is not stated in
the source report, the correct output is `null` even when it could be inferred
from other findings. Historical measurements mentioned only in clinical
history must not be substituted for current-specimen values.

## Pre-registered inference

- Models: `google/medgemma-4b-it` and
  `google/medgemma-1.5-4b-it`.
- Inputs: the model-ready PNG rendered from each PDF.
- Prompt: exactly `prompts/extraction_prompt.txt`.
- Decoding: one greedy run per input; `do_sample=False`, temperature 0.
- Prompt changes after inspecting scored outputs are prohibited.
- Raw model text is retained unchanged.
- Deterministic parsing may remove one outer Markdown code fence and leading or
  trailing whitespace. No semantic correction is allowed before scoring.
- Model repository revision, dependency versions, renderer version, run time,
  and device must be recorded.

## Render conditions

`clean` is rendered from the vector/text PDF at 160 DPI.

`ocr_degraded` is an image-only PDF and PNG derived from the clean raster with
a bounded, deterministic combination of:

- rotation no greater than 0.65 degrees,
- mild Gaussian blur,
- reduced contrast,
- JPEG recompression,
- sparse speckling and faint scan lines.

These are readability perturbations, not adversarial corruptions. The exact
case-level parameters are written to `data/report_manifest.csv`.

## Metrics

For each field:

- exact match: normalized prediction equals ground truth;
- extraction precision/recall/F1: non-null exact value matches are true
  positives; wrong non-null values are false positives and, when a source value
  exists, false negatives;
- unsupported-field rate: non-null prediction when ground truth is null,
  divided by all ground-truth-null opportunities.

Also report:

- raw JSON parse rate;
- JSON Schema validity rate;
- document-ID accuracy against the source image;
- complete-report accuracy across all scored fields;
- results by template and render condition;
- paired clean-to-degraded change.

## Manual review

Every one of the 64 pilot outputs must be inspected against its paired source
report and deterministic ground truth. Reviewers must record parse validity,
document-ID correctness, each discrepancy, whether it is unsupported, source
ambiguity, and final adjudication. No result may be marked complete solely from
automated metrics.

## Expansion criterion

Only after the pilot pipeline is complete and errors are understood should the
corpus expand toward 75-100 reports. A dermatopathologist should first approve
the schema and review approximately 15 genuinely ambiguous cases.
