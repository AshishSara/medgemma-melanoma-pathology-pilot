# Held-out pilot protocol v2

Protocol identifier: `heldout-pilot-v2`

This document was frozen after development on `MEL-001` and before inspecting
any output from `MEL-002` through `MEL-008`. The earlier repository protocol is
retained in Git history; it should not be described as a preregistration of the
revised response constraint.

## Objective

Estimate extraction accuracy and unsupported-value generation for
management-critical melanoma pathology fields under two report layouts and two
render conditions.

## Corpus and split

The complete synthetic corpus contains eight semantic cases:

- Every case is rendered as compact synoptic Template A and narrative Template
  B, yielding 16 report-layout documents.
- Every document has a clean text PDF and a matched image-only degraded PDF,
  yielding 32 rendered inputs.
- All identifiers are explicitly synthetic; no names or real accessions appear.

The full `MEL-001` semantic case is development-only. Its Template A and B clean
images were used for response-format and prompt selection. Both degraded
versions are also excluded because they share the same underlying facts.

The held-out formal set is `MEL-002` through `MEL-008`: 7 semantic cases, 14
report-layout documents, and 28 rendered inputs forming 14 clean/degraded
pairs. Across two models, the formal review ledger contains 56 outputs. The 28
inputs per model are not described as 28 independent reports.

## Development prompt selection

MedGemma 1.5 was evaluated on the two `MEL-001` clean layouts only. Direct BF16
generation with the original illustrative-object prompt spent the 1,200-token
budget in reasoning and did not produce JSON. A one-character JSON assistant
prefix made output syntactically parseable, but the illustrative object caused
the model to copy the example identifier and null values. The example was
therefore removed.

Two no-example prompt candidates were then compared:

| Candidate | Template A field exact | Template B field exact | Pooled |
|---|---:|---:|---:|
| v2 | 8/16 | 5/16 | 13/32 |
| v3, stricter wording | 7/16 | 6/16 | 13/32 |

The candidates tied on pooled development exactness. V2 was selected before
held-out inference as the simpler fully preserved specification; no claim is
made that it had higher pooled accuracy. This is a manually prompt-optimized
no-example protocol, not an unqualified stock-prompt or fine-tuned evaluation.

All original, v2, and v3 attempts are development evidence and must remain
outside `results/raw/`, `results/run_records/`, and every formal aggregate.

## Target fields

- document ID
- specimen site and laterality
- diagnosis context
- Breslow thickness and qualifier
- ulceration
- mitotic rate and qualifier
- invasive and in-situ peripheral/deep margin status
- explicitly reported pT, pN, pM, and stage group

The target is extraction, not derivation. If a staging value is not stated, the
correct output is `null` even when it could be inferred. Historical
measurements mentioned only in clinical history must not be substituted for
current-specimen values.

## Frozen formal inference

- Primary model: `google/medgemma-1.5-4b-it` at revision
  `91850547d9f0b2fdd21aa7c5f4f3d1a8a52c243b`.
- Secondary shared-protocol comparator: `google/medgemma-4b-it` at revision
  `290cda5eeccbee130f987c4ad74a59ae6f196408`.
- Inputs: exact committed PNGs whose SHA-256 values appear in
  `data/report_manifest.csv`.
- Prompt: exactly `prompts/extraction_prompt.txt`, SHA-256
  `139bc7f1dc858af0a5a2643f38b396f71c40e39e1419c14a281dd78f03ee801b`.
- Runtime: free Google Colab Tesla T4; BF16; dependency versions recorded per
  call.
- Decoding: one greedy run per input; `do_sample=False`; maximum 1,200 new
  tokens; normal EOS stopping.
- Response constraint: append an assistant message containing exactly `{` and
  call Transformers with `continue_final_message=True`.
- The strict scored response is the assembled assistant response, including
  that harness-supplied prefix. The model-generated continuation is retained
  separately.
- Prompt or parser changes after any held-out output is opened are prohibited.

The 1,200-token ceiling is retained from development. Reaching the ceiling with
an incomplete response is a model output failure, not a reason to extend the
budget post hoc.

Only a documented infrastructure failure may be retried, and only with the
same image, prompt, revision, dtype, decoding settings, and response mode. A
prior attempt must be preserved before an overwrite. Semantic or formatting
failure is not retryable.

## Strict parsing and provenance

Raw assistant-response bytes are retained unchanged. Deterministic parsing may:

1. trim surrounding whitespace; and
2. remove at most one complete outer Markdown fence.

It may not search for a later JSON object, strip a reasoning envelope, insert
or remove keys, coerce types or units, map synonyms, infer qualifiers or
staging, or otherwise repair output. Duplicate keys, non-finite numbers,
missing required keys, wrong types, and noncanonical enum strings remain model
failures.

Every formal run record contains the repository commit, protocol version,
requested and resolved model revisions, runtime versions, device, dtype,
generation settings, assistant-prefix hash, image/prompt/schema hashes, raw
response and continuation hashes/byte counts, and parse/schema results.
Evaluation refuses outputs whose provenance does not match the frozen
protocol.

## Render conditions

`clean` is rendered from the vector/text PDF at 160 DPI.

`ocr_degraded` is an image-only PDF and PNG derived from the clean raster with
a bounded, deterministic combination of:

- rotation no greater than 0.65 degrees;
- mild Gaussian blur;
- reduced contrast;
- JPEG recompression;
- sparse speckling and faint scan lines.

These are readability perturbations, not adversarial corruptions. Exact
case-level parameters are in `data/report_manifest.csv`.

## Metrics

For each of 16 atomic fields:

- exact match: type-aware canonical prediction equals ground truth; a missing
  key does not equal an explicit `null`;
- extraction precision/recall/F1: non-null exact matches are true positives;
  wrong non-null values are false positives and, when a source value exists,
  false negatives;
- unsupported-field rate: non-null prediction when ground truth is `null`,
  divided by all ground-truth-null opportunities.

Also report:

- strict JSON parse rate;
- strict JSON Schema validity rate;
- document-ID accuracy;
- complete-report accuracy;
- results by model, template, and condition;
- paired OCR-degraded minus clean field-exact change.

Unsupported-field rate must be shown beside non-null recall or micro-F1 so an
all-null response cannot appear safe. Zero unsupported values must not be
described as “hallucination-free.”

## Manual review

Every one of the 56 formal outputs must be inspected against the source image
and deterministic ground truth. Reviewers record parse and schema validity,
document-ID correctness, discrepancies, unsupported fields, source ambiguity,
and adjudication. A verified review is bound to the raw-output SHA-256; changed
bytes automatically invalidate that verification.

Automated metrics do not replace visual review. A qualified
dermatopathologist should approve the schema before expansion and review
approximately 15 genuinely ambiguous cases in a larger study.

## Expansion and outreach

Only after all formal calls, metrics, and manual reviews are complete should
the corpus expand toward 75–100 reports. A negative feasibility result remains
a valid result. Complete the HAI-DEF use-case submission and route technical
questions through the developer forum or GitHub before individual research
outreach.
