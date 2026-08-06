# Technical gate status

Last updated: 2026-08-05

| Gate | Evidence required | Status |
|---|---|---|
| 1. Create 10–20 reports from two templates | Current v4 corpus: 6 semantic cases rendered as 12 report-layout documents across two templates, yielding 24 rendered inputs forming 12 clean/degraded pairs (4 development and 20 execution-held-out) | **Complete and deterministic** |
| 2. Confirm MedGemma 1.5 access and inference | Accepted gated terms, exact revision access, BF16 model load, OCR, and both constrained JSON paths | **Complete preflight:** exact revision loaded on Tesla T4; OCR and both small-schema, non-corpus serializer smokes passed |
| 3. Produce valid JSON for clean and degraded inputs | Complete four-input v4 development matrix before the 20-input formal run | **Incomplete:** sole development runner launched; Kaggle remained in `Error` before any report output or metric was observed or recovered; classified terminal incomplete under the single-iteration policy |
| 4. Human-verify every formal output | Human review bound to all completed formal outputs | **Not started:** no v4 formal outputs exist |
| 5. Publish reproducible repository | Public GitHub history with source, corpus, protocol, and honest outcome record | **Complete on the public branch:** source, corpus, protocol, and v4 incident record are linked from the README |
| 6. Submit through HAI-DEF | Feedback-form confirmation recorded locally | Pending user-approved submission |

## Current v4 disposition

The v4 source and corpus passed local and Kaggle setup validation. In the live
GPU runtime, all 122 tests passed, the exact MedGemma 1.5 revision loaded in
BF16, Tesseract read a non-corpus preflight image, and both effective
constrained-JSON configurations generated strict JSON valid against their
small smoke schemas without a cap hit. These smokes did not exercise either
full clinical schema or any report.

The sole four-input development runner was then launched. Kaggle entered a
persistent session-level `Error` before any report-level output was observed
or recovered. No archive from the wrapper's planned `finally` block was
observed or recovered after the session and its `/kaggle/working` listing
became unavailable. Per the single-development-iteration policy and the
runner's claim-before-backend design, the project conservatively classifies
this launch as terminal incomplete; no retry was attempted. No formal lock or
formal execution followed.

See [`v4_development_incident.md`](v4_development_incident.md) for the exact
observations, reproducibility identifiers, and limitations.

## Historical v2 gate 2 evidence

The HAI-DEF terms are accepted for both Hugging Face repositories. A read-only
token confirmed access to these exact revisions:

- MedGemma 1 4B IT:
  `290cda5eeccbee130f987c4ad74a59ae6f196408`
- MedGemma 1.5 4B IT:
  `91850547d9f0b2fdd21aa7c5f4f3d1a8a52c243b`

The free Colab runtime exposes a Tesla T4 with 15,360 MiB. FP16 produced an
empty harness response and a direct diagnostic generated only padding; this
attempt is classified as an infrastructure/runtime failure and preserved.
PyTorch reported BF16 support, BF16 matrix multiplication was finite, and a
direct BF16 diagnostic produced coherent text. The frozen formal protocol
therefore uses BF16.

Development on the two clean `MEL-001` layouts confirmed that the chosen
no-example prompt plus a literal `{` assistant prefix produces a strict
assembled JSON response. These outputs remain development evidence and are not
formal results.

## Historical v2 gates 3 and 4 evidence

Both pinned models completed all 28 held-out inputs on the immutable protocol
commit `32f1453e2acbc79dbe50c7f80d9f42a519fdba8f`. The evaluator reports 56/56
completed outputs, zero invalid-provenance outputs, and 56/56 parse-valid JSON
responses. Schema validation is 0/28 for each model, complete-report accuracy
is 0/28 for each model, and the extraction-quality gate therefore did not
pass.

MedGemma 1.5 achieved 178/448 field exact matches (39.7%), 40.0% non-null
micro-F1, and 14 unsupported values in 172 ground-truth-null opportunities
(8.1%). The MedGemma 1 comparator achieved 230/448 (51.3%), 52.0% micro-F1,
and 47/172 unsupported values (27.3%).

Codex reviewer agents visually audited all 28 unique source images and all 56
raw responses, parsed objects, deterministic truth records, and provenance
records. Every output-level row is marked `ai_audited` and bound to the current
raw-output SHA-256. The bounded degradation remained visually legible in every
case; adjudicated discrepancies were model errors rather than source
ambiguity. This AI-assisted audit is not represented as human manual review,
which remains 0/56.

## Outreach hold

Do not describe v4 as a successful pilot or passed technical gate. Publish the
incident record, submit the use case through the HAI-DEF feedback form, and
route a reproducible runtime/document-extraction question through the
developer forum or GitHub first. If individual contact occurs before a
completed gate, use only the transparent, result-free wording in
[`outreach_email.md`](outreach_email.md).
