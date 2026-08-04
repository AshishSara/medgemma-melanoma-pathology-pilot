# Technical gate status

Last updated: 2026-08-04

| Gate | Evidence required | Status |
|---|---|---|
| 1. Create 10–20 reports from two templates | Complete corpus: 8 semantic cases / 16 report-layout documents / 32 rendered inputs; held-out set: 7 / 14 / 28 after quarantining `MEL-001` | Complete |
| 2. Confirm MedGemma 1.5 access and inference | Accepted gated terms, pinned revision access, and recorded GPU/dtype behavior | **Complete: both revisions accessible; Tesla T4 BF16 generation confirmed** |
| 3. Produce valid JSON for clean and degraded inputs | 28 strict held-out responses for MedGemma 1.5 plus the 28-output secondary comparator | **Execution complete; quality gate not passed:** 56/56 parse-valid, 0/56 schema-valid |
| 4. Human-verify every formal output | 56 human-reviewed rows bound to raw-output hashes | **Pending:** Codex AI-assisted audit is 56/56 with 0 stale hashes; human verification is 0/56 |
| 5. Publish reproducible repository | Public GitHub history with corpus, prompt, code, schema, ledgers, and formal evidence | Baseline [`fe97c9c`](https://github.com/AshishSara/medgemma-melanoma-pathology-pilot/tree/fe97c9c2987df9e3b23cee8bfa7dba9a17d68bc3); held-out protocol and formal-evidence work in [draft PR #1](https://github.com/AshishSara/medgemma-melanoma-pathology-pilot/pull/1) |
| 6. Submit through HAI-DEF | Feedback-form confirmation recorded locally | Pending user-approved submission |

## Gate 2 evidence

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

## Gates 3 and 4 evidence

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

Do not contact an individual Google researcher yet. Publish the explicitly
labeled AI-audited evidence, complete human manual verification, replace the
pilot-result placeholder with measured denominators, submit the result through
the HAI-DEF feedback form, and route a reproducible schema-compliance question
through the developer forum or GitHub first. Reconsider individual outreach
only after those steps and in light of the negative quality-gate result.
