# Technical gate status

Last updated: 2026-08-04

| Gate | Evidence required | Status |
|---|---|---|
| 1. Create 10–20 reports from two templates | Complete corpus: 8 semantic cases / 16 report-layout documents / 32 rendered inputs; held-out set: 7 / 14 / 28 after quarantining `MEL-001` | Complete |
| 2. Confirm MedGemma 1.5 access and inference | Accepted gated terms, pinned revision access, and recorded GPU/dtype behavior | **Complete: both revisions accessible; Tesla T4 BF16 generation confirmed** |
| 3. Produce valid JSON for clean and degraded inputs | 28 strict held-out responses for MedGemma 1.5 plus the 28-output secondary comparator | Pending frozen-protocol run |
| 4. Verify every formal output | 56 completed manual-review rows bound to raw-output hashes | Pending gate 3 |
| 5. Publish reproducible repository | Public GitHub history with corpus, prompt, code, schema, and ledgers | Baseline published at [`fe97c9c`](https://github.com/AshishSara/medgemma-melanoma-pathology-pilot/tree/fe97c9c2987df9e3b23cee8bfa7dba9a17d68bc3); held-out protocol update in progress |
| 6. Submit through HAI-DEF | Feedback-form confirmation recorded locally | Pending completed pilot |

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

## Outreach hold

Do not contact Google based on repository publication or development output
alone. Complete the 56-output held-out comparison, manually verify every raw
response, replace the pilot-result placeholder with measured denominators,
submit the HAI-DEF feedback form, and route technical questions through the
developer forum or GitHub first.
