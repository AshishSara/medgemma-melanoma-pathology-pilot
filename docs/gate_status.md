# Technical gate status

Last updated: 2026-08-04

| Gate | Evidence required | Status |
|---|---|---|
| 1. Create 10-20 reports from two templates | Eight cases rendered in both templates: 16 base reports and 32 paired document conditions | Materials ready |
| 2. Confirm MedGemma 1.5 access and inference | Accepted gated terms, authenticated model pull or deployed endpoint, recorded hardware | **In progress: repository terms accepted; GPU runtime pending** |
| 3. Produce valid JSON for clean and degraded inputs | 32 raw outputs for MedGemma 1.5 and schema-validation log | Pending gate 2 |
| 4. Verify every output | Completed manual-review rows with adjudication | Pending gate 3 |
| 5. Publish reproducible repository | Public GitHub URL at an immutable commit | Published at [`5a4d965`](https://github.com/AshishSara/medgemma-melanoma-pathology-pilot/tree/5a4d965e1ca67637e49fc4bd5c96889eb28c3b33) |
| 6. Submit the use case through HAI-DEF | Feedback-form confirmation recorded locally | Pending completed pilot |

## Why gate 2 is not marked complete

The HAI-DEF terms have been accepted for both Hugging Face repositories. The
official shared Hugging Face inference providers do not serve these models.
This Mac has 8 GB of RAM and insufficient free storage for a defensible BF16
comparison; Google's published sizing estimates 6.4 GB for static BF16 weights
alone, before runtime memory. An authenticated GPU-backed environment or a
self-deployed endpoint is therefore still required.

No synthetic or substitute model output is presented as MedGemma output.

## Outreach hold

Do not contact Google based on repository publication alone. Complete the
64-output comparison, manually verify it, replace the pilot-result placeholder
in the private outreach draft with measured numbers, and submit the HAI-DEF
feedback form first.
