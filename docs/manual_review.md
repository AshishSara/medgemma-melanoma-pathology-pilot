# Manual review procedure

Use `results/manual_review.csv` as the review ledger. Each row corresponds to
one model, held-out report-layout document, and render condition. The ledger
contains 56 rows: 14 report-layout documents x 2 conditions x 2 models.

For every row, a reviewer must:

1. Open the source PDF and its rendered PNG.
2. Read the unmodified raw model output.
3. Compare normalized JSON with the deterministic ground truth.
4. Record any wrong, missing, or unsupported field.
5. Mark whether source wording is genuinely ambiguous.
6. Add a concise adjudication note and reviewer identity.

Review status has three explicit meanings:

- `pending`: no hash-bound review has been completed.
- `ai_audited`: an AI reviewer completed the six-step visual and provenance
  audit. This may prepare the ledger but does not satisfy the human-review
  outreach gate.
- `human_verified`: a human reviewer personally completed all six steps. Only
  this status satisfies the manual-review outreach gate.

Both completed statuses are bound to `raw_output_sha256`; rerunning or changing
the raw bytes resets the row to `pending`. Automated metrics do not replace
visual review, and an AI-assisted audit does not replace human verification.

`MEL-001` is development-only and must not be added to this ledger. Review the
strict assembled assistant response as scored. Do not silently correct
noncanonical enum strings, missing keys, types, or values during adjudication.

At least one dermatopathologist should approve the schema before the full
75-100-report study and review the deliberately ambiguous/historical-context
cases.
