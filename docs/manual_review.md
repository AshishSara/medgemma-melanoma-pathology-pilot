# Manual review procedure

Use `results/manual_review.csv` as the review ledger. Each row corresponds to
one model, case, and render condition.

For every row:

1. Open the source PDF and its rendered PNG.
2. Read the unmodified raw model output.
3. Compare normalized JSON with the deterministic ground truth.
4. Record any wrong, missing, or unsupported field.
5. Mark whether source wording is genuinely ambiguous.
6. Add a concise adjudication note and reviewer initials.

`review_status=verified` is permitted only after all six steps. Automated exact
match does not replace visual review.

At least one dermatopathologist should approve the schema before the full
75-100-report study and review the deliberately ambiguous/historical-context
cases.
