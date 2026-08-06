# Final project report

Date: 2026-08-05

## Bottom line

The reproducible repository is public, but the current MedGemma v4 technical
gate is **incomplete and did not pass**.

The model-access and infrastructure preflight succeeded: the exact pinned
MedGemma 1.5 revision loaded on a Kaggle Tesla T4 in BF16, OCR worked, and two
small non-corpus serializer-configuration checks returned JSON valid against
their smoke schemas. The sole four-input development runner was then launched,
but the Kaggle session remained in `Error` through the final observation
window. No report-level v4 output or metric was observed or recovered, and the
20-input formal evaluation was not run.

This means there is no positive v4 extraction result to report. A
success-oriented email should not be sent.

## What was completed

1. A wholly synthetic melanoma-pathology extraction project was created in a
   dedicated Obsidian project folder and Git repository.
2. Deterministic report generators, JSON ground truth, prompts, schemas,
   evaluators, provenance checks, and regression tests were implemented.
3. Clean and degraded paired report inputs were generated across two report
   templates.
4. Historical v2 inference completed for 56 held-out outputs. It produced a
   complete negative quality result: 56/56 outputs parsed as JSON, but 0/56
   conformed to the frozen clinical schema.
5. V3 introduced schema-constrained two-pass extraction. Its formal attempt
   stopped after 3 of 20 inputs when an audit response reached the frozen
   768-token cap before completing valid JSON.
6. V4 prespecified a serialization-only recovery without changing the
   clinical task, model revision, prompts, schemas, compiler, fields, or pass
   thresholds.
7. The v4 source and corpus passed local and Kaggle validation. All 122 tests,
   Ruff lint, Ruff formatting, and `git diff --check` passed.
8. The exact model revision
   `91850547d9f0b2fdd21aa7c5f4f3d1a8a52c243b` loaded on `cuda:0` in
   `torch.bfloat16`, with no CPU or disk model placement.
9. Tesseract read a non-corpus preflight image. Two small serializer smokes
   confirmed the frozen pass-specific configuration, but did not exercise a
   clinical schema or report.
10. The v4 development runner was launched once. The hosted session then
    entered a persistent error state. No report output, metric, remote claim
    artifact, or planned evidence archive was observed or recovered afterward.
11. The incident, limitations, gate status, reproducible source, synthetic
    corpus, and transparent outreach draft were published on GitHub.

## V4 corpus denominator

V4 contains 6 semantic cases rendered as 12 report-layout documents across two
templates. Clean and degraded versions produce 24 rendered inputs forming 12
pairs:

- 4 development inputs
- 20 execution-held-out inputs

The report-construction gate is therefore complete and deterministic. The
overall technical gate is not complete because clinical JSON generation did
not complete.

## Why no retry was performed

V4 permits one documented development iteration. The runner is designed to
atomically create its exclusive `.claimed.json` artifact before model backend
construction. That remote artifact could not be recovered after the platform
failure.

The protocol does not separately prescribe how to classify a pre-output
development infrastructure loss when the remote claim artifact is
unrecoverable. The project therefore makes the conservative decision to treat
the observed launch as consuming the single development iteration. No retry
was attempted, and no further recovery protocol was created.

This is a protocol-conservatism decision, not proof of the exact Kaggle failure
mechanism.

## What cannot be claimed

- V4 development did not pass.
- V4 formal inference was not run.
- No v4 extraction accuracy, schema-validity, unsupported-field, or robustness
  metric exists.
- The small serializer smokes were not clinical extraction tests.
- The number of report-level model-generation calls cannot be established from
  the recovered evidence.
- Human report review was not started for v4.
- A dermatopathologist has not reviewed the schema or ambiguous cases.
- The HAI-DEF use-case form has not been submitted.
- No email has been sent to Daniel Golden.

## Outreach recommendation

The best official next step is to submit the use case through HAI-DEF and use a
developer forum or GitHub question for the reproducible runtime/document
extraction issue.

If you prefer to spend no more time unless Google responds, it is reasonable
to send one short, transparent feasibility email to Daniel Golden alone. It
must say that there is no positive pilot result, distinguish the v2, v3, and
v4 outcomes, and ask whether the narrow use case is useful before expanding to
75–100 reports.

The reviewed fallback email is in
[`docs/outreach_email.md`](docs/outreach_email.md). Do not use the original
success-oriented subject or describe the technical gate as passed.

## Published artifacts

- [Public repository branch](https://github.com/AshishSara/medgemma-melanoma-pathology-pilot/tree/agent/v3-constrained-pipeline)
- [Draft pull request #2](https://github.com/AshishSara/medgemma-melanoma-pathology-pilot/pull/2)
- [`docs/v4_development_incident.md`](docs/v4_development_incident.md)
- [`docs/gate_status.md`](docs/gate_status.md)
- [`docs/outreach_email.md`](docs/outreach_email.md)

For the preceding incident-publication commit, GitHub tree
`7bed3cb5ed1922acae834dc3daa1b35823b1d8f6` matched the locally tested tree
exactly, and its draft-pull-request CI completed successfully.

## Tooling note

Claude Code and Qwen command-line tools were not available in this workspace.
Independent factual review was instead performed by separate reviewer agents
before publication.
