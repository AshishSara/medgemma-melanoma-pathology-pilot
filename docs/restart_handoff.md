# Project restart handoff

Last updated: 2026-08-06

## Current stopping point

The project is paused after the v5 qualification launch ended before model inference because the
Colab `HF_TOKEN` received HTTP 401 for the gated MedGemma repository. V5 is closed as terminal
incomplete and unscored. Do not silently resume or overwrite its result namespace.

No email was sent, no HAI-DEF use case was submitted, formal v5 inference was not run, and no
human review began.

## Authoritative locations

- Public repository:
  `https://github.com/AshishSara/medgemma-melanoma-pathology-pilot`
- Working branch: `agent/v3-constrained-pipeline`
- Draft pull request:
  `https://github.com/AshishSara/medgemma-melanoma-pathology-pilot/pull/2`
- OneDrive Obsidian project:
  `Obsidian Vaults/Liver Digital Twins Research/MedGemma Melanoma Pathology Extraction Pilot`

Use a fresh Git clone for execution. The OneDrive copy is a documentation mirror and may have
stale embedded Git metadata.

## Immutable v5 identifiers

- Prospective source commit:
  `3d25b8f60a9e3d37fe79195331f90866e2571654`
- Pre-inference lock commit:
  `509a63fc051065b2c6db2e5ef57d574036f459e8`
- Protocol-lock SHA-256:
  `fc782dc075fe2739165f0350e7e46ecd11cb39b4f6c7bb3c5b4c8ade052ffe9a`
- Incomplete evidence ZIP SHA-256:
  `e2db5ae5dbe9c6799d67091d1fe7dcbf4bce5b050c563b19fd7b22a1b49494e4`

Do not modify any v5 lock-bound source and continue calling it the same frozen protocol.

## Recommended future restart: create v6 prospectively

1. Read `FINAL_REPORT.md`, `docs/gate_status.md`, and `docs/v5_access_incident.md`.
2. Clone the public branch and verify the archived evidence checksums.
3. Confirm gated model access outside the runner:
   - use the same Hugging Face user account that accepted model access;
   - create a new read token or a fine-grained token explicitly authorized for the model;
   - call `whoami` and download the exact pinned `config.json` without printing the token.
4. Decide whether the project is worth restarting before spending GPU time.
5. If restarting, create a new v6 protocol and namespace. Carry forward the v5 corpus only if the
   new protocol explicitly discloses that those artifacts were previously generated and exposed
   but never used in model inference.
6. Fix the runner sequence before the v6 lock so backend access/runtime validation completes
   before the execution-attempt `started` event.
7. Add a regression test proving a gated-access failure cannot claim a report-level attempt.
8. Re-run independent protocol review before freezing v6.
9. Publish the prospective source commit and lock before any report-level inference.
10. Run the small qualification. Proceed to formal inference only after a public qualification
    pass.
11. Complete blinded human review before any success-oriented outreach.

## What must remain transparent

- V2 completed with a negative quality result.
- V3 had an encouraging tuned development result but incomplete formal execution.
- V4 was terminal incomplete after a hosted runtime incident.
- V5 was terminal incomplete before model inference because of gated-access authentication.
- There is no successful confirmatory pilot.

## External engagement state

The original outreach target and proposed use case are preserved in `docs/outreach_email.md`, but
the current recommendation is not to send an email. If the work restarts and passes its locked
technical and human-review gates, draft a new email using the measured result rather than reviving
an earlier success-oriented draft.
