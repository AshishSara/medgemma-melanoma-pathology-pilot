# Final archival project report

Date: 2026-08-06

## Bottom line

The reproducible repository, frozen v5 protocol, synthetic corpus, and incomplete execution
evidence are public. The project is now paused.

There is **no successful confirmatory pilot result**. V5 ended before model inference because the
Colab `HF_TOKEN` received HTTP 401 while accessing the gated MedGemma repository. No model call,
semantic output, or completed qualification row occurred. The evaluator recorded
`DEVELOPMENT_INCOMPLETE`, `provenance_valid_count: 0`, and `ground_truth_read: false`.

The result is terminal incomplete and unscored—not evidence of either good or poor model
extraction quality. Formal v5 inference was not run. No success-oriented email should be sent.

## What was completed

1. A wholly synthetic melanoma-pathology extraction project was created in a dedicated Obsidian
   project folder and public Git repository.
2. Deterministic report generators, JSON ground truth, prompts, schemas, evaluators, provenance
   checks, append-only event ledgers, and regression tests were implemented.
3. Clean and degraded paired report inputs were generated across two known report templates.
4. Historical v2 inference completed for 56 held-out outputs. It produced a complete negative
   quality result: 56/56 outputs parsed as JSON, but 0/56 conformed to the frozen clinical schema.
5. V3 introduced schema-constrained two-pass extraction. Its tuned development result was
   encouraging, but formal execution stopped after 3 of 20 inputs when an audit response reached
   its frozen token cap.
6. V4 prespecified a serialization-only recovery. Its hosted GPU session failed after the
   development runner launch, before a recoverable clinical output.
7. V5 was designed as a fresh, narrower confirmatory protocol with six fresh semantic vectors,
   two known layouts, paired clean/OCR-degraded inputs, a four-input qualification, and a
   twenty-input formal set.
8. V5 underwent multiple prospective reviews. A pre-lock semantic collision was detected,
   corrected, regenerated, and disclosed before inference.
9. The v5 prospective source commit and pre-inference lock were published. The final verifier
   binds every locked artifact to the prospective source commit, execution HEAD, and working
   bytes.
10. All 175 repository tests, Ruff lint, Ruff formatting, and `git diff --check` passed before
    execution.
11. The exact Colab T4 runtime profile was captured and frozen.
12. The v5 qualification launch and its HTTP 401 evidence were preserved, evaluated as
    incomplete without reading ground truth, archived, and documented.
13. No email or HAI-DEF submission was sent.

## V5 design and denominator

V5 contains six independent synthetic clinical fact vectors:

- qualification-only: `MEL-190`
- formal-only: `MEL-201` through `MEL-205`

Each vector was rendered in layouts A and B and in clean and OCR-degraded form:

- qualification: 4 rendered inputs
- formal: 20 rendered inputs
- complete corpus: 24 rendered inputs

The 20 formal inputs represent five independent fact vectors, not twenty independent cases.

The frozen qualification thresholds were:

- 4/4 complete and provenance-valid;
- 4/4 strict candidate and audit JSON;
- 4/4 canonical final JSON;
- no generation cap hits;
- at least 58/64 exact fields overall;
- at least 28/32 exact fields per condition;
- at least 45/52 non-null fields recalled;
- 0/12 unsupported values;
- 100% valid evidence for accepted non-null values; and
- zero history carryover.

None of these semantic criteria was evaluated because no model output existed.

## V5 execution outcome

| Item | Observed result |
|---|---|
| Source commit | `3d25b8f60a9e3d37fe79195331f90866e2571654` |
| Lock commit | `509a63fc051065b2c6db2e5ef57d574036f459e8` |
| Lock SHA-256 | `fc782dc075fe2739165f0350e7e46ecd11cb39b4f6c7bb3c5b4c8ade052ffe9a` |
| Model revision requested | `91850547d9f0b2fdd21aa7c5f4f3d1a8a52c243b` |
| Backend access result | HTTP 401 from gated Hugging Face repository |
| Completed qualification rows | 0/4 |
| Candidate/audit model calls | 0 |
| Persisted model responses | 0 |
| Ground truth read | `false` |
| Evaluator status | `DEVELOPMENT_INCOMPLETE` |
| Evidence ZIP SHA-256 | `e2db5ae5dbe9c6799d67091d1fe7dcbf4bce5b050c563b19fd7b22a1b49494e4` |

The exact cause within Hugging Face authentication is not proven. Plausible explanations include
an invalid or expired token, a token belonging to a different account, or insufficient gated-repo
scope. The evidence establishes only the 401 response.

## Protocol/runner discrepancy

The v5 protocol says backend access/runtime matching must complete before an execution attempt is
claimed. The locked runner writes the execution-attempt `started` event before constructing the
backend. Consequently, the evidence contains a started/failed execution attempt even though the
model never loaded.

The record is preserved rather than erased. Any future v6 should fix this ordering before lock
and add a regression test for gated-access failure.

## Complete outcome history

| Protocol | Scope reached | Outcome |
|---|---|---|
| v2 | 56 held-out outputs | Complete negative extraction-quality result |
| v3 | Tuned development plus 3/20 formal inputs | Terminal incomplete serialization attempt |
| v4 | Runtime/model preflight and development launch | Terminal incomplete hosted-runtime attempt |
| v5 | Qualification backend-access stage; 0/4 inputs | Terminal incomplete gated-access attempt |

These outcomes are not interchangeable. V3's development result is not a confirmatory formal
result. V4 and v5 have no extraction-quality metric.

## What cannot be claimed

- No confirmatory protocol passed.
- V5 did not produce valid or invalid clinical JSON; it produced no clinical response.
- V5 has no accuracy, recall, unsupported-field, or robustness score.
- V5 formal inference was not run.
- Human report review was not performed for v5.
- A dermatopathologist has not approved the schema or reviewed ambiguous cases.
- The HAI-DEF use-case form has not been submitted.
- No email has been sent to Daniel Golden.

## Outreach decision

The project owner chose to stop rather than invest additional time in authentication recovery and
GPU execution. This is a reasonable stopping point.

Do not send a success-oriented email. The previous transparent fallback draft is retained only as
historical planning material and is marked on hold. If the project later restarts, it should
produce a new v6 result and a fresh outreach decision.

## Future restart

Use [`docs/restart_handoff.md`](docs/restart_handoff.md). The recommended path is:

1. verify gated access with a non-inference download before claiming an attempt;
2. create a prospectively reviewed v6 namespace;
3. correct the attempt/backend ordering;
4. publish source and lock before report inference;
5. run and publish a small qualification;
6. proceed to formal inference only after qualification passes; and
7. complete blinded human review before success-oriented outreach.

## Published evidence and handoff

- [Public repository branch](https://github.com/AshishSara/medgemma-melanoma-pathology-pilot/tree/agent/v3-constrained-pipeline)
- [Draft pull request #2](https://github.com/AshishSara/medgemma-melanoma-pathology-pilot/pull/2)
- [`docs/v5_access_incident.md`](docs/v5_access_incident.md)
- [`docs/gate_status.md`](docs/gate_status.md)
- [`docs/restart_handoff.md`](docs/restart_handoff.md)
- [`docs/outreach_email.md`](docs/outreach_email.md)
- `results/v5/development/`
- `results/bundles/v5-development-incomplete-509a63fc0510.zip`

## Tooling note

Claude Code and Qwen command-line tools were not installed in this workspace. Claude web was used
for prospective v5 protocol review before lock; it did not inspect or alter held-out outputs.
Independent reviewer agents also audited the frozen implementation. Neither substitutes for the
human clinical review that would have been required after a formal run.
