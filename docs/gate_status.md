# Technical gate status

Last updated: 2026-08-06

Overall disposition: **project paused; v5 terminal incomplete and unscored**.

| Gate | Evidence required | Status |
|---|---|---|
| 1. Create 10–20 reports from two templates | Frozen v5 corpus: 6 fact vectors, 12 report-layout documents, and 24 paired rendered inputs; 4 qualification and 20 formal | **Complete and deterministic** |
| 2. Confirm MedGemma 1.5 access and inference | Exact gated-repository access plus model load in the frozen Colab runtime | **Incomplete for v5:** runtime passed, but the Colab token received HTTP 401 before model load |
| 3. Produce valid JSON for clean and degraded inputs | Complete and pass the four-input v5 qualification before formal inference | **Terminal incomplete:** 0/4 inputs completed; no candidate or audit model call occurred |
| 4. Human-verify every formal output | Human review bound to all completed formal outputs | **Not started:** no v5 formal outputs exist |
| 5. Publish reproducible repository | Public source, lock, corpus, evidence, and honest outcome record | **Complete on the public branch** |
| 6. Submit through HAI-DEF | Submission confirmation | **Not submitted; project paused** |

## Current v5 disposition

The v5 prospective source and pre-inference lock were published before report-level execution:

- source commit: `3d25b8f60a9e3d37fe79195331f90866e2571654`
- lock commit: `509a63fc051065b2c6db2e5ef57d574036f459e8`
- lock SHA-256:
  `fc782dc075fe2739165f0350e7e46ecd11cb39b4f6c7bb3c5b4c8ade052ffe9a`

In Colab, the lock, exact T4 runtime, packages, OCR stack, and 175 tests passed. The runner then
received HTTP 401 while requesting the exact pinned MedGemma `config.json`. The failure occurred
during backend construction:

- completed rows: `0`
- current row: `null`
- candidate/audit call events: `0`
- response bundles: `0`
- evaluator status: `DEVELOPMENT_INCOMPLETE`
- ground truth read: `false`

The user elected to stop rather than replace the token and resume. V5 is therefore closed as
terminal incomplete without a quality score. Formal inference was neither allowed nor launched.

See [`v5_access_incident.md`](v5_access_incident.md) for the exact incident and evidence
identifiers.

## Complete protocol history

| Protocol | Outcome | Interpretation |
|---|---|---|
| v2 | Complete negative result | 56/56 outputs parse-valid, 0/56 schema-valid; extraction-quality gate failed |
| v3 | Terminal incomplete | Tuned development was encouraging; formal stopped after 3/20 inputs on serialization cap |
| v4 | Terminal incomplete | Hosted GPU session failed after development runner launch; no recoverable clinical output |
| v5 | Terminal incomplete | Gated-repository HTTP 401 before model load; no model call or clinical output |

No later protocol converts or replaces any earlier outcome.

## Evidence status

The original Colab ZIP and extracted v5 JSON evidence are committed under `results/`. The archive
SHA-256 is:

```text
e2db5ae5dbe9c6799d67091d1fe7dcbf4bce5b050c563b19fd7b22a1b49494e4
```

The evaluator did not deserialize ground truth because required response artifacts were absent.
The incomplete metrics must not be converted into a zero-accuracy score: no inference output
existed to score.

## Outreach hold

Do not describe any protocol as a successful confirmatory pilot. Do not send the original
success-oriented email. No email has been sent to Daniel Golden and no HAI-DEF submission has
been made.

If the project is restarted, follow [`restart_handoff.md`](restart_handoff.md), create a
prospectively reviewed v6, and require a public qualification pass plus completed blinded human
review before success-oriented outreach.
