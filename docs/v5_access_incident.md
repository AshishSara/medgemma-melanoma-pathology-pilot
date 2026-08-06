# V5 qualification access incident and terminal disposition

Date observed: 2026-08-06

Protocol: `fresh-heldout-confirmatory-pilot-v5`

Protocol version: `fresh-confirmatory-v5`

## Executive result

V5 is **terminal incomplete and unscored**. It is not a positive pilot and it is not a negative
model-quality result.

The frozen four-input qualification was launched from the public lock commit in a Google Colab
Tesla T4 runtime. Source, lock, runtime, package, OCR, and repository tests passed before the
runner launch. Backend construction then received HTTP 401 while requesting the exact pinned
MedGemma configuration from the gated Hugging Face repository.

The failure occurred before the model loaded, before any report-level candidate or audit call,
and before any semantic output. The evaluator recorded `DEVELOPMENT_INCOMPLETE` with
`ground_truth_read: false` and `provenance_valid_count: 0`.

The project owner elected to stop rather than repair authentication and resume. Formal v5
inference was never permitted or launched.

## Reproducibility identifiers

| Item | Frozen or observed value |
|---|---|
| Prospective source commit | `3d25b8f60a9e3d37fe79195331f90866e2571654` |
| Public pre-inference lock commit | `509a63fc051065b2c6db2e5ef57d574036f459e8` |
| Protocol-lock SHA-256 | `fc782dc075fe2739165f0350e7e46ecd11cb39b4f6c7bb3c5b4c8ade052ffe9a` |
| Evidence archive SHA-256 | `e2db5ae5dbe9c6799d67091d1fe7dcbf4bce5b050c563b19fd7b22a1b49494e4` |
| Qualification case | `MEL-190` |
| Assigned qualification inputs | 4: layouts A/B × clean/OCR-degraded |
| Model | `google/medgemma-1.5-4b-it` |
| Model revision | `91850547d9f0b2fdd21aa7c5f4f3d1a8a52c243b` |
| Runtime | Google Colab, one Tesla T4 |
| Evaluator status | `DEVELOPMENT_INCOMPLETE` |
| Ground truth read | `false` |
| Completed new rows | `0` |
| Model call-start events | `0` |
| Candidate/audit responses | `0` |

## Observed chronology

1. The public repository was cloned and detached at the lock commit.
2. The committed lock matched SHA-256
   `fc782dc075fe2739165f0350e7e46ecd11cb39b4f6c7bb3c5b4c8ade052ffe9a`.
3. The exact frozen runtime profile was verified: Python 3.12.13, PyTorch 2.11.0+cu128,
   CUDA 12.8, one Tesla T4 with 14,912 MiB, Transformers 4.57.6, Accelerate 1.14.0,
   LM Format Enforcer 0.11.3, pytesseract 0.3.13, and Tesseract 4.1.1.
4. All 175 repository tests passed in the Colab runtime.
5. The runner wrote an execution-attempt `started` event at
   `2026-08-06T14:49:31.043240+00:00`.
6. During backend construction, Hugging Face returned HTTP 401 for the pinned
   `config.json` URL. The runner wrote a `failed` event at
   `2026-08-06T14:49:47.340772+00:00`.
7. No per-report call ledger, response bundle, parsed object, normalized result, or run record
   was created.
8. The evaluator wrote an incomplete payload at
   `2026-08-06T14:49:52.039865+00:00` without deserializing ground truth.
9. The complete incomplete-result namespace was archived and downloaded.
10. The project owner chose to stop the project and preserve a restart handoff.

## Exact failure classification

The failed event records:

```text
error_type: OSError
HTTP status: 401
failure stage: gated model configuration access during backend construction
completed_new_row_count: 0
current_row: null
```

This establishes an authentication/access failure. It does not establish whether the Colab
secret was expired, invalid, scoped to a different account, or lacked access to the gated model.
No token value is recorded.

## Protocol/implementation sequencing note

The frozen protocol states that the backend must load and match the runtime profile before an
execution attempt is claimed. The locked runner verifies the lock and manifest, writes the
execution-attempt `started` event, and then constructs the backend. Therefore the preserved
ledger contains a started/failed execution attempt even though backend loading never completed.

This sequencing discrepancy was not used to erase or relabel the event. It is disclosed here and
should be corrected prospectively before any future v6 lock.

## Consequences

- V5 qualification status: **terminal incomplete**
- V5 qualification pass: **no**
- V5 extraction-quality score: **none**
- V5 formal inference: **not run**
- V5 human review: **not started**
- Positive outreach claim: **not permitted**
- Email to Daniel Golden: **not sent**
- HAI-DEF submission: **not submitted**

The v2 negative result and the v3/v4 terminal-incomplete outcomes remain unchanged.

## Preserved evidence

- `results/v5/development/metrics.json`
- `results/v5/development/execution_attempts/`
- `results/bundles/v5-development-incomplete-509a63fc0510.zip`
- `results/bundles/v5-development-incomplete-509a63fc0510.zip.sha256`

The ZIP is the original Colab download. The extracted JSON files are retained for direct review.
