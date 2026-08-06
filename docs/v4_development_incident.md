# V4 development execution incident

Date observed: 2026-08-05

Protocol: `prespecified-serialization-recovery-pilot-v4`

Protocol version: `evidence-gated-v4`

## Executive result

The v4 automated technical gate is **incomplete and did not pass**.

The exact source, corpus, runtime, OCR, model, and constrained-generation
preflights passed. The sole four-input development runner was then launched.
Immediately after the staged wrapper invoked the runner, the Kaggle notebook
session entered a persistent session-level `Error` state. The cell remained
marked as running, no report-level runner output appeared, and the remote
`/kaggle/working` filesystem stopped returning a file listing.

No development metrics or recoverable v4 report outputs were obtained. Formal
v4 inference was not launched. The incident therefore supports no claim about
v4 extraction accuracy, schema validity, evidence acceptance, unsupported
fields, or robustness to degradation.

## Reproducibility identifiers

| Item | Frozen or observed value |
|---|---|
| Source commit | `5bb993f16372c5971d97f1a5d2ae3babf385edb6` |
| Source bundle SHA-256 | `6380ef20c4e7dce5f052c010db9d05d1585d238536f1ac0b9d275eac6fde2ce4` |
| V4 corpus digest | `e1aab2bb874b58df973fceaf3dd8bcca9e9a9e2ac17f8c017b4f41a568a086fc` |
| Model | `google/medgemma-1.5-4b-it` |
| Model revision | `91850547d9f0b2fdd21aa7c5f4f3d1a8a52c243b` |
| Device used by model | `cuda:0` on Tesla T4 |
| Available accelerators | 2 × Tesla T4 |
| Dtype | `torch.bfloat16` |
| Python | `3.12.13` |
| PyTorch / CUDA | `2.10.0+cu128` / `12.8` |
| Transformers | `4.57.6` |
| Accelerate | `1.14.0` |
| LM Format Enforcer | `0.11.3` |
| Tesseract / pytesseract | `4.1.1` / `0.3.13` |
| Frozen `pip freeze` digest | `f8ad1d67471d54f23f2fe79a8eb62d5ab2bddb5ed44f656c8fd4a11aa276d1de` |

## What passed before the development launch

The following checks completed in the same Kaggle GPU runtime:

1. The uploaded Git bundle matched the frozen SHA-256 and resolved to the
   expected branch and commit.
2. The checkout had no tracked modifications, no v4 result namespace, and no
   v4 formal lock.
3. Both Tesla T4 devices were visible. PyTorch reported BF16 support and a
   BF16 CUDA matrix multiplication produced the expected value.
4. All 122 repository tests passed. Ruff lint, Ruff formatting, and
   `git diff --check` passed.
5. Tesseract read a non-corpus image labeled `PREFLIGHT ONLY`.
6. The exact pinned MedGemma revision loaded in BF16 on `cuda:0`, with no CPU
   or disk placement.
7. A non-corpus candidate-configuration smoke produced strict JSON valid
   against its small smoke schema in 8 tokens with no cap hit and confirmed
   the frozen effective settings:
   `force_json_field_order=false` and
   `max_consecutive_whitespaces=12`.
8. A non-corpus audit-configuration smoke produced strict JSON valid against
   its small smoke schema in 7 tokens with no cap hit and confirmed:
   `force_json_field_order=true` and
   `max_consecutive_whitespaces=0`.
9. The `pip freeze` output was hash-recorded, the Hugging Face token was
   removed from the process environment, and subsequent subprocesses were
   configured to use the exact cached snapshot offline.

An earlier non-corpus preflight invocation removed the token before an
optional authenticated processor-file check and received HTTP 401. No pilot
runner had been launched. The sequencing was corrected, the same pinned
snapshot was reused, and the complete non-corpus preflight then passed as
listed above.

These two serializer-configuration smokes did not exercise either full
clinical schema or any report.

## Development launch and observed failure

The staged wrapper invoked exactly:

```text
/usr/bin/python3 -u scripts/run_v4_inference.py --split development --condition all --development-iteration 1
```

After launch:

- the notebook cell displayed the running marker and the runner command;
- no subsequent runner line or report-level result appeared;
- the Kaggle session status changed to `Draft Session Error`;
- the cell remained marked as running during a final recovery window;
- the `/kaggle/working` tree could no longer be enumerated; and
- no archive from the wrapper's planned `finally` block was observed or
  recovered.

The remote development claim artifact could not be recovered after the
platform failure. V4 permits one documented development iteration, and the
runner is designed to atomically write its exclusive `.claimed.json` artifact
before backend construction. This incident record therefore conservatively
treats the observed launch as consuming that iteration, and no retry was
attempted. The protocol does not separately specify how to classify a
pre-output infrastructure loss when the remote claim artifact cannot be
recovered.

The exact mechanism of the Kaggle session failure is not proven. A GPU or
process-memory failure during backend construction is plausible, but should
not be reported as established fact without a recoverable platform log.

## Consequences

- Development status: **terminal incomplete**
- Development pass: **no**
- Development metrics: **none**
- Formal protocol lock: **not created**
- Formal inference: **not run**
- Formal ground-truth evaluation: **not run**
- Human report review: **not started**
- HAI-DEF submission: **not submitted**
- Individual Google outreach: **not recommended as a successful pilot**

V2 and v3 results remain unchanged. The v4 incident must not be used to
reinterpret either earlier protocol, and no partial v4 accuracy denominator
may be invented.

## Recommended disposition

Publish this incident record with the reproducible source and keep the
technical gate labeled incomplete. If external contact is pursued now, it
should be framed only as a transparent protocol/runtime question—not as a
successful pilot, benchmark result, or request based on measured v4
performance. The preferred order remains the HAI-DEF engagement form and a
developer forum or GitHub technical question before individual research
outreach.
