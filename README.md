# MedGemma melanoma pathology extraction pilot

This repository is a deterministic technical gate for testing whether MedGemma
can extract management-critical melanoma fields from heterogeneous synthetic
pathology-report images without inventing unstated values.

> Current status (2026-08-06): **the project is paused and v5 is terminal
> incomplete and unscored.** A frozen four-input v5 qualification was launched
> from the public lock commit in an exact Colab Tesla T4 runtime. Backend
> construction received HTTP 401 while requesting the gated MedGemma
> configuration. No model call, semantic output, or completed row occurred;
> the evaluator recorded `DEVELOPMENT_INCOMPLETE` with
> `ground_truth_read: false`. The project owner elected to stop rather than
> repair authentication and resume. Formal v5 inference was not run, human
> review was not started, HAI-DEF was not submitted, and no email was sent.

The repository preserves four distinct, non-interchangeable outcomes:

| Protocol | Scope reached | Outcome |
|---|---|---|
| v2 | 56 held-out model outputs | Complete negative quality result: 56/56 parse-valid, 0/56 schema-valid |
| v3 | 3 of 20 formal inputs | Terminal incomplete serialization attempt |
| v4 | Infrastructure/model preflight; development runner launch | Terminal incomplete infrastructure attempt; no v4 extraction metric |
| v5 | Qualification execution preflight; 0 of 4 assigned inputs | Terminal incomplete gated-access incident; no model call or extraction metric |

See [`docs/v5_access_incident.md`](docs/v5_access_incident.md) for the current
incident record, [`docs/restart_handoff.md`](docs/restart_handoff.md) for the
future restart boundary, and [`docs/outreach_email.md`](docs/outreach_email.md)
for the outreach hold.

For a plain-language account of the complete project, evidence, outcome, and
recommended next step, see [`FINAL_REPORT.md`](FINAL_REPORT.md).

## V3–v5 constrained follow-ups

V3 introduced two blind, schema-constrained MedGemma 1.5 calls and a
deterministic evidence compiler. Its formal attempt stopped after three of
twenty inputs when a blind-audit response reached its frozen 768-token cap
before completing valid JSON. V3 remains immutable and incomplete; see
[`docs/v3_formal_failure.md`](docs/v3_formal_failure.md).

V4 prespecified a serialization-only recovery: the clinical prompts, schemas,
OCR rules, compiler, fields, model revision, and pass thresholds remained
unchanged, while the blind-audit cap and effective serializer configuration
were corrected. Its four-input development gate did not complete because the
hosted GPU session failed after runner launch. The project conservatively
treats that launch as consuming the sole development iteration, and the
protocol permits no further v4 recovery. No v4 formal lock or formal result
exists.

V5 was a separately frozen fresh confirmatory protocol with one four-input
qualification vector and five formal vectors. Its source and pre-inference
lock were published before execution. The exact qualification runtime and
tests passed, but the runner stopped during gated-repository authentication
before loading the model. The preserved evidence contains a started/failed
execution attempt, zero call events, zero outputs, and an incomplete evaluator
payload that did not read ground truth. V5 is closed without a score; any
future restart should use a prospectively reviewed v6 namespace.

The v3–v5 corpora can be regenerated and checked without model access:

```bash
uv run python scripts/generate_v3_reports.py
uv run python scripts/generate_v4_reports.py
uv run python scripts/generate_v5_reports.py
uv run pytest tests/test_v3_corpus.py tests/test_v3_pipeline.py \
  tests/test_v3_inference.py tests/test_v4_corpus.py \
  tests/test_v4_inference_regression.py \
  tests/test_v4_evaluator_regression.py tests/test_v4_protocol.py \
  tests/test_v5_corpus.py tests/test_v5_inference.py \
  tests/test_v5_evaluator.py tests/test_v5_protocol.py
```

## Research question

Can MedGemma 1.5 accurately extract specimen site and laterality, Breslow
thickness, ulceration, mitotic rate, margin status, and explicitly reported
staging elements from heterogeneous pathology-report PDFs without inventing
unstated values, and how robust is extraction to bounded raster/OCR
degradation?

The defensible novelty is the combination of narrative melanoma management
fields, explicit abstention targets, unsupported-field measurement, and paired
layout/degradation testing. Synthetic PDFs or layout variation alone are not
novel: the MedGemma 1.5 report describes a synthetic custom-PDF training and
evaluation dataset.

## Historical v2 pilot design

- Eight semantic cases are rendered through two original report templates,
  producing 16 wholly synthetic, single-page report-layout documents.
- Every document has a clean text PDF and a matched image-only degraded PDF,
  producing 32 rendered inputs in the complete corpus.
- `MEL-001` (two layouts and both renderings) is quarantined as development
  data. Its two clean layouts were used to select the prompt and response
  constraint; none of its four inputs enters a formal metric.
- The held-out set is `MEL-002` through `MEL-008`: 7 semantic cases, 14
  report-layout documents, and 28 rendered inputs forming 14 clean/degraded
  pairs.
- The primary technical gate is MedGemma 1.5 4B IT. MedGemma 1 4B IT is a
  secondary shared-protocol version comparator.
- The formal matrix is 28 inputs x 2 models = 56 outputs.
- Primary metrics are 16-field exact match/F1 and unsupported-field rate.
  Secondary metrics include strict JSON/Schema validity, complete-report
  accuracy, paired clean-to-degraded change, and results by template and
  condition.
- Every formal output must be visually reviewed by a human before outreach.
  An AI-assisted audit may prepare the ledger but does not satisfy that gate.

This is a manually prompt-optimized, no-example feasibility pilot—not an
unqualified stock-prompt or fine-tuned evaluation.

## Historical v2 formal result

| Model | Parse-valid | Schema-valid | Field exact | Non-null micro-F1 | Unsupported-field rate |
|---|---:|---:|---:|---:|---:|
| MedGemma 1.5 4B IT | 28/28 | 0/28 | 178/448 (39.7%) | 40.0% | 14/172 (8.1%) |
| MedGemma 1 4B IT | 28/28 | 0/28 | 230/448 (51.3%) | 52.0% | 47/172 (27.3%) |

Both models returned the correct document ID in 28/28 outputs and achieved
0/28 complete-report accuracy. For MedGemma 1.5, mean field exact match
decreased by 1.8 percentage points under bounded degradation; the comparator's
mean exact-match change was 0.0 points. Codex reviewer agents visually audited
all 28 unique source images and all 56 raw responses. Every output-level audit
is bound to the raw-output SHA-256 in `results/manual_review.csv` and labeled
`ai_audited`; human verification remains pending.

Frequent schema failures included noncanonical categorical strings, `null`
where the schema requires structured `margins` or `staging` objects, and
string-valued numerics. The deliberately separated prior-report case also
exposed unsupported historical-value carryover in the comparator. See
[`docs/results.md`](docs/results.md) for the full interpretation and
limitations.

The compact AI-audited evidence archive is
[`results/bundles/formal-results-32f1453e-reviewed.tgz`](results/bundles/formal-results-32f1453e-reviewed.tgz).
Its adjacent `.sha256` file records the archive digest. The archive contains
all raw responses, model-only continuations, parsed objects, run records,
runtime metadata, aggregate metrics, and the hash-bound audit ledger.

Verify the archive from the repository root with:

```bash
(cd results/bundles && \
  shasum -a 256 -c formal-results-32f1453e-reviewed.sha256)
```

## Historical v2 frozen inference behavior

The `heldout-pilot-v2` protocol uses:

- exact pinned model revisions recorded in `results/run_manifest.json`;
- BF16 on a Tesla T4;
- greedy decoding (`do_sample=False`) with a 1,200-token ceiling;
- `prompts/extraction_prompt.txt`;
- a literal one-character assistant prefix, `{`, supplied through
  Transformers `continue_final_message=True`.

The strict scored response is the assembled assistant message, including the
harness-supplied opening brace. The generated continuation is retained
separately. Raw bytes, hashes, image/prompt/schema hashes, model revision,
device, dtype, dependency versions, and generation settings are recorded for
every call.

Parsing may trim surrounding whitespace and remove at most one complete outer
Markdown fence. It does not extract a later brace-delimited object, strip a
reasoning envelope, insert or drop keys, coerce values, map synonyms, infer
staging, or otherwise repair model output.

## Reproduce the materials

Python 3.9+ and [`uv`](https://docs.astral.sh/uv/) are recommended.

```bash
uv sync --extra dev
uv run python scripts/validate_artifacts.py
uv run pytest
```

To regenerate the synthetic corpus:

```bash
uv run python scripts/generate_reports.py
uv run python scripts/validate_artifacts.py
```

The generator uses fixed data, dates, and perturbation parameters. Model inputs
are the committed PNGs under `output/rendered/`; their SHA-256 values are frozen
in `data/report_manifest.csv`. Clean PDFs are under `output/pdf/clean/`, and
degraded image-only PDFs are under `output/pdf/ocr_degraded/`.

## Run the historical v2 held-out inference matrix

The local backend requires accepted HAI-DEF terms, authenticated Hugging Face
access, and the optional inference dependencies:

```bash
uv sync --extra inference

uv run python scripts/run_inference.py \
  --backend transformers \
  --model google/medgemma-1.5-4b-it \
  --revision 91850547d9f0b2fdd21aa7c5f4f3d1a8a52c243b \
  --condition all \
  --scope formal \
  --response-mode json-prefill \
  --dtype bfloat16 \
  --max-new-tokens 1200

uv run python scripts/run_inference.py \
  --backend transformers \
  --model google/medgemma-4b-it \
  --revision 290cda5eeccbee130f987c4ad74a59ae6f196408 \
  --condition all \
  --scope formal \
  --response-mode json-prefill \
  --dtype bfloat16 \
  --max-new-tokens 1200

uv run python scripts/evaluate.py
uv run python scripts/validate_artifacts.py
```

In a full checkout or after unpacking the reviewed evidence archive, formal raw
responses are under `results/raw/`, model-only continuations under
`results/generated_continuations/`, parsed objects under `results/normalized/`,
and provenance records under `results/run_records/`. Development runs are
routed under `results/development/` and cannot enter the formal evaluator.

An OpenAI-compatible endpoint remains available for separately labeled direct
exploration, but it is not equivalent to the frozen Transformers JSON-prefill
protocol.

## Gate status

See [`docs/gate_status.md`](docs/gate_status.md). Publication is a
reproducibility artifact, not evidence that the technical gate passed. The
original v2 pilot completed with a negative extraction-quality result; v3,
v4, and v5 are terminal incomplete attempts with no replacement formal
metric. The project is paused. No success-oriented outreach is warranted.

## Safety and scope

All reports are synthetic and visibly watermarked. No real patient data are
used. This is a research evaluation, not a medical device, diagnostic system,
or clinical recommendation. A qualified dermatopathologist should approve the
schema before expansion and review genuinely ambiguous cases.

## Evidence

- [MedGemma 1.5 technical report](https://arxiv.org/abs/2604.05081)
- [MedGemma 1.5 model card](https://developers.google.com/health-ai-developer-foundations/medgemma/model-card)
- [Official MedGemma get-started and engagement paths](https://developers.google.com/health-ai-developer-foundations/medgemma/get-started)
- [Current CAP cancer protocol index](https://www.cap.org/protocols-and-guidelines/cancer-protocols/current-cancer-protocols/)

MedGemma weights remain governed by the HAI-DEF terms. This repository's
original code and synthetic artifacts are MIT-licensed and do not redistribute
model weights.
