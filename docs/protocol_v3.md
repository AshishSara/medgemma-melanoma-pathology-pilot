# Fresh held-out hybrid-pipeline protocol v3

Protocol identifier: `fresh-heldout-pilot-v3`

Status: **development iteration 2 passed; formal configuration awaiting its
immutable lock and one-shot execution**. No formal model output or formal
ground truth has been inspected.

## Claim and objective

This protocol evaluates a hybrid document-extraction pipeline that uses
MedGemma 1.5, OCR assistance, two independently schema-constrained model
passes, and a deterministic evidence gate.

It does **not** evaluate raw MedGemma generation as a standalone extractor.
Any result must be described as performance of the complete hybrid pipeline.
Raw candidate and audit responses remain useful diagnostic artifacts, but the
prespecified endpoint is the final evidence-gated canonical JSON.

The objective is to determine whether this pipeline can extract 16
management-critical melanoma pathology fields from synthetic clean and
OCR-degraded report images while abstaining when a current-specimen value is
not supported.

## Fresh corpus and split

The v3 corpus is separate from the v2 evaluation corpus. No v2 development or
formal case is reused.

| Split | Semantic cases | Report layouts | Rendered inputs |
|---|---:|---:|---:|
| Development | `MEL-101` through `MEL-103` | 6 | 12 |
| Formal held-out | `MEL-104` through `MEL-108` | 10 | 20 |
| Total | 8 | 16 | 32 |

Every semantic case is rendered in both report layouts:

- Template A: compact synoptic report
- Template B: narrative report

Every report-layout document has two paired render conditions:

- `clean`
- `ocr_degraded`

The formal matrix therefore contains five semantic cases, ten report-layout
documents, and twenty rendered inputs: ten clean and ten degraded. These are
paired variants, not twenty independent clinical cases.

The formal set includes explicit missingness, minimum and approximate numeric
qualifiers, mixed margin states, unstated laterality and staging, and a
prior-report history distractor. Formal ground truth contains 124 null-field
opportunities across the 20 inputs and 16 scored fields.

## Scored fields

The 16 clinical fields are:

1. specimen site
2. laterality
3. diagnosis
4. Breslow thickness
5. Breslow qualifier
6. ulceration
7. mitotic rate
8. mitotic qualifier
9. invasive peripheral margin
10. invasive deep margin
11. in-situ peripheral margin
12. in-situ deep margin
13. pT
14. pN
15. pM
16. stage group

Document ID is required for artifact binding and schema validity but is not
included in the 16-field pooled clinical exact-match denominator.

Only values explicitly supported for the current specimen may be non-null.
The pipeline must not calculate staging, convert missing mitotic rate to zero,
or copy a measurement or stage from clinical history or a prior specimen.

## Pipeline under evaluation

The only model used in v3 is `google/medgemma-1.5-4b-it`, requested at commit
`91850547d9f0b2fdd21aa7c5f4f3d1a8a52c243b`. The development run must confirm
that the resolved revision matches this pin; that resolved value is then
included in the formal lock.

### OCR-assisted input

Each committed report image is processed by Tesseract through `pytesseract`
with language `eng`, configuration `--oem 1 --psm 6`, a 120-second timeout,
and no image preprocessing. The word-level `pytesseract.Output.DICT` result is
retained in the OCR artifact. Words are grouped in first-seen
page/block/paragraph/line order, assigned stable `L001` through `Lnnn` IDs,
and deterministically tagged as header, current-specimen, history, or
footer/comment. Explicit current-specimen and synoptic headings start the
current section. Known labeled rows may use a colon, a spaced dash, or
whitespace between the label and value so that flattened table OCR is treated
the same as narrative label-value text.

The image and numbered OCR transcript are supplied together to both model
passes. OCR is an assistive observation, not ground truth.

Development recorded Tesseract 4.1.1 and `pytesseract` 0.3.13. These exact
versions and the unchanged settings above are frozen in the formal lock.

### Pass 1: extraction candidate

MedGemma 1.5 receives the report image, numbered OCR transcript, frozen
candidate prompt, and candidate response schema. It returns one extraction
candidate covering the complete target structure.

The pass is constrained with LM Format Enforcer 0.11.3 using a fresh
`JsonSchemaParser`, forced JSON field order, and Transformers
`prefix_allowed_tokens_fn`. Generation uses direct `model.generate`, no
assistant prefill, greedy decoding, one beam, random seed 0, BF16, and a
512-token maximum. Its response must conform to the committed candidate
schema.

### Pass 2: blind presence audit

A separate MedGemma 1.5 call receives the same report image and numbered OCR
transcript, the frozen audit prompt, and audit response schema.

The audit pass does not receive or inspect the extraction candidate. For every
field it reports:

- `present`, with the smallest one or two supporting OCR line IDs;
- `absent`, with no evidence line IDs; or
- `unclear`, with relevant line IDs when available.

This pass uses the same schema-enforcement and decoding mechanism with a fresh
parser and a 768-token maximum.

### Deterministic evidence gate and canonical compiler

The deterministic compiler receives only the candidate response, blind audit
response, and numbered OCR lines. It does not receive ground truth.

Except for the three linked binding/normalization rules below, a non-null
clinical candidate value is accepted only when all of the following hold:

1. the candidate value can be deterministically canonicalized;
2. the blind audit marks the field `present`;
3. the audit cites one or two known, non-duplicated OCR line IDs;
4. cited evidence is in an allowed non-history section;
5. deterministic parsing of the cited text produces one unambiguous canonical
   value; and
6. that value exactly agrees with the canonicalized candidate.

The compiler has three narrowly prespecified linked rules:

1. `document_id` is an artifact-binding field, not one of the 16 scored
   clinical fields. If its blind-audit entry is absent or unclear, it may still
   be bound only when the non-null candidate identifier exactly agrees with
   exactly one identifier parsed from a labeled report-header line.
2. Specimen site and laterality are linked components of one site statement.
   If accepted, non-history specimen-site evidence contains an explicit
   laterality token, a non-null candidate laterality that exactly agrees with
   that token may reuse the same citation when its own audit entry is absent or
   unclear. A missing laterality token can never become a non-null value.
3. Each numeric measurement and qualifier is a linked pair. When a candidate
   qualifier is null or omitted, the numeric value has already passed the
   ordinary gate, and deterministic parsing of the same accepted,
   non-history evidence yields exactly one qualifier, the compiler derives
   `at_least` for prespecified lower-bound wording, `approximate` for
   prespecified approximate Breslow wording, or `exact` for an otherwise
   unmodified numeric statement. Unrecognized, damaged, ambiguous, or
   conflicting modifier text causes joint abstention.

For specimen site only, exact candidate/evidence agreement may also tolerate
one aligned substitution between a fixed pair of commonly confused OCR glyphs
in one token. Insertions, deletions, reordered tokens, multiple substitutions,
and non-whitelisted character pairs fail closed. This is evidence validation,
not dictionary correction.

Any other non-null candidate that fails the ordinary gate becomes `null`.
Evidence from a history section is always rejected. Numeric values and
qualifiers are jointly abstained when their pair remains incoherent. Raw
candidate and blind-audit decisions are retained as separate diagnostics. The
compiler validates the final object against the unchanged canonical extraction
schema and records every ordinary or linked decision in a field-level log.

The compiler is a verifier and serializer, not a second extractor. It may not
consult ground truth or infer missing clinical facts.

## Development budget

Only `MEL-101` through `MEL-103` may be used for pipeline development.
Development outputs and metrics must remain outside every formal aggregate.

At most **two documented development iterations** are permitted before the
formal freeze. An iteration is one coherent, versioned pipeline configuration
evaluated on the declared development inputs. Each iteration record must
include:

- inputs used;
- OCR, prompt, schema, compiler, model, and runtime configuration;
- hashes of all prompts and schemas;
- raw candidate and audit responses;
- final compiled outputs and development metrics;
- changes made from the preceding iteration; and
- the rule used to select or reject the iteration.

All unsuccessful development attempts must be preserved. Configuration may
not be tuned on any `MEL-104` through `MEL-108` model output. After the second
iteration, development stops and one configuration is either frozen or the v3
formal run is abandoned.

### Documented development revisions

- **Iteration 1:** The first `MEL-101-A` clean input produced schema-valid
  candidate and blind-audit responses, but compilation failed before a
  canonical output. The preserved failure showed that a Template A synoptic
  table was section-tagged as header, colonless label-value OCR was not parsed,
  and the two passes omitted some linked normalized subfields. Execution
  stopped before the remaining development inputs.
- **Iteration 2:** Corrected the development-observed section labeling and
  delimiter handling; clarify field-by-field, laterality, and numeric
  qualifier instructions; and prespecify the narrow document-ID,
  site/laterality, measurement/qualifier, and bounded site-OCR rules above.
  Candidate and audit schema shapes, corpus, model revision, inference
  settings, formal set, and pass thresholds remain unchanged. No formal model
  output or formal ground truth was inspected. The complete 12-input
  development matrix passed with 182/192 clinical fields exact (94.79%),
  12/12 canonical-schema-valid outputs, 86/86 accepted non-null values backed
  by valid non-history evidence, 0/96 unsupported values, and zero history
  carryover. The raw candidate pass alone scored 167/192 (86.98%) with 8/12
  candidate outputs satisfying the canonical schema and 9/96 unsupported
  values; the passing endpoint is therefore the full hybrid pipeline, not raw
  MedGemma.

## Ground-truth firewall

Formal ground truth may be generated and structurally validated as a corpus
artifact, but it is unavailable to the operational extraction pipeline.

During OCR, prompt assembly, both model passes, evidence compilation, and
final-output sealing:

- runtime code must not read `data/v3/ground_truth/`;
- runtime code must not derive target facts from `data/v3/cases.csv`;
- ground-truth values must not appear in prompts, OCR corrections, caches, or
  fallback logic; and
- the compiler must operate only on candidate, audit, OCR, schema, and frozen
  configuration artifacts.

Formal evaluation code may read ground truth only after all 20 final outputs
and their provenance records have been produced and hash-sealed. Tests that
validate corpus construction must remain separate from the inference entry
point.

## Freeze requirements

Before the first formal model call, the repository must contain a lock record
that resolves every item below:

- repository commit and protocol identifier;
- v3 case table, manifest, image, PDF, and ground-truth hashes;
- exact MedGemma 1.5 requested and resolved revision;
- OCR engine, revision/version, and complete OCR settings;
- candidate and audit prompt hashes;
- candidate, audit, and canonical schema hashes;
- inference backend and relevant runtime versions;
- device and numerical dtype;
- schema-enforcement method for each model pass;
- decoding, stopping, token-limit, and seed settings;
- compiler version and source hash;
- formal execution order;
- exact permitted infrastructure-retry rule;
- the two development-iteration records or an explicit statement that fewer
  were used; and
- all pass criteria in this document.

The development runtime recorded Python 3.12.13, PyTorch 2.11.0+cu128,
Transformers 4.57.6, Accelerate 1.14.0, LM Format Enforcer 0.11.3,
`pytesseract` 0.3.13, Tesseract 4.1.1, CUDA 12.8, and a Tesla T4 using
`torch.bfloat16`. The lock records these values, the two documented
development iterations, and hashes for every bound artifact. No formal call
may occur while any required lock field remains unresolved. The inference
entry point fails closed when the lock is absent or any locked source, corpus,
configuration, or enforced runtime value differs.

Changing a frozen prompt, schema, OCR configuration, model revision, compiler,
evidence rule, pass threshold, or scored corpus after formal inference begins
invalidates the v3 run. Continuing would require a new protocol version and a
fresh formal set.

## Formal one-shot execution

Each of the 20 formal inputs receives exactly one candidate call and one blind
audit call under the frozen configuration. There is no semantic retry,
best-of-N selection, majority vote, or output repair by a language model.

There is no automatic retry. After an infrastructure failure in which no
valid model response was obtained, execution may resume only for unchanged,
missing rows under the same lock after the failed attempt is preserved in the
append-only execution-attempt ledger. The runner exits on the first failure;
resumption requires a new manual invocation over the full manifest. Completed
rows are skipped only after their lock-bound provenance and every artifact hash
are verified. A started attempt without exactly one matching failed or completed
terminal event blocks resumption because retry eligibility cannot be proven.

A valid candidate response is persisted immediately. If a later stage fails,
that immutable partial row and its failed-attempt record prohibit another model
call for the row. A semantic/schema failure is likewise not retryable. Only a
failure during OCR or candidate generation before a valid model response is
obtained can leave a retry-eligible missing row. No row artifact is ever
overwritten.

The orchestration layer may check completion, schema, and provenance during
execution, but no formal accuracy result may be used to change the pipeline.
Aggregate and field-level accuracy evaluation begins only after all final
outputs are hash-sealed.

## Artifact provenance

For every formal input, the retained evidence chain must include:

- manifest row and committed PDF/image hashes;
- raw OCR output and hash;
- numbered, section-tagged OCR transcript and hash;
- OCR engine and configuration;
- rendered candidate and audit prompts and hashes;
- candidate and audit schema hashes;
- raw and parsed candidate response with hashes;
- raw and parsed blind-audit response with hashes;
- requested and resolved MedGemma 1.5 revision;
- inference backend, runtime versions, device, dtype, and frozen generation
  settings;
- timestamps, elapsed time, and attempt status;
- append-only execution-attempt start/failure/completion events, including the
  failed stage and valid-response count;
- compiler version and source hash;
- complete field-level evidence-gate audit log;
- final canonical JSON and hash;
- final canonical-schema validation result; and
- repository commit and protocol-lock hash.

Provenance validation must fail closed when a required artifact is missing,
unreadable, stale, or hash-mismatched. Authentication tokens and other secrets
must never be written to artifacts.

## Metrics and pass criteria

The formal v3 gate passes only if **all** criteria below are satisfied:

1. **Completion and provenance:** 20/20 formal rows are complete and
   provenance-valid, with the output bound to the intended manifest document.
2. **Canonical schema:** 20/20 final outputs, or 100%, validate against the
   canonical extraction schema.
3. **Evidence acceptance:** 100% of non-null values in final outputs have a
   successful evidence-gate decision supported by valid non-history evidence.
4. **Pooled field exactness:** exact match across the 16 clinical fields is at
   least 90% over all 20 inputs. The denominator is 320 field instances, so at
   least 288 must be correct.
5. **Condition field exactness:** clean exact match is at least 85% and
   `ocr_degraded` exact match is at least 85%. Each denominator is 160 field
   instances, so each condition requires at least 136 correct.
6. **Unsupported-field rate:** unsupported non-null predictions divided by
   ground-truth-null opportunities are at most 2%. The frozen corpus has 124
   such opportunities, so no more than two unsupported values are permitted.
7. **History carryover:** zero final values may be copied from prior-report or
   history text when that value is absent from or contradicted by the current
   specimen.

All thresholds are conjunctive. Strong performance on one criterion cannot
compensate for failure on another. The unsupported-field rate must be reported
beside non-null recall or micro-F1 so an all-null output cannot appear safe.

Also report, without substituting them for the gate:

- document-ID accuracy;
- complete-report accuracy;
- precision, recall, and F1 for non-null extraction;
- per-field results;
- results by template and condition;
- paired degraded-minus-clean change;
- candidate and blind-audit schema-validity diagnostics; and
- counts and reasons for evidence-gate rejection.

## Review status and interpretation

The blind audit pass is a MedGemma model pass. It is not a human review.
Likewise, a later AI-assisted artifact audit may be recorded as `ai_audited`,
but it must never be reported as `human_verified`.

Human verification status must be tracked separately and bound to the exact
final-output and evidence hashes. Before external outreach describes the
formal artifacts as manually verified, a human must inspect the relevant
source image, current-versus-history distinction, final JSON, and cited
evidence for all 20 formal rows.

Passing v3 would support only a bounded claim: this frozen hybrid pipeline met
the prespecified thresholds on five fresh synthetic semantic cases across ten
paired report layouts and twenty rendered inputs. It would not establish raw
MedGemma accuracy, clinical validity, generalization to unseen institutions or
layouts, or readiness for patient care.
