# Serialization-only recovery protocol v4

Protocol identifier: `prespecified-serialization-recovery-pilot-v4`

Protocol version: `evidence-gated-v4`

Status: **prespecified pre-execution protocol; its immutable outcome is
recorded separately**. V3 remains an immutable incomplete attempt and cannot
be resumed or reported as a pass.

## Recovery scope

V4 is a narrow recovery protocol for the blind-audit serialization failure
documented in `docs/v3_formal_failure.md`. It does not change the clinical
extraction task to improve accuracy.

The following v3 components remain byte-identical and are reused by path:

- candidate prompt: `prompts/v3/candidate_prompt.txt`
- blind-audit prompt: `prompts/v3/audit_prompt.txt`
- candidate schema: `schema/v3/candidate.schema.json`
- blind-audit schema: `schema/v3/audit.schema.json`
- canonical schema: `schema/extraction.schema.json`
- OCR settings and line sectioning
- candidate decoding settings and 512-token cap
- deterministic compiler and every evidence-acceptance rule
- 16 scored clinical fields
- formal pass thresholds
- model ID and pinned model revision

The only inference changes are serialization safeguards:

1. the blind-audit cap increases from 768 to 1280 generated tokens;
2. for the blind-audit pass only, after LM Format Enforcer constructs its token
   enforcer, the active root parser is explicitly configured to force schema
   field order and allow zero consecutive structural whitespace characters
   outside JSON strings; and
3. exact decoded model text and token-level termination telemetry are retained
   for both passes.

These changes address the observed failure mode rather than a clinical field
error. No prompt, response schema, compiler rule, or metric threshold is
relaxed.

## V3 boundary and transparent corpus reuse

The frozen v3 run completed three of twenty rows and stopped on the fourth,
`MEL-104-B` under `ocr_degraded`, when its blind-audit generation reached the
768-token limit before completing valid JSON. All four `MEL-104` input cells
were therefore exposed to model generation in v3. V4 reassigns `MEL-104` to
development and does not treat it as held out.

V3 never made a model call for `MEL-105` through `MEL-108`. V4 reuses those
four semantic cases and their report artifacts unchanged in its formal set,
then adds the new `MEL-109` semantic case to restore a five-case, twenty-input
formal matrix.

| V4 case | V4 split | Origin | V3 model-call status |
|---|---|---|---|
| `MEL-104` | Development | Reused v3 case and report artifacts | Exposed; all four cells entered model generation |
| `MEL-105` to `MEL-108` | Formal | Reused v3 cases and report artifacts | Never called |
| `MEL-109` | Formal | New v4 case and report artifacts | Not applicable |

For `MEL-104` through `MEL-108`, the v4 generator uses the same report
construction code and degradation parameters as v3. Their source text,
ground-truth JSON, clean PDFs, degraded PDFs, clean PNGs, and degraded PNGs
must be byte-identical to the corresponding committed v3 artifacts. The v4
case table and manifest explicitly record each case and artifact origin.

No v3 OCR, prompt, candidate, audit, compiler, normalized, or run-record
artifact is copied into v4 results. Every v4 model response must come from a
new call under the v4 lock. V3 results and any incomplete-v3 diagnostics are
excluded from every v4 development and formal metric.

## V4 corpus and split

| Split | Semantic cases | Report layouts | Rendered inputs |
|---|---:|---:|---:|
| Development | `MEL-104` | 2 | 4 |
| Formal execution-held-out | `MEL-105` through `MEL-109` | 10 | 20 |
| Total | 6 | 12 | 24 |

Each semantic case has two deterministic layouts:

- Template A: compact synoptic report
- Template B: narrative report

Each layout has two paired conditions:

- `clean`
- `ocr_degraded`

The formal execution-held-out set therefore contains five semantic cases, ten
report-layout documents, and twenty paired rendered inputs: ten clean and ten
degraded. The paired conditions are not twenty independent clinical cases.

The formal set includes explicit missingness, exact, minimum, and approximate
numeric qualifiers, mixed margin states, unstated laterality and staging, and
a prior-report history distractor. It contains 124 ground-truth-null
opportunities across the 20 rendered inputs and 16 scored clinical fields.

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
part of the 16-field pooled clinical exact-match denominator.

Only values explicitly supported for the current specimen may be non-null.
The pipeline must not calculate staging, convert a missing mitotic rate to
zero, or copy a measurement or stage from clinical history or a prior
specimen.

## Pipeline under evaluation

The only model is `google/medgemma-1.5-4b-it`, requested at commit
`91850547d9f0b2fdd21aa7c5f4f3d1a8a52c243b`. The resolved model revision must
exactly match the pin.

### OCR-assisted input

Each committed image is processed by Tesseract through `pytesseract`, with
language `eng`, configuration `--oem 1 --psm 6`, a 120-second timeout, and no
image preprocessing. Word-level `pytesseract.Output.DICT` data are retained.
Words are grouped in first-seen page, block, paragraph, and line order,
assigned stable `L001` through `Lnnn` IDs, and deterministically tagged as
header, current specimen, history, or footer/comment.

The report image and numbered OCR transcript are supplied together to both
model passes. OCR is an assistive observation, not ground truth. The OCR
implementation, settings, and line-sectioning code are unchanged from v3.

### Pass 1: extraction candidate

MedGemma receives the report image, numbered OCR transcript, unchanged
candidate prompt, and unchanged candidate response schema. It returns one
complete extraction candidate.

Generation uses direct `model.generate`, greedy decoding, one beam, random
seed 0, BF16, no assistant prefill, and a 512-token maximum. A fresh
schema-constrained parser is used. Candidate serialization remains exactly as
it operated in v3: after the LM Format Enforcer builder replaces the requested
configuration, the effective parser uses `force_json_field_order = false` and
`max_consecutive_whitespaces = 12`. V4 does not repair or otherwise alter the
candidate serializer.

### Pass 2: blind presence audit

A separate MedGemma call receives the same image, numbered OCR transcript,
unchanged audit prompt, and unchanged audit response schema. The audit prompt
is finalized before candidate generation and receives no candidate content.

For every field it reports:

- `present`, with the smallest one or two supporting OCR line IDs;
- `absent`, with no evidence line IDs; or
- `unclear`, with relevant line IDs when available.

The audit uses greedy decoding, one beam, random seed 0, BF16, no assistant
prefill, and one fresh schema-constrained parser. Its maximum is 1280 generated
tokens.

The larger cap was selected solely from retained v3 serialization diagnostics:
successful audits used 656 to 766 of the 768 allowed tokens, while the failed
audit reached the 768-token boundary with incomplete JSON. Candidate responses
used 225 to 275 of 512 tokens. No formal ground truth or accuracy feedback was
used to choose 1280.

### Effective JSON serialization enforcement

LM Format Enforcer 0.11.3 constructs a `TokenEnforcer` and replaces the
supplied parser configuration with a tokenizer-alphabet configuration. V3
configured field order before that builder call, so its requested field order
was not effective and the library default still allowed up to twelve
consecutive whitespace characters.

For the audit pass only, v4 constructs the Transformers prefix function first,
then explicitly applies the frozen configuration to the active post-builder
root parser:

- tokenizer alphabet from the frozen tokenizer data;
- `force_json_field_order = true`;
- `max_consecutive_whitespaces = 0`; and
- the unchanged schema-defined array bounds.

The runner must assert the pass-specific effective post-builder values before
every generation. The candidate assertion requires the v3-effective
`force_json_field_order = false` and
`max_consecutive_whitespaces = 12`; the audit assertion requires
`force_json_field_order = true` and `max_consecutive_whitespaces = 0`.
Decoded output from either pass must be one strict JSON object with no
duplicate keys, non-finite numbers, leading prose, trailing prose, or schema
violation. Whitespace inside JSON string values is unaffected.

### Raw response and token telemetry

For each candidate and audit call, v4 retains:

- exact decoded generated text and its hash;
- generated token IDs and their hash;
- prompt token count;
- generated token count;
- configured maximum generated tokens;
- whether the final generated token was an EOS token;
- whether generation consumed the configured maximum;
- stopping and decoding settings; and
- the asserted effective post-builder parser configuration.

The raw response is never reconstructed from parsed JSON. Parsed JSON is a
separate artifact. This telemetry distinguishes a complete constrained
response from a token-cap truncation without consulting ground truth.

### Deterministic evidence gate and compiler

The compiler is byte-identical to v3. It receives only the candidate response,
blind audit response, and numbered OCR lines. It never receives ground truth.

Except for the unchanged linked binding and normalization rules, a non-null
clinical candidate is accepted only when all of the following hold:

1. the candidate can be deterministically canonicalized;
2. the blind audit marks the field `present`;
3. the audit cites one or two known, non-duplicated OCR line IDs;
4. cited evidence is in an allowed non-history section;
5. deterministic parsing of the cited text produces one unambiguous canonical
   value; and
6. that value exactly agrees with the canonicalized candidate.

The unchanged linked rules are:

1. Document ID may bind from exactly one matching labeled header identifier
   when the non-null candidate agrees and its audit entry is absent or unclear.
2. Laterality may reuse accepted specimen-site evidence only when that evidence
   contains an explicit matching laterality token.
3. A null or omitted numeric qualifier may be derived only from the same
   accepted non-history evidence as its already accepted numeric value, using
   the fixed exact, lower-bound, and approximate wording rules.

The unchanged specimen-site allowance permits at most one aligned substitution
from the fixed OCR-glyph whitelist. Insertions, deletions, reordering, multiple
substitutions, and non-whitelisted substitutions fail closed.

Every other failed non-null candidate becomes null. History evidence is always
rejected. Numeric values and qualifiers abstain jointly when incoherent. The
compiler validates the final object against the unchanged canonical schema and
retains its field-level decision log.

## Development gate and revision budget

Only the four `MEL-104` v4 inputs may be used for v4 development. They are
explicitly not held out because their v3 model generations were exposed.

V4 permits one documented development iteration. It may test only the frozen
serialization safeguards and the unchanged v3 clinical pipeline. No prompt,
schema, OCR rule, compiler rule, scored field, corpus fact, or threshold may
be revised based on this run. If the complete four-input development matrix
does not serialize and compile successfully under the frozen configuration,
v4 is abandoned.

Before formal execution, the four-input development serialization smoke must
satisfy every conjunctive requirement:

1. 4/4 inputs complete with valid provenance;
2. 4/4 raw candidate responses parse as strict candidate-schema JSON;
3. 4/4 raw blind-audit responses parse as strict audit-schema JSON;
4. 4/4 final outputs validate against the canonical schema;
5. every accepted non-null value has valid non-history evidence;
6. neither pass reaches its configured generated-token cap on any input;
7. zero unsupported non-null final values; and
8. zero history carryover.

Development 16-field exact match, precision, recall, and F1 are calculated and
reported descriptively, but no development accuracy threshold is a
configuration-selection estimate. No v4 development output is included in a
formal metric.

## Ground-truth firewall

Formal ground truth may be generated, hash-bound, and structurally validated
as a corpus artifact, but it is unavailable to operational inference.

During OCR, prompt assembly, both model passes, compilation, and final-output
sealing:

- runtime code must not read `data/v4/ground_truth/`;
- runtime code must not derive target facts from `data/v4/cases.csv`;
- ground-truth values must not appear in prompts, OCR corrections, caches, or
  fallback logic; and
- the compiler may use only candidate, audit, OCR, schema, and frozen
  configuration artifacts.

Formal evaluation may deserialize ground truth only after all 20 final outputs
and provenance records have been produced and hash-sealed. Corpus-construction
tests remain separate from the inference entry point.

## Freeze requirements

Before the first formal v4 model call, an immutable committed lock must bind:

- repository source commit and protocol identifier;
- v4 cases, manifest, image, PDF, source-text, and ground-truth hashes;
- explicit case and artifact reuse provenance;
- exact requested and resolved MedGemma revision;
- OCR engine, version, and all OCR settings;
- unchanged prompt, schema, and compiler hashes;
- candidate and audit token caps;
- effective post-builder parser field-order and whitespace settings;
- raw-response and token-telemetry requirements;
- inference backend, runtime versions, device, and dtype;
- decoding, stopping, and seed settings;
- formal execution order and absolute no-resume rule;
- the sole v4 development-iteration record and its artifact hashes; and
- every pass criterion in this document.

The formal entry point must fail closed when the lock is absent or when any
locked source, corpus artifact, configuration, runtime value, development
record, or hash differs.

Changing a clinical prompt, schema, OCR configuration, model revision,
compiler rule, pass threshold, or scored corpus after a formal call begins
invalidates v4. This project authorizes no further recovery version.

## Formal one-shot execution

A complete formal attempt gives each of the 20 inputs exactly one candidate
call and one blind-audit call in committed manifest order. There is no
semantic retry, infrastructure retry, resume, best-of-N selection, majority
vote, output repair, or accuracy-guided continuation.

The first committed formal-attempt start event consumes the v4 one-shot
budget. Any exception, process loss, infrastructure failure, semantic failure,
serialization failure, schema failure, evidence failure, or compilation
failure ends v4 as incomplete and prohibits another formal invocation,
including for an unchanged missing row that received no valid response. The
runner exits on the first failure. Started, failed, and completed formal
attempt state is preserved in the append-only ledger; an unterminated started
attempt is also a terminal incomplete v4 outcome.

No formal accuracy result may be used to change the pipeline. Aggregate and
field-level evaluation begins only after all 20 final outputs and their
provenance records are hash-sealed.

## Formal metrics and unchanged pass criteria

The formal v4 gate passes only if all criteria below are satisfied:

1. **Completion and provenance:** 20/20 formal rows are complete and
   provenance-valid.
2. **Canonical schema:** 20/20 final outputs validate against the canonical
   extraction schema.
3. **Evidence acceptance:** 100% of non-null final clinical values have a
   successful evidence decision supported by valid non-history evidence.
4. **Pooled field exactness:** at least 90% over 320 field instances, requiring
   at least 288 correct.
5. **Condition field exactness:** at least 85% over each condition's 160 field
   instances, requiring at least 136 correct for clean and at least 136 correct
   for `ocr_degraded`.
6. **Unsupported-field rate:** at most 2% over 124 ground-truth-null
   opportunities, permitting no more than two unsupported non-null values.
7. **History carryover:** zero final values copied from prior-report or history
   text when absent from or contradicted by the current specimen.

All criteria are conjunctive. Strong performance on one cannot compensate for
failure on another. Unsupported-field rate is reported beside non-null recall
or micro-F1 so an all-null output cannot appear safe.

Also report document-ID accuracy, complete-report accuracy, precision, recall,
F1, per-field results, results by template and condition, paired
degraded-minus-clean change, candidate and audit schema diagnostics, token
telemetry, and evidence-rejection reasons.

## Review status and interpretation

The blind audit is a MedGemma model pass, not a human review. AI-assisted
artifact checks may be labeled `ai_audited` but never `human_verified`.

Before outreach describes the v4 outputs as manually verified, a human must
inspect the source image, current-versus-history distinction, final JSON, and
cited evidence for all 20 formal rows.

A v4 pass would support only this bounded statement: the frozen hybrid
pipeline met its prespecified thresholds on five synthetic semantic cases,
ten paired report layouts, and twenty rendered inputs under this protocol. It
would not establish raw MedGemma accuracy, clinical validity, institutional
generalization, or readiness for patient care.
