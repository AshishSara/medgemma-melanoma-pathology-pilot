# Fresh held-out confirmatory pilot (v5)

Status before lock: prospective draft. This document becomes immutable when
`data/v5/protocol_lock.json` is committed and published.

The source commit must not contain the lock. At execution, every hash in all four locked artifact
maps must match the corresponding blob in both that prospective source commit and the execution
HEAD, as well as the working-tree file. This prevents a descendant commit from changing an
artifact and coordinating a replacement lock while continuing to claim the earlier source
commit.

## Scientific identity

This is a new confirmatory feasibility pilot. It is not a retry, continuation, recovery, or
reinterpretation of v4. Earlier outcomes remain part of the public record:

- v2 completed with a negative result;
- v3 produced an encouraging tuned development result, but formal inference was incomplete;
- v4 ended terminally incomplete after a hosted-session incident.

The new question is deliberately narrower than real-world validation:

> Can the already-developed MedGemma 1.5 + OCR + deterministic evidence-compiler pipeline
> reproduce prespecified feasibility performance on five fresh synthetic clinical fact vectors
> across two fixed, previously developed layout families and paired clean/OCR-degraded inputs?

A pass supports only descriptive, in-distribution synthetic-pilot feasibility. It does not
establish raw MedGemma accuracy, clinical validity, patient safety, unseen-layout robustness, or
real-world generalization.

## Fixed pipeline

The clinical logic is carried forward byte-for-byte from the v4 implementation:

- model: `google/medgemma-1.5-4b-it`;
- revision: `91850547d9f0b2fdd21aa7c5f4f3d1a8a52c243b`;
- dtype: BF16;
- decoding: greedy, one beam, no assistant prefill, one candidate call and one blind-audit call
  per input;
- candidate maximum: 512 new tokens;
- blind-audit maximum: 1,280 new tokens;
- candidate serializer: field order not forced, at most 12 consecutive whitespace characters;
- audit serializer: field order forced, zero consecutive whitespace characters;
- OCR: Tesseract English, `--oem 1 --psm 6`, no image preprocessing;
- prompts: `prompts/v3/candidate_prompt.txt` and `prompts/v3/audit_prompt.txt`;
- response schemas: `schema/v3/candidate.schema.json` and `schema/v3/audit.schema.json`;
- final schema: `schema/extraction.schema.json`;
- deterministic compiler: `scripts/v3_pipeline.py`.

The blind-audit prompt is rendered before candidate generation and receives no candidate data.
No fine-tuning, best-of-N, voting, post-hoc repair, manual correction, or alternative extractor is
allowed.

## Corpus and fixed matrix

V5 contains exactly six new clinical fact vectors and uses IDs not present in v2-v4. One vector
is assigned only to operational qualification and five are assigned only to the formal set:

- Operational qualification: `MEL-190`, one fact vector rendered in layouts A and B and in clean
  and OCR-degraded form (4 inputs).
- Formal held-out set: `MEL-201` through `MEL-205`, five fact vectors rendered in layouts A and B
  and in clean and OCR-degraded form (20 inputs).

The complete v5 corpus therefore contains six independent clinical fact vectors and 24 rendered
inputs. The formal denominator contains only the five formal vectors: ten report-layout documents
and twenty scored rendered inputs. `MEL-190` and its four qualification inputs are excluded from
every formal denominator. Twenty formal inputs must never be described as twenty independent
cases.

Layouts A and B are intentionally known fixed layout families. Freshness applies to semantic
facts, source text, report IDs, dates, degradation salt, PDFs, and rendered images—not to layout
family.

The mechanical semantic-fingerprint rule is fixed before inference. A fingerprint is the exact
ordered 16-string tuple of `specimen_site`, `laterality`, `diagnosis`,
`breslow_thickness_mm`, `breslow_qualifier`, `ulceration`, `mitotic_rate_per_mm2`,
`mitotic_qualifier`, `invasive_peripheral`, `invasive_deep`, `in_situ_peripheral`,
`in_situ_deep`, `pT`, `pN`, `pM`, and `stage_group`, as decoded from the source CSV. Empty strings
are retained and no fuzzy, similarity, or post-hoc judgment rule is used. The generator and
automated corpus test load `data/cases.csv` (the complete v2 corpus), `data/v3/cases.csv`
(`MEL-101` through `MEL-103`, used in both v3 development iterations including tuned iteration 2,
plus the full v3 formal allocation), and `data/v4/cases.csv` (the complete v4 allocation). They
form the union of every row in those three files and require every v5 fingerprint to be exactly
absent. They also require every v5 case ID and report date to be absent from the prior files. This
exact-match control rules out reuse of a complete prior management-field vector; it does not claim
that individual attributes or layout families are novel.

### Pre-lock freshness audit

The first pre-lock exact-match check found that the draft `MEL-202` vector was identical to v2
`MEL-004` across all 16 fingerprint fields. No model inference had occurred and no lock existed.
Under the already declared mechanical rule, `MEL-202`'s specimen site was changed from `shoulder`
to `abdomen`, with the other 15 fingerprint fields unchanged, and every v5 corpus artifact was
regenerated. The repeated check found zero prior-vector collisions, six unique v5 fingerprints,
zero prior ID overlaps, and zero prior report-date overlaps.

This correction is legitimate pre-lock corpus construction, but it makes `MEL-202` a
one-attribute near-neighbour of the earlier v2 case. That derivation and the fact that the overlap
test initially failed must be disclosed in the final report. It must not be represented as proof
that every attribute combination is novel.

The formal denominator is frozen as:

- 320 management-critical field opportunities (20 inputs x 16 fields);
- 196 ground-truth non-null opportunities;
- 124 ground-truth null opportunities;
- 160 field opportunities in the clean condition;
- 160 field opportunities in the OCR-degraded condition.

Ground truth, source text, PDFs, images, generation source rows, and the manifest are hash-locked
before report-level model inference.

## Runtime qualification

The execution environment is fixed in `data/v5/runtime_profile.json` and
`requirements/colab-v5.txt`: one Tesla T4, Python 3.12.13, PyTorch 2.11.0+cu128, CUDA 12.8,
Transformers 4.57.6, Accelerate 1.14.0, LM Format Enforcer 0.11.3, pytesseract 0.3.13, and
Tesseract 4.1.1.

Environment capture and package installation are not model inference. The backend must load and
match the fixed profile before an execution attempt is claimed. Any serializer smoke uses only a
non-clinical dummy image and tiny non-clinical schemas; it may not access `MEL-190` or any formal
case.

## Operational qualification gate

The four `MEL-190` inputs run only after the protocol is locked and always before any formal
model call. Qualification runs and is scored to completion unconditionally; its response seal,
metrics, and pass/fail result are committed and publicly published before formal inference can
begin. Formal inference is allowed only if all of the following pass without changing any locked
byte:

- 4/4 complete and provenance-valid;
- 4/4 candidate and audit responses are strict schema-valid JSON;
- 4/4 final outputs are canonical-schema-valid;
- no candidate or audit generation reaches its token cap;
- at least 58/64 fields are exact overall;
- at least 28/32 fields are exact in each of clean and OCR-degraded conditions;
- at least 45/52 ground-truth non-null fields are recalled;
- 0/12 ground-truth null opportunities contain an unsupported value;
- every accepted non-null value has valid current-report evidence;
- zero history carryover.

These qualification criteria are part of the same protocol source inventory and hash lock as the
corpus, formal gate, prompts, schemas, compiler, and execution rules; no separate or later gate
specification is permitted.

Qualification is a stop/go test, not a tuning phase. No source, prompt, schema, compiler,
threshold, token cap, corpus row, or execution rule may change after observing qualification
outputs. A failed qualification makes v5 terminally unsuccessful and formal inference is not run.
The qualification result is disclosed in the final report and in any v5 outreach regardless of
direction; it may not be skipped, delayed until after formal inference, or selectively omitted.

## Formal success gate

Success is conjunctive. Every criterion must pass:

1. 20/20 assigned inputs complete and provenance-valid.
2. 20/20 candidate and blind-audit responses are strict schema-valid JSON.
3. 20/20 compiled outputs are canonical-schema-valid.
4. No candidate or audit generation reaches its token cap.
5. At least 288/320 management-critical fields are exactly correct (90%).
6. At least 136/160 clean-condition fields are exactly correct (85%).
7. At least 136/160 OCR-degraded-condition fields are exactly correct (85%).
8. At least 167/196 ground-truth non-null fields are recalled (85%).
9. No more than `floor(0.02 * 124) = 2` unsupported values occur across ground-truth null
   opportunities.
10. Every accepted non-null value is backed by valid, non-history evidence under the fixed
    compiler contract.
11. History carryover count is zero.

Thresholds are operational feasibility thresholds, not clinically validated cutoffs. The 90%
pooled requirement limits the pilot to at most 32 field errors; the 85% per-condition floors
prevent pooled performance from hiding a condition-specific collapse; the 85% non-null recall
floor prevents blanket abstention from appearing safe; and the 2% unsupported ceiling represents
a safety-oriented maximum of one invented value per fifty null opportunities. These rationales
do not depend on the observed v3 development percentage.

Complete-report accuracy, per-field results, per-template results, paired clean-to-degraded
changes, raw-pass accuracy, latency, token counts, and compiler rejection reasons are secondary
descriptive endpoints and cannot override the conjunctive gate.

## Attempt, retry, and resume policy

The fixed manifest order is used. Formal progress output may show only row identity, stage,
timing, and mechanical completion; semantic values and interim metrics are not inspected until
all 20 rows are sealed.

Each candidate and audit pass has an append-only call ledger and an atomic response bundle.

- A call-start event is written before every model call.
- A returned response bundle is atomically persisted before parsing or scoring.
- A valid persisted response is never regenerated.
- A cap hit, malformed JSON, schema-invalid JSON, deterministic-compilation failure, or
  provenance mismatch is a terminal content failure and is never retried.
- If a call-start event exists but no response bundle was durably persisted because the runtime
  terminated or raised before returning, exactly one unchanged infrastructure retry of that pass
  is permitted. A second no-response failure is terminal.
- A completed row is hash-verified and skipped on resume.
- A valid persisted candidate is reused directly when only the audit pass is missing.
- A valid persisted audit is reused directly for deterministic compilation.
- Corrupt, partial, inconsistent, or wrong-lock checkpoints fail closed.
- Model revision, repository commit, lock hash, runtime fingerprint, row order, prompts, schemas,
  compiler, decoding settings, and image hash must remain unchanged across a resume.

Every execution attempt and call, including failed attempts, is retained in the public result
ledger.

## No-peek and no-exclusion rules

Once formal inference begins:

- it runs to completion unless interrupted by a mechanically classified terminal or
  infrastructure failure;
- no semantic output or interim metric is inspected;
- no row is excluded, reclassified, reweighted, substituted, regenerated, or reordered;
- no early stopping or case rerandomization is allowed;
- missing or failed assigned rows remain in the denominator and make the gate fail;
- automated scoring is performed against the hash-locked ground truth only after a 20-row output
  seal is written.

## Human review and outreach

After automated scoring, the project owner—not an AI system—must visually review all 20 image,
OCR, candidate, audit, final-JSON, and evidence bundles without correcting predictions. The
reviewer should not see the automated pass/fail result until the checklist is complete. Claude,
Qwen, or another model may review source code before lock but cannot tune from held-out outputs,
repair predictions, vote as an extractor, or substitute for this human check.

The pilot may be described as positive, and a success-oriented email may be drafted, only if the
automated gate passes and the blinded human checklist is complete. Any email must give exact
fractions, disclose that there are five independent synthetic fact vectors, link the complete
v2-v5 history, and use the limited claim stated in this protocol.
