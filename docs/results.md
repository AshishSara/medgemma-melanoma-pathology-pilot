# Formal held-out results

## Outcome

The `heldout-pilot-v2` run is complete and reproducible. Codex reviewer agents
completed an AI-assisted visual and provenance audit of every output; human
manual verification remains pending. The measured extraction result is
negative: both models generated syntactically valid JSON for every held-out
input, but neither produced a single response that conformed to the frozen
extraction schema.

| Metric | MedGemma 1.5 4B IT | MedGemma 1 4B IT |
|---|---:|---:|
| Completed outputs | 28/28 | 28/28 |
| Parse-valid JSON | 28/28 | 28/28 |
| Schema-valid JSON | 0/28 | 0/28 |
| Correct document ID | 28/28 | 28/28 |
| Field exact match | 178/448 (39.7%) | 230/448 (51.3%) |
| Non-null micro-F1 | 40.0% | 52.0% |
| Unsupported-field rate | 14/172 (8.1%) | 47/172 (27.3%) |
| Complete-report accuracy | 0/28 | 0/28 |

The target model produced fewer unsupported values than the comparator, but
also lower exact match and micro-F1. The comparator's higher recall came with a
substantially higher unsupported-field rate.

## Clean versus degraded

MedGemma 1.5 field exact match was 40.6% on clean inputs and 38.8% on degraded
inputs, a mean paired change of -1.8 percentage points. Across its 14 matched
pairs, the degraded version was worse in 4, tied in 8, and better in 2.

The comparator field exact match was 51.3% in both conditions, a mean paired
change of 0.0 points. Its degraded version was worse in 3 pairs, tied in 6,
and better in 5. These small paired aggregates should not be interpreted as a
general OCR-robustness estimate.

## Error interpretation

Frequent schema failures included:

- free-text or differently formatted values instead of frozen enum strings;
- `null` instead of the required structured `staging` object;
- string-valued numerics and quoted `null` values;
- omitted qualifiers, margins, and explicitly reported staging elements;
- unsupported margin, laterality, or staging values.

The deliberately separated prior-report case (`MEL-008`) was unambiguous in
both layouts and conditions. MedGemma 1 often copied the historical 0.9 mm
Breslow value into the current-specimen extraction. MedGemma 1.5 generally
avoided that historical carryover but failed exact match on the explicit
current-specimen in-situ margins in all four outputs: three omitted the values,
and one used noncanonical formatting.

No semantic repair, type coercion, synonym mapping, staging inference, or
post-hoc key insertion was applied. A more permissive normalization layer
would answer a different research question and is not reported as model
performance here.

## Provenance and review

- Protocol commit:
  `32f1453e2acbc79dbe50c7f80d9f42a519fdba8f`
- Prompt SHA-256:
  `139bc7f1dc858af0a5a2643f38b396f71c40e39e1419c14a281dd78f03ee801b`
- Schema SHA-256:
  `c06982e97db88758cb8cbf10c7006d3bc76d0599cd4ffb6334cec0b76c180d72`
- Runtime: Tesla T4, BF16, greedy decoding, 1,200-token ceiling
- Completed outputs: 56/56
- Invalid provenance: 0
- Hash-bound Codex AI audits: 56/56
- Unique source images visually audited by Codex: 28/28
- AI-audited output rows with source ambiguity: 0/56
- Human manual reviews: 0/56

Raw responses, model-only continuations, parsed objects, run records,
aggregate metrics, and the review ledger are under `results/`. The public
compact evidence package is
[`results/bundles/formal-results-32f1453e-reviewed.tgz`](../results/bundles/formal-results-32f1453e-reviewed.tgz);
verify it before unpacking:

```bash
(cd results/bundles && \
  shasum -a 256 -c formal-results-32f1453e-reviewed.sha256)
```

## Limits and next decision

This is a seven-semantic-case synthetic feasibility pilot with manually
selected prompting and a one-character JSON assistant prefix. It is not an
estimate of clinical performance, a diagnostic system, or evidence that the
same behavior will hold on real pathology reports.

The next appropriate step is not individual research outreach. First complete
human verification of every output, submit the measured use case through
HAI-DEF, and ask a reproducible technical question through the developer forum
or GitHub. A qualified dermatopathologist should approve the schema before any
expanded study.
