# Extraction schema

The canonical machine-readable schema is
[`schema/extraction.schema.json`](../schema/extraction.schema.json).

## Null policy

`null` means the field is not stated for the current specimen. The extractor
must not:

- derive pT from Breslow thickness and ulceration;
- copy a prior-case measurement from clinical history;
- convert a missing mitotic rate into zero;
- treat "cannot be assessed" as "not identified";
- invent laterality from an anatomic site;
- infer negative nodes or distant disease when none are reported.

Explicit source phrases such as "cannot be assessed" are retained as controlled
categorical values. Only genuinely unstated fields are `null`.

## Numeric policy

Breslow thickness and mitotic rate are numeric values plus a separate qualifier.
For example, "at least 1.0 mm" becomes:

```json
{
  "breslow_thickness_mm": 1.0,
  "breslow_qualifier": "at_least"
}
```

Units are fixed by schema. A model must not emit unit-bearing strings in numeric
fields. A numeric value and its qualifier must either both be present or both be
`null`; incoherent value/qualifier pairs fail schema validation.

## Margin policy

Invasive melanoma and melanoma in situ are scored separately at peripheral and
deep margins. A missing component remains `null`; it is not automatically
"not involved."

## Staging policy

Only staging explicitly reported in the document is extracted. This is a
document-understanding benchmark, not a staging calculator.
