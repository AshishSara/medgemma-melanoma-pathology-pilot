# Outreach decision and email draft

## Recommendation

**Do not send an email. The project is paused.**

V5 was prospectively frozen and publicly locked, but its four-input
qualification stopped during gated Hugging Face access with HTTP 401 before
the model loaded. No model call, report-level output, or extraction metric
exists. Formal v5 inference and human review were not run.

The project owner elected to stop rather than repair authentication and
resume. No email has been sent to Daniel Golden and no HAI-DEF submission has
been made. Reconsider outreach only after a future prospectively reviewed
protocol passes both its automated and human-review gates.

## Archived transparent fallback draft — not approved for sending

This draft is retained only so the original outreach context is not lost. It
must be rewritten if the project restarts.

**To:** Daniel Golden `<dangolden@google.com>`

**Subject:** MedGemma 1.5 – melanoma pathology extraction feasibility question

Hi Daniel,

I am exploring a wholly synthetic evaluation of MedGemma 1.5 for extracting
management-critical melanoma pathology fields—site and laterality, Breslow
thickness, ulceration, mitotic rate, margins, and explicitly reported
staging—from clean and OCR-degraded reports, with unsupported values measured
explicitly.

I want to be clear that I do not have a positive pilot result. A small initial
v2 baseline did not meet its extraction-quality gate. A constrained v3 attempt
ended incomplete on a serialization cap, a prespecified v4 recovery ended in
a hosted GPU incident, and a fresh v5 qualification stopped on gated-model
authentication before any model call. No v4 or v5 report-level result exists,
and neither formal evaluation was run. I have published the code, synthetic
corpora, protocols, and incident records here:

[GitHub repository and incident history](https://github.com/AshishSara/medgemma-melanoma-pathology-pilot/tree/agent/v3-constrained-pipeline)

Before I spend more time or compute on a larger 75–100-report study, would
this narrow use case be useful or meaningfully distinct to the MedGemma team,
or is there related work I should review first? I understand formal use-case
engagement should go through HAI-DEF.

Best,

Ashish Saragadam

## Claims deliberately excluded

This draft does not claim that:

- any confirmatory development or formal gate passed;
- MedGemma processed the v5 qualification reports;
- any v4 or v5 accuracy, unsupported-field, or robustness metric exists;
- outputs were manually reviewed by a human; or
- the use case was submitted through HAI-DEF.
