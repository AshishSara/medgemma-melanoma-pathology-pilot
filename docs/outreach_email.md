# Outreach decision and email draft

## Recommendation

Do **not** send a success-oriented pilot email yet. V4 passed its exact model,
OCR, and small non-corpus serializer-configuration preflights, but its sole
development execution ended in a hosted-session failure before any
report-level result was observed or recovered. The automated technical gate
therefore remains incomplete, formal inference was not run, and there is no
measured v4 result to present.

The preferred next external step is the HAI-DEF engagement pathway or a
developer forum/GitHub technical question. If an individual email is sent
despite that ordering, it should be the transparent note below and should go
to Daniel Golden alone.

## Transparent fallback draft

Send only after the public commit and repository link below have been
independently verified.

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
ended incomplete on a serialization cap, and a prespecified v4 recovery
attempt ended when the hosted GPU session entered `Error` after runner launch;
no report-level v4 output was recovered. The 20-input v4 formal evaluation was
not run. I have published the code, synthetic corpus, protocol, and incident
record here:

[GitHub repository and v4 incident record](https://github.com/AshishSara/medgemma-melanoma-pathology-pilot/tree/agent/v3-constrained-pipeline)

Before I spend more time or compute on a larger 75–100-report study, would
this narrow use case be useful or meaningfully distinct to the MedGemma team,
or is there related work I should review first? I understand formal use-case
engagement should go through HAI-DEF.

Best,

Ashish Saragadam

## Claims deliberately excluded

This draft does not claim that:

- the v4 development or formal gate passed;
- MedGemma processed the four development reports successfully;
- any v4 accuracy, unsupported-field, or robustness metric exists;
- outputs were manually reviewed by a human; or
- the use case was submitted through HAI-DEF.
