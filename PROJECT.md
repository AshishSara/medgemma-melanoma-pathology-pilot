---
title: MedGemma Melanoma Pathology Extraction Pilot
status: pilot-complete-human-review-pending-outreach-hold
created: 2026-08-04
tags:
  - medgemma
  - melanoma
  - pathology
  - document-extraction
  - synthetic-data
---

# MedGemma Melanoma Pathology Extraction Pilot

## Current decision

The public corpus is canonical and the held-out `heldout-pilot-v2` protocol is
frozen. `MEL-001` is development-only; formal scoring uses `MEL-002` through
`MEL-008`. The pinned 56-output run is complete. Codex reviewer agents
completed a hash-bound AI-assisted audit of every output, but human manual
verification remains pending. All outputs parse as JSON, but 0/56 conform to
the frozen schema, so the extraction-quality gate did not pass. Individual
outreach remains on hold.

## Working links

- [[docs/protocol|Protocol]]
- [[docs/gate_status|Technical gate]]
- [[docs/schema|Extraction schema]]
- [[docs/manual_review|Manual review]]
- [[docs/results|Formal results]]
- [[docs/sources|Evidence and overlap check]]
- [[local-notes/Outreach Draft|Private outreach draft]]

## Next concrete action

Publish the explicitly labeled AI-audited formal evidence in the draft GitHub
pull request, then have a human verify all 56 hash-bound rows. After that,
submit the measured negative result through the HAI-DEF feedback pathway and
route a reproducible schema-compliance question through the developer forum or
GitHub. Do not email an individual researcher until those steps are complete
and the outreach decision is reconsidered in light of the result.
