---
title: MedGemma Melanoma Pathology Extraction Pilot
status: active
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
`MEL-008`. Outreach remains on hold until both pinned-model runs and all manual
reviews are complete. A published harness is not a completed technical gate.

## Working links

- [[docs/protocol|Protocol]]
- [[docs/gate_status|Technical gate]]
- [[docs/schema|Extraction schema]]
- [[docs/manual_review|Manual review]]
- [[docs/sources|Evidence and overlap check]]
- [[local-notes/Outreach Draft|Private outreach draft]]

## Next concrete action

Publish the held-out protocol commit, pin Colab to that immutable commit, and
run the 56-output BF16 matrix on the confirmed Tesla T4. Then evaluate and
visually review every hash-bound output, update the pilot sentence, and submit
the HAI-DEF feedback form before emailing an individual researcher.
