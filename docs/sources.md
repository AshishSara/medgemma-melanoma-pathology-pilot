# Evidence and overlap check

Checked 2026-08-04.

## Model and task

- [MedGemma 1.5 technical report](https://arxiv.org/abs/2604.05081)
  is valid (v2, 2026-05-01). It reports PDF/image-to-JSON laboratory-report
  extraction and identifies three corresponding authors.
- [MedGemma 1.5 model card](https://developers.google.com/health-ai-developer-foundations/medgemma/model-card)
  specifies text-plus-image input, text output, 896 x 896 image normalization,
  and the model ID `google/medgemma-1.5-4b-it`.
- [MedGemma get started](https://developers.google.com/health-ai-developer-foundations/medgemma/get-started)
  directs use-case feedback to the HAI-DEF feedback form, technical support to
  the developer forum, and technical issues to GitHub.
- [Google Research announcement](https://research.google/blog/next-generation-medical-image-interpretation-with-medgemma-15-and-medical-speech-to-text-with-medasr/)
  identifies Daniel Golden as Engineering Manager, Google Research.

## Novelty boundary

The technical report already describes a synthetic custom-PDF dataset and
layout-diverse lab-report extraction. The pilot therefore does not claim
novelty for synthetic PDFs or layout perturbation alone.

The official extraction targets reported in the paper are laboratory-result
fields, not melanoma specimen site/laterality, Breslow thickness, ulceration,
mitotic rate, margins, or reported staging. Internal training corpora are not
fully disclosed, so the repository makes no absolute claim that Google never
trained on similar narrative pathology content.

ISIC is explicitly listed as additional MedGemma 1.5 training data. The reviewed
official materials do not establish that SCIN was used to train MedGemma.
Accordingly, SCIN is described only as a Google-released dataset that may be a
less clean novelty benchmark, not as verified MedGemma training data.

## Reporting schema reference

- [CAP current cancer protocols](https://www.cap.org/protocols-and-guidelines/cancer-protocols/current-cancer-protocols/)
  lists the March 2025 invasive melanoma biopsy and excision protocols.

The synthetic templates are original and do not reproduce CAP protocol text.
CAP is used only to inform field selection. A dermatopathologist must review the
schema before any clinical interpretation.

## Engagement order

1. Complete and manually review the technical gate.
2. Submit the use case through the
   [HAI-DEF feedback form](https://services.google.com/fb/forms/hai-def-feedback/).
3. Use the [HAI-DEF developer forum](https://discuss.ai.google.dev/c/hai-def/62)
   or [MedGemma GitHub issues](https://github.com/Google-Health/medgemma/issues)
   for technical questions.
4. Only then consider a concise individual research email.

The HAI-DEF Showcase form currently targets active clinical studies or live
deployments and is not the right form for this early pilot.
