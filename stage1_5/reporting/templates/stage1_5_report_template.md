# Stage 1.5 — Latent Separability Report

## Overview

- **Dataset:** {{ dataset_name }}
- **Date:** {{ date }}
- **Config:** {{ config_path }}

## Metrics Summary

| Representation | Accent F1 | Accent F1 (Text Split) | Accent Text Drop | Speaker Acc | Leak A→S | Leak S→A |
| --- | --- | --- | --- | --- | --- | --- |
{% for row in metrics %}
| {{ row.label }} | {{ row.accent_f1 | round(3) }} | {{ row.accent_text_f1 | round(3) }} | {{ row.accent_text_drop | round(3) }} | {{ row.speaker_acc | round(3) }} | {{ row.leakage_a2s | round(3) }} | {{ row.leakage_s2a | round(3) }} |
{% endfor %}

## Heatmaps

1. `layer × F1_accent` — see `figures/accent_f1.png`
2. `layer × leakage` — see `figures/leakage.png`
3. `layer × text robustness` — see `figures/accent_text_robustness.png`

## Domain Comparison

Comparison of Accent F1 across Real Audio, SSL upstream, and Backbone TTS features:

![Domain Comparison](figures/domain_comparison.png)

## Decision

- **Best representation:** {{ best_layer }}
- **Decision:** {{ decision }}
- **Rationale:** {{ rationale }}

## LoRA Recommendation

{{ lora_recommendation }}

## Risk Diagnostic

{{ risk_diagnostic }}

## Notes

{{ notes }}
