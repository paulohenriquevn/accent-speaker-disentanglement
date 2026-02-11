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

1. `layer × F1_accent`
2. `layer × leakage`
3. `layer × text robustness`

## Decision

- **Best representation:** {{ best_layer }}
- **Decision:** {{ decision }}
- **Rationale:** {{ rationale }}

## Notes

{{ notes }}
