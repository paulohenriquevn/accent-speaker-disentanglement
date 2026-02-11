from __future__ import annotations

import pandas as pd

from stage1_5.pipelines.run_all import _decide


def _base_cfg() -> dict:
    return {
        "experiment": {
            "leakage_margin_pp": 7,
            "leakage_conditional_margin_pp": 12,
            "min_f1_go": 0.55,
            "min_f1_conditional": 0.45,
            "text_drop_tolerance_pp": 10,
        }
    }


def test_decide_filters_non_accent_targets() -> None:
    cfg = _base_cfg()
    df = pd.DataFrame([
        {
            "label": "backbone:rep_speaker",
            "target": "speaker",
            "accent_f1": 0.95,
            "accent_text_drop": 0.0,
            "leakage_a2s": 0.10,
            "chance_speaker": 0.25,
        },
        {
            "label": "backbone:rep_accent",
            "target": "accent",
            "accent_f1": 0.60,
            "accent_text_drop": 0.02,
            "leakage_a2s": 0.30,
            "chance_speaker": 0.25,
        },
    ])

    decision = _decide(df, cfg)
    assert decision.status == "GO"
    assert decision.label == "backbone:rep_accent"


def test_decide_go_conditional() -> None:
    cfg = _base_cfg()
    df = pd.DataFrame([
        {
            "label": "backbone:rep",
            "target": "accent",
            "accent_f1": 0.46,
            "accent_text_drop": 0.05,
            "leakage_a2s": 0.30,
            "chance_speaker": 0.25,
        }
    ])

    decision = _decide(df, cfg)
    assert decision.status == "GO_CONDITIONAL"


def test_decide_nogo_when_backbone_and_ssl_weak() -> None:
    cfg = _base_cfg()
    df = pd.DataFrame([
        {
            "label": "backbone:rep",
            "target": "accent",
            "accent_f1": 0.30,
            "accent_text_drop": 0.02,
            "leakage_a2s": 0.20,
            "chance_speaker": 0.25,
        },
        {
            "label": "ssl:layer_0",
            "target": "accent",
            "accent_f1": 0.35,
            "accent_text_drop": 0.02,
            "leakage_a2s": 0.20,
            "chance_speaker": 0.25,
        },
    ])

    decision = _decide(df, cfg)
    assert decision.status == "NOGO"
    assert "weak" in decision.rationale.lower()
