"""High-level orchestration for Stage 1.5 experiments."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import yaml

from ..data import Manifest
from ..probes.dataset import FeaturePlan
from ..probes.trainer import ProbeConfig, ProbeRunner
from ..reporting.writer import StageReportWriter
from ..utils.io import ensure_dir


@dataclass
class Decision:
    label: str | None
    status: str
    rationale: str


def run_pipeline(config_path: Path) -> Dict[str, Any]:
    cfg = yaml.safe_load(Path(config_path).read_text())
    cfg["config_path"] = str(config_path)
    manifest = Manifest.from_jsonl(cfg["paths"]["manifest"])

    plans = [_plan_from_dict(plan_cfg) for plan_cfg in cfg["probes"]["feature_spaces"]]
    probe_cfg = ProbeConfig(
        max_iter=cfg["probes"]["max_iter"],
        test_size=cfg["probes"]["test_size"],
        seed=cfg["experiment"]["seed"],
        speaker_disjoint=cfg["probes"].get("speaker_disjoint", True),
    )
    runner = ProbeRunner(manifest, plans, probe_cfg)
    results = runner.run()

    metrics_df = pd.DataFrame([r.__dict__ for r in results])
    metrics_path = Path(cfg["analysis"]["metrics_csv"])
    ensure_dir(metrics_path.parent)
    metrics_df.to_csv(metrics_path, index=False)

    decision = _decide(metrics_df, cfg)

    writer = StageReportWriter(cfg)
    summary = writer.generate(decision, metrics_df, metrics_path)
    return summary


def _plan_from_dict(plan_cfg: Dict[str, Any]) -> FeaturePlan:
    keys = plan_cfg.get("keys")
    if keys is not None:
        keys = list(keys)
    return FeaturePlan(
        name=plan_cfg["name"],
        root=plan_cfg["root"],
        keys=keys,
        combine=plan_cfg.get("combine", False),
        target=plan_cfg.get("target", "joint"),
    )


def _decide(metrics_df: pd.DataFrame, cfg: Dict[str, Any]) -> Decision:
    margin = cfg["experiment"]["leakage_margin_pp"] / 100.0
    cond_margin = cfg["experiment"]["leakage_conditional_margin_pp"] / 100.0
    min_go = cfg["experiment"]["min_f1_go"]
    min_cond = cfg["experiment"]["min_f1_conditional"]
    text_drop_tol = cfg["experiment"]["text_drop_tolerance_pp"] / 100.0
    min_nogo = 0.40

    accent_rows = metrics_df[metrics_df["target"].isin(["accent", "joint"])]
    backbone_rows = accent_rows[accent_rows["label"].str.startswith("backbone:")]
    ssl_rows = accent_rows[accent_rows["label"].str.startswith("ssl:")]

    candidate_rows = backbone_rows if not backbone_rows.empty else accent_rows

    def _passes(row: pd.Series, min_f1: float, leak_margin: float) -> bool:
        return (
            row.accent_f1 >= min_f1
            and row.leakage_a2s <= row.chance_speaker + leak_margin
            and row.accent_text_drop <= text_drop_tol
        )

    sorted_df = candidate_rows.sort_values("accent_f1", ascending=False)
    for _, row in sorted_df.iterrows():
        if _passes(row, min_go, margin):
            rationale = (
                f"Layer {row.label} passes GO thresholds (F1={row.accent_f1:.2f}, "
                f"leakage={row.leakage_a2s:.2f}, text_drop={row.accent_text_drop:.2f})."
            )
            return Decision(label=row.label, status="GO", rationale=rationale)

    for _, row in sorted_df.iterrows():
        if _passes(row, min_cond, cond_margin):
            rationale = (
                f"Layer {row.label} meets conditional GO (F1={row.accent_f1:.2f}, "
                f"leakage={row.leakage_a2s:.2f}, text_drop={row.accent_text_drop:.2f})."
            )
            return Decision(label=row.label, status="GO_CONDITIONAL", rationale=rationale)

    best_backbone = backbone_rows["accent_f1"].max() if not backbone_rows.empty else float("nan")
    best_ssl = ssl_rows["accent_f1"].max() if not ssl_rows.empty else float("nan")
    if (np.isnan(best_backbone) or best_backbone < min_nogo) and (np.isnan(best_ssl) or best_ssl < min_nogo):
        rationale = "Accent separability is weak in both backbone and SSL baselines (below 0.40)."
        return Decision(label=None, status="NOGO", rationale=rationale)

    rationale = "No backbone layer achieved required accent separability with low leakage and text robustness."
    return Decision(label=None, status="NOGO", rationale=rationale)
