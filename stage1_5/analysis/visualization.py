"""Plotting helpers for metrics heatmaps and comparison charts."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def metric_heatmap(df: pd.DataFrame, metrics: Iterable[str], output_path: str | Path,
                   title: str) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    matrix = df.set_index("label")[list(metrics)]
    plt.figure(figsize=(8, 0.4 * len(matrix.index) + 2))
    sns.heatmap(matrix, annot=True, cmap="viridis", fmt=".2f")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    return output_path


def domain_comparison_chart(
    metrics_df: pd.DataFrame,
    output_path: str | Path,
    metric: str = "accent_f1",
    domains: Sequence[str] = ("acoustic:", "ssl:", "backbone:"),
    title: str = "Accent F1 — Real Audio vs SSL vs Backbone",
) -> Path:
    """Create a grouped bar chart comparing a metric across feature domains.

    Each bar represents a probe row (label).  Bars are colour-coded by
    domain so that the reader can visually compare Real-audio features,
    SSL upstream features and Backbone TTS features side-by-side.

    Args:
        metrics_df: DataFrame with at least ``label`` and ``metric`` columns.
        output_path: Where to save the PNG figure.
        metric: Column name to plot on the y-axis.
        domains: Label prefixes that identify each domain group.
        title: Figure title.

    Returns:
        Path to the saved figure.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Only keep rows whose label starts with one of the known domain prefixes
    # and that contain the requested metric column.
    if metric not in metrics_df.columns:
        # Nothing to plot — return empty figure with a note
        plt.figure(figsize=(6, 3))
        plt.text(0.5, 0.5, f"Metric '{metric}' not available", ha="center", va="center")
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        return output_path

    # Filter to accent/joint targets (comparison only makes sense for accent probes)
    df = metrics_df.copy()
    if "target" in df.columns:
        df = df[df["target"].isin(["accent", "joint"])]

    # Assign a domain label based on prefix
    def _domain_label(label: str) -> str:
        for prefix in domains:
            if label.startswith(prefix):
                return prefix.rstrip(":")
        return "other"

    df["domain"] = df["label"].apply(_domain_label)
    df = df[df["domain"] != "other"]

    if df.empty:
        plt.figure(figsize=(6, 3))
        plt.text(0.5, 0.5, "No domain data available for comparison", ha="center", va="center")
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        return output_path

    # Sort: acoustic first, then ssl, then backbone
    domain_order = [d.rstrip(":") for d in domains if d.rstrip(":") in df["domain"].values]
    palette = {"acoustic": "#4c72b0", "ssl": "#55a868", "backbone": "#c44e52"}

    fig, ax = plt.subplots(figsize=(max(6, len(df) * 0.8), 5))
    sns.barplot(
        data=df,
        x="label",
        y=metric,
        hue="domain",
        hue_order=domain_order,
        palette={k: palette.get(k, "#999999") for k in domain_order},
        dodge=False,
        ax=ax,
    )
    ax.set_title(title)
    ax.set_xlabel("Feature Space")
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_ylim(0, 1)
    plt.xticks(rotation=45, ha="right")
    plt.legend(title="Domain")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    return output_path
