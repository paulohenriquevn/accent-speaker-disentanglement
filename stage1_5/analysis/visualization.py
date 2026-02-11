"""Plotting helpers for metrics heatmaps."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

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
