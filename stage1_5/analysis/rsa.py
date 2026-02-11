"""Representational Similarity Analysis utilities."""

# pyright: reportGeneralTypeIssues=false

from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr


def rsa_against_labels(features: np.ndarray, labels: np.ndarray, metric: str = "cosine") -> float:
    """Compute Spearman correlation between feature distances and label matches."""
    label_sim = (labels[:, None] == labels[None, :]).astype(float)
    label_dist = 1.0 - label_sim

    if metric == "cosine":
        norm = np.linalg.norm(features, axis=1, keepdims=True)
        normalized = features / np.clip(norm, 1e-12, None)
        sim = normalized @ normalized.T
        feat_dist = 1.0 - sim
    elif metric == "euclidean":
        diffs = features[:, None, :] - features[None, :, :]
        feat_dist = np.sqrt(np.sum(diffs ** 2, axis=-1))
    else:
        raise ValueError(f"Unsupported metric: {metric}")

    triu = np.triu_indices_from(label_dist, k=1)
    corr, _ = spearmanr(feat_dist[triu], label_dist[triu])
    return float(corr)
