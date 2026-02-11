"""Representational Similarity Analysis utilities."""

from __future__ import annotations

import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr


def rsa_against_labels(features: np.ndarray, labels: np.ndarray, metric: str = "cosine") -> float:
    """Compute Spearman correlation between feature distances and label matches."""

    feat_dist = squareform(pdist(features, metric=metric))
    label_matrix = (labels[:, None] == labels[None, :]).astype(float)
    corr, _ = spearmanr(feat_dist.flatten(), label_matrix.flatten())
    return float(corr)
