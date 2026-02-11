"""Linear CKA implementation."""

from __future__ import annotations

import numpy as np


def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """Centered Kernel Alignment with linear kernels."""

    X_centered = X - X.mean(axis=0, keepdims=True)
    Y_centered = Y - Y.mean(axis=0, keepdims=True)
    hsic = np.linalg.norm(X_centered.T @ Y_centered, ord="fro") ** 2
    x_var = np.linalg.norm(X_centered.T @ X_centered, ord="fro") ** 2
    y_var = np.linalg.norm(Y_centered.T @ Y_centered, ord="fro") ** 2
    if x_var == 0 or y_var == 0:
        return 0.0
    return float(hsic / np.sqrt(x_var * y_var))
