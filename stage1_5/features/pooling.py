"""Pooling utilities for temporal representations."""

from __future__ import annotations

import torch


def temporal_pooling(x: torch.Tensor, mode: str = "mean") -> torch.Tensor:
    if x.ndim == 1:
        return x
    if mode == "mean":
        return x.mean(dim=0)
    if mode == "max":
        return x.max(dim=0).values
    if mode == "mean_std":
        mean = x.mean(dim=0)
        std = x.std(dim=0)
        return torch.cat([mean, std], dim=-1)
    raise ValueError(f"Unsupported pooling mode: {mode}")
