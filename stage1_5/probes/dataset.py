"""Feature loading utilities for probes."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np

from ..data import Manifest


@dataclass
class FeaturePlan:
    name: str
    root: str
    keys: Sequence[str] | None = None
    combine: bool = False
    target: str = "joint"


@dataclass
class FeatureSpec:
    label: str
    root: Path
    keys: Sequence[str]
    target: str


class NPZFeatureStore:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        if not self.root.exists():
            raise FileNotFoundError(self.root)
        self.index = {path.stem: path for path in self.root.glob("*.npz")}
        if not self.index:
            raise RuntimeError(f"No npz files found in {self.root}")
        first = next(iter(self.index.values()))
        with np.load(first) as sample:
            self._keys = list(sample.files)
        self._cache: Dict[str, Dict[str, np.ndarray]] = {}

    @property
    def keys(self) -> List[str]:
        return self._keys

    def _load_entry(self, utt_id: str) -> Dict[str, np.ndarray]:
        if utt_id not in self.index:
            raise KeyError(f"Missing features for {utt_id} in {self.root}")
        if utt_id not in self._cache:
            with np.load(self.index[utt_id]) as data:
                self._cache[utt_id] = {k: data[k] for k in data.files}
        return self._cache[utt_id]

    def build_matrix(self, manifest: Manifest, keys: Sequence[str]) -> np.ndarray:
        vectors: List[np.ndarray] = []
        for entry in manifest:
            data = self._load_entry(entry.utt_id)
            missing = [k for k in keys if k not in data]
            if missing:
                raise KeyError(f"Missing keys {missing} for {entry.utt_id} in {self.root}")
            feats = [np.atleast_1d(data[k]).ravel() for k in keys]
            vectors.append(np.concatenate(feats).astype(np.float32))
        return np.vstack(vectors)


def expand_plans(plans: Sequence[FeaturePlan]) -> List[FeatureSpec]:
    specs: List[FeatureSpec] = []
    for plan in plans:
        store = NPZFeatureStore(plan.root)
        keys = list(plan.keys) if plan.keys else store.keys
        if plan.combine:
            specs.append(FeatureSpec(label=plan.name, root=store.root, keys=keys, target=plan.target))
        else:
            for key in keys:
                specs.append(FeatureSpec(label=f"{plan.name}:{key}", root=store.root, keys=[key], target=plan.target))
    return specs
