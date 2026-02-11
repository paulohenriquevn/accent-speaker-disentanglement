"""Probe training orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.preprocessing import StandardScaler

from ..analysis import rsa_against_labels, linear_cka
from ..data import Manifest, SpeakerDisjointSplitter
from .dataset import FeaturePlan, FeatureSpec, NPZFeatureStore, expand_plans


@dataclass
class ProbeConfig:
    max_iter: int = 4000
    test_size: float = 0.25
    seed: int = 1337


@dataclass
class ProbeResult:
    label: str
    target: str
    accent_f1: float
    speaker_acc: float
    leakage_a2s: float
    leakage_s2a: float
    chance_accent: float
    chance_speaker: float
    rsa_accent: float
    rsa_speaker: float
    cka_accent: float
    cka_speaker: float


class ProbeRunner:
    def __init__(self, manifest: Manifest, plans: Sequence[FeaturePlan], cfg: ProbeConfig | None = None):
        self.manifest = manifest
        self.cfg = cfg or ProbeConfig()
        self.specs = expand_plans(plans)
        self.df = manifest.to_dataframe()
        self.accents = self.df["accent"].values
        self.speakers = self.df["speaker"].values
        self._store_cache: Dict[Path, NPZFeatureStore] = {}

    def run(self) -> List[ProbeResult]:
        results: List[ProbeResult] = []
        for spec in self.specs:
            store = self._store_cache.setdefault(spec.root, NPZFeatureStore(spec.root))
            matrix = store.build_matrix(self.manifest, spec.keys)
            res = self._evaluate(spec, matrix)
            results.append(res)
        return results

    def _evaluate(self, spec: FeatureSpec, matrix: np.ndarray) -> ProbeResult:
        accent_split = SpeakerDisjointSplitter(self.speakers, seed=self.cfg.seed).train_test_split(self.cfg.test_size)
        accent_f1 = self._train_metric(matrix, self.accents, accent_split.train, accent_split.test, metric="f1")

        idx = np.arange(len(self.speakers))
        train_idx, test_idx = train_test_split(idx, test_size=self.cfg.test_size,
                                               random_state=self.cfg.seed,
                                               stratify=self.accents)
        speaker_acc = self._train_metric(matrix, self.speakers, train_idx, test_idx, metric="acc")

        chance_accent = 1.0 / len(np.unique(self.accents))
        chance_speaker = 1.0 / len(np.unique(self.speakers))
        leakage_a2s = speaker_acc
        leakage_s2a = accent_f1
        rsa_acc = rsa_against_labels(matrix, self.accents)
        rsa_spk = rsa_against_labels(matrix, self.speakers)
        cka_acc = linear_cka(matrix, _one_hot(self.accents))
        cka_spk = linear_cka(matrix, _one_hot(self.speakers))
        return ProbeResult(
            label=spec.label,
            target=spec.target,
            accent_f1=accent_f1,
            speaker_acc=speaker_acc,
            leakage_a2s=leakage_a2s,
            leakage_s2a=leakage_s2a,
            chance_accent=chance_accent,
            chance_speaker=chance_speaker,
            rsa_accent=rsa_acc,
            rsa_speaker=rsa_spk,
            cka_accent=cka_acc,
            cka_speaker=cka_spk,
        )

    def _train_metric(self, X: np.ndarray, y: np.ndarray, train_idx: np.ndarray,
                      test_idx: np.ndarray, metric: str) -> float:
        scaler = StandardScaler(with_mean=False)
        X_train = scaler.fit_transform(X[train_idx])
        X_test = scaler.transform(X[test_idx])
        clf = LogisticRegression(max_iter=self.cfg.max_iter, class_weight="balanced", multi_class="auto")
        clf.fit(X_train, y[train_idx])
        preds = clf.predict(X_test)
        if metric == "f1":
            return float(f1_score(y[test_idx], preds, average="macro"))
        return float(accuracy_score(y[test_idx], preds))


def _one_hot(labels: np.ndarray) -> np.ndarray:
    unique = {label: idx for idx, label in enumerate(sorted(set(labels)))}
    arr = np.zeros((len(labels), len(unique)), dtype=float)
    for i, label in enumerate(labels):
        arr[i, unique[label]] = 1.0
    return arr
