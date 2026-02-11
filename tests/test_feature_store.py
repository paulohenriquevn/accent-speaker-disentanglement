from __future__ import annotations

import numpy as np
from pathlib import Path

from stage1_5.probes.dataset import FeaturePlan, expand_plans, NPZFeatureStore
from stage1_5.data import Manifest, ManifestEntry


def create_npz(dir_path: Path, utt_id: str, values: dict[str, np.ndarray]) -> None:
    dir_path.mkdir(parents=True, exist_ok=True)
    np.savez(dir_path / f"{utt_id}.npz", **values)


def test_expand_plans_and_matrix(tmp_path: Path) -> None:
    feats_dir = tmp_path / "feats"
    create_npz(feats_dir, "u1", {"k1": np.array([1, 2]), "k2": np.array([3])})
    create_npz(feats_dir, "u2", {"k1": np.array([4, 5]), "k2": np.array([6])})

    plan = FeaturePlan(name="test", root=str(feats_dir), keys=None, combine=False)
    specs = expand_plans([plan])
    assert len(specs) == 2
    store = NPZFeatureStore(specs[0].root)
    manifest = Manifest([ManifestEntry("u1", "", "s1", "NE", "t1", "real"),
                         ManifestEntry("u2", "", "s2", "SE", "t1", "real")])
    matrix = store.build_matrix(manifest, ["k1"])
    assert matrix.shape == (2, 2)
