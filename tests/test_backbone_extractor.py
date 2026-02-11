from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from stage1_5.data import ManifestEntry
from stage1_5.features.backbone import BackboneFeatureConfig, BackboneFeatureExtractor


class DummyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layer1 = torch.nn.Identity()
        self.layer2 = torch.nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer1(x)
        x = self.layer2(x)
        return x


@dataclass
class DummyAdapter:
    model: DummyModel

    def prepare_inputs(self, entry: ManifestEntry, text: str) -> dict:
        data = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        return {"x": data}

    def forward(self, inputs: dict) -> torch.Tensor:
        return self.model(inputs["x"])


def test_backbone_feature_extractor_pools_over_time() -> None:
    adapter = DummyAdapter(model=DummyModel())
    cfg = BackboneFeatureConfig(layers=["layer1", "layer2"], pooling="mean", strict=True)
    entry = ManifestEntry("u1", "", "s1", "NE", "t1", "real")

    with BackboneFeatureExtractor(adapter, cfg) as extractor:
        feats = extractor.extract_entry(entry, "hello")

    assert set(feats.keys()) == {"layer1", "layer2"}
    assert feats["layer1"].shape == (2,)
    assert np.allclose(feats["layer1"], np.array([3.0, 4.0], dtype=np.float32))


def test_backbone_feature_extractor_non_strict_skips_missing_layer() -> None:
    adapter = DummyAdapter(model=DummyModel())
    cfg = BackboneFeatureConfig(layers=["layer1", "missing"], pooling="mean", strict=False)
    entry = ManifestEntry("u1", "", "s1", "NE", "t1", "real")

    with BackboneFeatureExtractor(adapter, cfg) as extractor:
        feats = extractor.extract_entry(entry, "hello")

    assert "layer1" in feats
    assert "missing" not in feats
