"""Capture internal backbone representations via adapters."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Sequence

import json
import numpy as np
from tqdm import tqdm

from ..backbone import BackboneAdapter
from ..backbone.huggingface import HFAttachConfig, HuggingFaceBackboneAdapter
from ..data import Manifest, ManifestEntry
from ..utils.io import ensure_dir
from .pooling import temporal_pooling
from .storage import save_npz_feature


@dataclass
class BackboneFeatureConfig:
    layers: Sequence[str]
    pooling: str = "mean"


class BackboneFeatureExtractor:
    def __init__(self, adapter: BackboneAdapter, cfg: BackboneFeatureConfig):
        self.adapter = adapter
        self.cfg = cfg
        self._buffers: Dict[str, torch.Tensor] = {}
        self._handles = self._register_hooks(cfg.layers)

    def _register_hooks(self, layers: Sequence[str]):
        handles = []
        modules = dict(self.adapter.model.named_modules())
        for name in layers:
            if name not in modules:
                raise ValueError(f"Layer {name} not found in backbone")

            def _capture(_, __, output, layer=name):
                if isinstance(output, tuple):
                    output = output[0]
                self._buffers[layer] = output.detach().cpu()

            handles.append(modules[name].register_forward_hook(_capture))
        return handles

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()

    def extract_entry(self, entry: ManifestEntry, text: str) -> Dict[str, np.ndarray]:
        self._buffers.clear()
        inputs = self.adapter.prepare_inputs(entry, text)
        _ = self.adapter.forward(inputs)
        feats: Dict[str, np.ndarray] = {}
        for layer, tensor in self._buffers.items():
            pooled = temporal_pooling(tensor.squeeze(0), mode=self.cfg.pooling)
            feats[layer] = pooled.numpy().astype(np.float32)
        return feats

    def process(self, manifest: Manifest, texts: Dict[str, str], output_dir: str | Path) -> None:
        ensure_dir(output_dir)
        for entry in tqdm(manifest, desc="Backbone features"):
            text = texts.get(entry.text_id)
            if not text:
                raise KeyError(f"Missing text for id {entry.text_id}")
            feats = self.extract_entry(entry, text)
            save_npz_feature(output_dir, entry.utt_id, feats)


def extract_backbone_cli(manifest_path: Path, text_json: Path, output_dir: Path, checkpoint: str,
                         layers: Sequence[str]) -> None:
    manifest = Manifest.from_jsonl(manifest_path)
    text_entries = json.loads(Path(text_json).read_text())
    texts = {item["text_id"]: item["text"] for item in text_entries}
    adapter = HuggingFaceBackboneAdapter(HFAttachConfig(checkpoint=checkpoint))
    extractor = BackboneFeatureExtractor(adapter, BackboneFeatureConfig(layers=layers))
    extractor.process(manifest, texts, output_dir)
