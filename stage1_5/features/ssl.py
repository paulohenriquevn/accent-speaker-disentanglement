"""SSL feature extraction via S3PRL hub."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torchaudio
from tqdm import tqdm

if not hasattr(torchaudio, "set_audio_backend"):
    def _set_audio_backend(_: str) -> None:  # pragma: no cover
        pass

    torchaudio.set_audio_backend = _set_audio_backend  # type: ignore[attr-defined]

from ..data import Manifest
from ..utils.io import ensure_dir
from .storage import save_npz_feature


@dataclass
class SSLConfig:
    model: str = "wavlm_large"
    layers: List[int] | None = None
    sample_rate: int = 16000
    device: str = "cpu"


class SSLFeatureExtractor:
    def __init__(self, cfg: SSLConfig | None = None):
        import importlib
        try:
            hub = importlib.import_module("s3prl.hub")
        except ModuleNotFoundError as exc:
            if "torchaudio.sox_effects" in str(exc):
                raise RuntimeError(
                    "torchaudio.sox_effects is unavailable; install torchaudio with SoX support (pip install 'torchaudio[sox]' and ensure libsox is present)"
                ) from exc
            raise

        self.cfg = cfg or SSLConfig()
        upstream_cls = getattr(hub, self.cfg.model)
        try:
            self.model = upstream_cls().to(self.cfg.device)
        except ModuleNotFoundError as exc:
            if "torchaudio.sox_effects" in str(exc):
                raise RuntimeError("torchaudio.sox_effects is unavailable; run `pip install 'torchaudio[sox]'` or install libsox") from exc
            raise
        self.model.eval()

    def extract_file(self, wav_path: str | Path) -> Dict[str, np.ndarray]:
        signal, sr = torchaudio.load(str(wav_path))
        if sr != self.cfg.sample_rate:
            signal = torchaudio.functional.resample(signal, sr, self.cfg.sample_rate)
        signal = signal.to(self.cfg.device)
        with torch.no_grad():
            outputs = self.model([signal])
        feats = {}
        selected_layers = self.cfg.layers or list(range(len(outputs["hidden_states"])) )
        for layer_id in selected_layers:
            tensor = outputs["hidden_states"][layer_id].squeeze(0)
            pooled = tensor.mean(dim=0).cpu().numpy().astype(np.float32)
            feats[f"layer_{layer_id}"] = pooled
        return feats

    def process_manifest(self, manifest: Manifest, output_dir: str | Path) -> None:
        ensure_dir(output_dir)
        for entry in tqdm(manifest, desc="SSL features"):
            feats = self.extract_file(entry.path)
            save_npz_feature(output_dir, entry.utt_id, feats)


def extract_ssl_cli(manifest_path: Path, output_dir: Path, model: str = "wavlm_large") -> None:
    manifest = Manifest.from_jsonl(manifest_path)
    extractor = SSLFeatureExtractor(SSLConfig(model=model))
    extractor.process_manifest(manifest, output_dir)
