"""ECAPA/x-vector embeddings via SpeechBrain."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torchaudio
from tqdm import tqdm

# SpeechBrain <0.6 expects torchaudio.list_audio_backends which was removed in torchaudio>=2.1
if not hasattr(torchaudio, "list_audio_backends"):
    def _list_audio_backends_stub():  # pragma: no cover - environment-dependent
        return ["sox_io"]

    torchaudio.list_audio_backends = _list_audio_backends_stub  # type: ignore[attr-defined]

try:  # huggingface_hub >=0.21 removed use_auth_token
    import huggingface_hub
except ImportError:  # pragma: no cover
    huggingface_hub = None  # type: ignore
else:
    import inspect

    sig = inspect.signature(huggingface_hub.hf_hub_download)
    if "use_auth_token" not in sig.parameters:
        _hf_hub_download = huggingface_hub.hf_hub_download

        def hf_hub_download_wrapper(*args, use_auth_token=None, **kwargs):  # pragma: no cover
            if use_auth_token is not None:
                kwargs.setdefault("token", use_auth_token)
            return _hf_hub_download(*args, **kwargs)

        huggingface_hub.hf_hub_download = hf_hub_download_wrapper  # type: ignore[attr-defined]

from speechbrain.inference.speaker import EncoderClassifier

from ..data import Manifest
from ..utils.io import ensure_dir
from .storage import save_npz_feature


@dataclass
class ECAPAConfig:
    model_source: str = "speechbrain/spkrec-ecapa-voxceleb"
    savedir: str = "artifacts/models/ecapa"
    sample_rate: int = 16000
    device: str = "cpu"


class ECAPAExtractor:
    def __init__(self, cfg: ECAPAConfig | None = None):
        self.cfg = cfg or ECAPAConfig()
        self.classifier = EncoderClassifier.from_hparams(source=self.cfg.model_source,
                                                         savedir=self.cfg.savedir,
                                                         run_opts={"device": self.cfg.device})

    def extract_file(self, wav_path: str | Path) -> Dict[str, np.ndarray]:
        signal, sr = torchaudio.load(str(wav_path))
        if sr != self.cfg.sample_rate:
            signal = torchaudio.functional.resample(signal, sr, self.cfg.sample_rate)
        signal = signal.to(self.cfg.device)
        with torch.no_grad():
            emb = self.classifier.encode_batch(signal).squeeze(0).cpu().numpy().astype(np.float32)
        return {"ecapa": emb}

    def process_manifest(self, manifest: Manifest, output_dir: str | Path) -> None:
        ensure_dir(output_dir)
        for entry in tqdm(manifest, desc="ECAPA embeddings"):
            feats = self.extract_file(entry.path)
            save_npz_feature(output_dir, entry.utt_id, feats)


def extract_ecapa_cli(
    manifest_path: Path,
    output_dir: Path,
    device: str = "cpu",
    max_per_speaker: int | None = None,
) -> None:
    manifest = Manifest.from_jsonl(manifest_path)
    if max_per_speaker is not None:
        manifest = manifest.subsample_per_speaker(max_per_speaker)
    extractor = ECAPAExtractor(ECAPAConfig(device=device))
    extractor.process_manifest(manifest, output_dir)
