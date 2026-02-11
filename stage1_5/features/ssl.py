"""SSL feature extraction via Hugging Face Transformers.

Extracts per-layer pooled embeddings (mean over time) from WavLM/HuBERT-like SSL models.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import torch
import torchaudio
from tqdm import tqdm

from ..data import Manifest
from ..utils.io import ensure_dir
from .storage import save_npz_feature


PathLike = Union[str, Path]


MODEL_ALIASES = {
    "wavlm_large": "microsoft/wavlm-large",
    "wavlm_base": "microsoft/wavlm-base",
    "hubert_large": "facebook/hubert-large-ls960-ft",
    "hubert_base": "facebook/hubert-base-ls960",
}


@dataclass(frozen=True)
class SSLConfig:
    model: str = "wavlm_large"
    layers: Optional[List[int]] = None  # layer indices to export; None = all
    sample_rate: int = 16000
    device: str = "cpu"
    torch_dtype: Optional[str] = None  # "float16" | "bfloat16" | "float32" | None


class SSLFeatureExtractor:
    def __init__(self, cfg: Optional[SSLConfig] = None):
        try:
            from transformers import (
                AutoConfig,
                AutoFeatureExtractor,
                AutoModelForAudioFeatureExtraction,
                AutoProcessor,
            )
        except ModuleNotFoundError as exc:  # pragma: no cover
            raise RuntimeError(
                "transformers is required for SSL extraction; install via `pip install transformers`."
            ) from exc

        self.cfg = cfg or SSLConfig()
        model_id = MODEL_ALIASES.get(self.cfg.model, self.cfg.model)

        config = AutoConfig.from_pretrained(model_id)
        config.output_hidden_states = True

        dtype = None
        if self.cfg.torch_dtype:
            dtype = getattr(torch, self.cfg.torch_dtype)

        # Prefer the audio feature extraction model class (more consistent outputs for SSL models)
        self.model = AutoModelForAudioFeatureExtraction.from_pretrained(
            model_id, config=config, torch_dtype=dtype
        ).to(self.cfg.device)
        self.model.eval()

        # Processor is optional; fall back to FeatureExtractor for wav2vec2-family
        try:
            self.processor = AutoProcessor.from_pretrained(model_id)
        except (ValueError, OSError):
            self.processor = AutoFeatureExtractor.from_pretrained(model_id)

    def _load_waveform(self, wav_path: PathLike) -> torch.Tensor:
        signal, sr = torchaudio.load(str(wav_path))  # (channels, time)
        if signal.size(0) > 1:
            signal = signal.mean(dim=0, keepdim=True)
        if sr != self.cfg.sample_rate:
            signal = torchaudio.functional.resample(signal, sr, self.cfg.sample_rate)
        # Return 1D float32 tensor (time,)
        return signal.squeeze(0).to(torch.float32)

    def extract_file(self, wav_path: PathLike) -> Dict[str, np.ndarray]:
        waveform = self._load_waveform(wav_path)

        # HF processors expect numpy float32 arrays
        wav_np = np.ascontiguousarray(waveform.cpu().numpy(), dtype=np.float32)

        inputs = self.processor(
            wav_np,
            sampling_rate=self.cfg.sample_rate,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.cfg.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

        hidden_states = getattr(outputs, "hidden_states", None)
        if hidden_states is None:
            raise RuntimeError("Model did not return hidden states; ensure output_hidden_states=True")

        feats: Dict[str, np.ndarray] = {}
        selected_layers = self.cfg.layers or list(range(len(hidden_states)))

        # Validate indices early (friendlier errors)
        max_layer = len(hidden_states) - 1
        for layer_id in selected_layers:
            if layer_id < 0 or layer_id > max_layer:
                raise ValueError(f"Invalid layer {layer_id}. Valid range: [0, {max_layer}]")

        for layer_id in selected_layers:
            # hidden_states[layer] shape: (batch=1, time, hidden)
            tensor = hidden_states[layer_id].squeeze(0)
            pooled = tensor.mean(dim=0).detach().to("cpu").numpy().astype(np.float32)
            feats[f"layer_{layer_id}"] = pooled

        return feats

    def process_manifest(self, manifest: Manifest, output_dir: PathLike) -> None:
        ensure_dir(output_dir)
        for entry in tqdm(manifest, desc="SSL features"):
            feats = self.extract_file(entry.path)
            save_npz_feature(output_dir, entry.utt_id, feats)


def extract_ssl_cli(manifest_path: Path, output_dir: Path, model: str = "wavlm_large") -> None:
    manifest = Manifest.from_jsonl(manifest_path)
    extractor = SSLFeatureExtractor(SSLConfig(model=model))
    extractor.process_manifest(manifest, output_dir)
