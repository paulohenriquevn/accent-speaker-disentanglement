"""Example adapter for Hugging Face style TTS models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
from transformers import AutoConfig, AutoModel, AutoModelForSeq2SeqLM, AutoProcessor, AutoTokenizer

from ..data import ManifestEntry
from .base import BackboneAdapter


@dataclass
class HFAttachConfig:
    checkpoint: str
    device: str = "cpu"


class HuggingFaceBackboneAdapter(BackboneAdapter):
    def __init__(self, cfg: HFAttachConfig):
        self.cfg = cfg
        self.tokenizer = None
        self.processor = None

        qwen_tts_available = False
        try:
            from qwen_tts.core.models import Qwen3TTSConfig, Qwen3TTSForConditionalGeneration, Qwen3TTSProcessor

            AutoConfig.register("qwen3_tts", Qwen3TTSConfig)
            AutoModel.register(Qwen3TTSConfig, Qwen3TTSForConditionalGeneration)
            AutoProcessor.register(Qwen3TTSConfig, Qwen3TTSProcessor)
            qwen_tts_available = True
        except Exception:
            qwen_tts_available = False

        config = AutoConfig.from_pretrained(cfg.checkpoint)
        if getattr(config, "model_type", None) == "qwen3_tts":
            if not qwen_tts_available:
                raise RuntimeError(
                    "Checkpoint requires qwen-tts. Install with: pip install -U qwen-tts"
                )
            self.processor = AutoProcessor.from_pretrained(cfg.checkpoint, fix_mistral_regex=True)
            self.model = AutoModel.from_pretrained(cfg.checkpoint).to(cfg.device)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(cfg.checkpoint)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(cfg.checkpoint).to(cfg.device)

        self.model.eval()

    def prepare_inputs(self, entry: ManifestEntry, text: str) -> Dict[str, torch.Tensor]:
        if self.processor is not None:
            encoded = self.processor(text=text, return_tensors="pt", padding=True)
        else:
            encoded = self.tokenizer(text, return_tensors="pt")
        return {k: v.to(self.cfg.device) for k, v in encoded.items()}

    def forward(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        with torch.no_grad():
            return self.model(**inputs)
