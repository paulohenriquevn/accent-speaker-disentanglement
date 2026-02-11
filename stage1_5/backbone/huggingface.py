"""Example adapter for Hugging Face style TTS models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from ..data import ManifestEntry
from .base import BackboneAdapter


@dataclass
class HFAttachConfig:
    checkpoint: str
    device: str = "cpu"


class HuggingFaceBackboneAdapter(BackboneAdapter):
    def __init__(self, cfg: HFAttachConfig):
        self.cfg = cfg
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.checkpoint)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(cfg.checkpoint).to(cfg.device)
        self.model.eval()

    def prepare_inputs(self, entry: ManifestEntry, text: str) -> Dict[str, torch.Tensor]:
        encoded = self.tokenizer(text, return_tensors="pt")
        return {k: v.to(self.cfg.device) for k, v in encoded.items()}

    def forward(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        with torch.no_grad():
            return self.model(**inputs).logits
