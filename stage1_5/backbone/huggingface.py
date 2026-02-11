"""Example adapter for Hugging Face style TTS models."""

from __future__ import annotations

# pyright: ignore

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, cast

import torch
from transformers import AutoConfig, AutoModel, AutoModelForSeq2SeqLM, AutoProcessor, AutoTokenizer

from ..data import ManifestEntry


@dataclass
class HFAttachConfig:
    checkpoint: str
    device: str = "cpu"
    dtype: str | None = None
    attn_implementation: str | None = None
    generation_mode: str = "custom_voice"
    generation_language: str = "Portuguese"
    generation_speaker: str = "ryan"
    generation_instruct: str | None = None
    generation_max_new_tokens: int = 256


class HuggingFaceBackboneAdapter:
    def __init__(self, cfg: HFAttachConfig):
        self.cfg = cfg
        self.tokenizer: Optional[Callable[..., Dict[str, torch.Tensor]]] = None
        self.processor: Optional[Callable[..., Dict[str, torch.Tensor]]] = None
        self._encoder: Optional[Callable[..., Dict[str, torch.Tensor]]] = None
        self._model_type = None

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
        self._model_type = getattr(config, "model_type", None)
        model_kwargs: Dict[str, Any] = {}
        if cfg.dtype:
            model_kwargs["torch_dtype"] = getattr(torch, cfg.dtype)
        if cfg.attn_implementation:
            model_kwargs["attn_implementation"] = cfg.attn_implementation
        if self._model_type == "qwen3_tts":
            if not qwen_tts_available:
                raise RuntimeError(
                    "Checkpoint requires qwen-tts. Install with: pip install -U qwen-tts"
                )
            self.processor = AutoProcessor.from_pretrained(cfg.checkpoint, fix_mistral_regex=True)
            self.model = AutoModel.from_pretrained(cfg.checkpoint, **model_kwargs).to(cfg.device)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(cfg.checkpoint)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(cfg.checkpoint, **model_kwargs).to(cfg.device)

        if self.processor is not None:
            self._encoder = self.processor
        elif self.tokenizer is not None:
            self._encoder = self.tokenizer
        else:
            raise RuntimeError("Text encoder is not initialized")

        self.model.eval()

    def prepare_inputs(self, entry: ManifestEntry, text: str) -> Dict[str, torch.Tensor]:
        if self._model_type == "qwen3_tts":
            return {
                "mode": self.cfg.generation_mode,
                "text": text,
                "language": self.cfg.generation_language,
                "speaker": self.cfg.generation_speaker,
                "instruct": self.cfg.generation_instruct,
                "max_new_tokens": self.cfg.generation_max_new_tokens,
            }
        if self._encoder is None:
            raise RuntimeError("Text encoder is not initialized")
        encoder = cast(Callable[..., Dict[str, torch.Tensor]], self._encoder)
        encoded = encoder(text=text, return_tensors="pt", padding=True)
        return {k: v.to(self.cfg.device) for k, v in encoded.items()}

    def forward(self, inputs):
        with torch.no_grad():
            if self._model_type == "qwen3_tts":
                # inputs aqui NÃO é {"input_ids": ..., ...}
                # inputs precisa ser algo como:
                # {"mode":"custom_voice","text":..., "language":..., "speaker":..., "instruct":...}
                mode = inputs.get("mode", "custom_voice")
                if mode == "custom_voice":
                    self.model.generate_custom_voice(
                        text=inputs["text"],
                        language=inputs.get("language", "Portuguese"),
                        speaker=inputs.get("speaker", "ryan"),
                        instruct=inputs.get("instruct"),
                        non_streaming_mode=True,
                        max_new_tokens=inputs.get("max_new_tokens", 256),
                    )
                    return torch.empty(0)
                raise ValueError(f"Unsupported qwen3_tts mode: {mode}")

            return self.model(**inputs)

    def resolve_layer(self, alias: str) -> torch.nn.Module:
        if self._model_type != "qwen3_tts":
            raise KeyError(alias)

        modules = dict(self.model.named_modules())
        if alias in modules:
            return modules[alias]

        if alias == "text_encoder_out":
            candidate = "talker.text_projection"
            if candidate in modules:
                return modules[candidate]
        if alias == "pre_vocoder":
            candidate = "talker.codec_head"
            if candidate in modules:
                return modules[candidate]

        if alias.startswith("decoder_block_"):
            suffix = alias.split("decoder_block_", 1)[1]
            if suffix.isdigit():
                idx = int(suffix)
                candidate = f"talker.model.layers.{idx}"
                if candidate in modules:
                    return modules[candidate]

        raise KeyError(alias)
