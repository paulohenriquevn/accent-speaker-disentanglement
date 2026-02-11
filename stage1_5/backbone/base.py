"""Base adapter contract for hooking TTS backbones."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Protocol

import torch

from ..data import ManifestEntry


class BackboneAdapter(Protocol):
    model: torch.nn.Module

    def prepare_inputs(self, entry: ManifestEntry, text: str) -> Dict[str, torch.Tensor | str | int]:
        ...

    def forward(self, inputs: Dict[str, torch.Tensor | str | int]) -> torch.Tensor:
        ...


@dataclass
class LayerCapture:
    name: str
    value: torch.Tensor
