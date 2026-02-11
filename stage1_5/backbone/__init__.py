"""Adapters for hooking into TTS backbones."""

from .base import BackboneAdapter
from .huggingface import HuggingFaceBackboneAdapter

__all__ = ["BackboneAdapter", "HuggingFaceBackboneAdapter"]
