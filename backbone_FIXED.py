# backbone.py
"""Capture internal backbone representations via adapters (robust hooks)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional, Sequence

import numpy as np
import torch
from tqdm import tqdm

from ..backbone import BackboneAdapter
from ..backbone.huggingface import HFAttachConfig, HuggingFaceBackboneAdapter
from ..data import Manifest, ManifestEntry
from ..utils.io import ensure_dir
from .pooling import temporal_pooling
from .storage import save_npz_feature


@dataclass(frozen=True)
class BackboneFeatureConfig:
    # IMPORTANT: keep order (it defines feature key ordering & reproducibility)
    layers: Sequence[str]
    pooling: str = "mean"  # mean|max|first|last (whatever your temporal_pooling supports)
    strict: bool = True    # if True: hard-fail on missing layers / non-tensor outputs


class BackboneFeatureExtractor:
    """
    Extract fixed-size representations from internal backbone activations.

    Contract with BackboneAdapter (minimum):
      - adapter.model: torch.nn.Module (or at least has named_modules())
      - adapter.prepare_inputs(entry, text) -> Any
      - adapter.forward(inputs) -> Any (must execute hooked modules)

    Optional adapter extensions supported:
      - adapter.resolve_layer(alias: str) -> Optional[torch.nn.Module]
        (useful for Qwen3-TTS where you want stable aliases instead of fragile module names)
    """

    def __init__(self, adapter: BackboneAdapter, cfg: BackboneFeatureConfig):
        self.adapter = adapter
        self.cfg = cfg
        self._buffers: Dict[str, torch.Tensor] = {}
        self._handles: list[torch.utils.hooks.RemovableHandle] = []
        self._register_hooks(cfg.layers)

    # --- lifecycle ---------------------------------------------------------

    def __enter__(self) -> "BackboneFeatureExtractor":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def close(self) -> None:
        # idempotent close
        for h in self._handles:
            try:
                h.remove()
            except Exception:
                pass
        self._handles.clear()

    # --- internals ---------------------------------------------------------

    def _resolve_module(self, layer: str) -> Optional[torch.nn.Module]:
        """
        Resolve layer alias to module. Returns None if not found.
        
        Args:
            layer: Layer name or alias
            
        Returns:
            Optional[torch.nn.Module]: Resolved module or None
        """
        # Prefer adapter-provided resolver (aliases -> modules)
        resolver = getattr(self.adapter, "resolve_layer", None)
        if callable(resolver):
            try:
                mod = resolver(layer)
                if mod is not None:
                    if not isinstance(mod, torch.nn.Module):
                        raise TypeError(f"adapter.resolve_layer({layer!r}) must return torch.nn.Module, got {type(mod)}")
                    return mod
            except Exception as e:
                if self.cfg.strict:
                    raise RuntimeError(f"Error resolving layer {layer!r}: {e}") from e
                return None

        # Fallback: direct lookup by name in named_modules()
        modules = dict(self.adapter.model.named_modules())
        if layer not in modules:
            msg = (
                f"Layer {layer!r} not found in backbone named_modules().\n"
                f"Available modules: {list(modules.keys())[:10]}...\n"
                f"Tip: print available module names or implement adapter.resolve_layer(alias)."
            )
            if self.cfg.strict:
                raise ValueError(msg)
            return None
        return modules[layer]

    @staticmethod
    def _pick_tensor_from_output(output) -> Optional[torch.Tensor]:
        """
        Try hard to extract a torch.Tensor from common output structures:
        - Tensor
        - tuple/list (take first tensor-ish)
        - dict/Mapping (prefer 'last_hidden_state', else first tensor value)
        - HF ModelOutput (Mapping + attributes)
        """
        if isinstance(output, torch.Tensor):
            return output

        if isinstance(output, (tuple, list)):
            for item in output:
                t = BackboneFeatureExtractor._pick_tensor_from_output(item)
                if isinstance(t, torch.Tensor):
                    return t
            return None

        if isinstance(output, Mapping):
            if "last_hidden_state" in output and isinstance(output["last_hidden_state"], torch.Tensor):
                return output["last_hidden_state"]
            # common alternates
            for k in ("hidden_states", "encoder_last_hidden_state", "x", "y"):
                if k in output and isinstance(output[k], torch.Tensor):
                    return output[k]
            # fallback: first tensor value
            for v in output.values():
                t = BackboneFeatureExtractor._pick_tensor_from_output(v)
                if isinstance(t, torch.Tensor):
                    return t
            return None

        # Some objects behave like HF ModelOutput but aren't Mapping
        for attr in ("last_hidden_state", "hidden_states"):
            if hasattr(output, attr):
                v = getattr(output, attr)
                t = BackboneFeatureExtractor._pick_tensor_from_output(v)
                if isinstance(t, torch.Tensor):
                    return t

        return None

    def _register_hooks(self, layers: Sequence[str]) -> None:
        """Register forward hooks for specified layers."""
        for layer in layers:
            module = self._resolve_module(layer)
            if module is None:
                if self.cfg.strict:
                    raise ValueError(f"Could not resolve layer {layer!r}")
                continue

            def _capture(_module, _inputs, output, layer_name=layer):
                t = self._pick_tensor_from_output(output)
                if t is None:
                    if self.cfg.strict:
                        raise RuntimeError(
                            f"Hook for layer {layer_name!r} did not receive a torch.Tensor output. "
                            f"Got type: {type(output)}"
                        )
                    return
                # Detach to CPU immediately to avoid holding GPU graph / VRAM
                self._buffers[layer_name] = t.detach().cpu()

            self._handles.append(module.register_forward_hook(_capture))

    def _pool_tensor(self, t: torch.Tensor) -> np.ndarray:
        """
        Pool temporal dimension of tensor.
        
        Expected shapes (common):
          - (B, T, H)  -> pool over T
          - (T, H)     -> pool over T
          - (H,)       -> already pooled (no-op)
        """
        if t.ndim == 3:
            # (B, T, H) -> assume B=1
            if t.size(0) != 1 and self.cfg.strict:
                raise RuntimeError(f"Expected batch size 1, got {t.shape}")
            t2 = t.squeeze(0)  # (T, H)
            pooled = temporal_pooling(t2, mode=self.cfg.pooling)
            return pooled.numpy().astype(np.float32)

        if t.ndim == 2:
            pooled = temporal_pooling(t, mode=self.cfg.pooling)
            return pooled.numpy().astype(np.float32)

        if t.ndim == 1:
            return t.numpy().astype(np.float32)

        raise RuntimeError(f"Unsupported tensor shape for pooling: {tuple(t.shape)}")

    # --- public API --------------------------------------------------------

    def extract_entry(self, entry: ManifestEntry, text: str) -> Dict[str, np.ndarray]:
        """
        Extract features for a single manifest entry.
        
        Args:
            entry: Manifest entry with metadata
            text: Text to process
            
        Returns:
            Dict mapping layer names to pooled feature vectors
        """
        self._buffers.clear()

        try:
            inputs = self.adapter.prepare_inputs(entry, text)
            _ = self.adapter.forward(inputs)
        except Exception as e:
            raise RuntimeError(f"Error processing entry {entry.utt_id}: {e}") from e

        feats: Dict[str, np.ndarray] = {}
        # deterministic order: follow cfg.layers
        for layer in self.cfg.layers:
            if layer not in self._buffers:
                if self.cfg.strict:
                    raise RuntimeError(
                        f"No activation captured for layer {layer!r}. "
                        f"Either the layer isn't executed in forward(), or the name/alias is wrong."
                    )
                continue
            try:
                feats[layer] = self._pool_tensor(self._buffers[layer])
            except Exception as e:
                raise RuntimeError(f"Error pooling features for layer {layer!r}: {e}") from e
        return feats

    def process(self, manifest: Manifest, texts: Dict[str, str], output_dir: str | Path) -> None:
        """
        Process entire manifest and save features.
        
        Args:
            manifest: Input manifest
            texts: Mapping from text_id to text content
            output_dir: Output directory for NPZ files
        """
        ensure_dir(output_dir)
        failed = []
        
        for entry in tqdm(manifest, desc="Backbone features"):
            text = texts.get(entry.text_id)
            if not text:
                if self.cfg.strict:
                    raise KeyError(f"Missing text for id {entry.text_id}")
                failed.append(entry.utt_id)
                continue
            
            try:
                feats = self.extract_entry(entry, text)
                save_npz_feature(output_dir, entry.utt_id, feats)
            except Exception as e:
                if self.cfg.strict:
                    raise
                failed.append(entry.utt_id)
                print(f"Warning: Failed to process {entry.utt_id}: {e}")
        
        if failed:
            print(f"Warning: {len(failed)} entries failed: {failed[:5]}...")


def extract_backbone_cli(
    manifest_path: Path,
    text_json: Path,
    output_dir: Path,
    checkpoint: str,
    layers: Sequence[str],
    device: str = "cpu",
    dtype: str | None = None,
    attn_implementation: str | None = None,
    generation_mode: str = "custom_voice",
    generation_language: str = "Portuguese",
    generation_speaker: str = "ryan",
    generation_instruct: str | None = None,
    generation_max_new_tokens: int = 256,
    pooling: str = "mean",
    strict: bool = True,
) -> None:
    """
    CLI entrypoint for backbone feature extraction.
    """
    if len(layers) == 1 and " " in layers[0]:
        layers = [l for l in layers[0].split(" ") if l]

    manifest = Manifest.from_jsonl(manifest_path)
    text_entries = json.loads(Path(text_json).read_text())
    texts = {item["text_id"]: item["text"] for item in text_entries}

    adapter = HuggingFaceBackboneAdapter(
        HFAttachConfig(
            checkpoint=checkpoint,
            device=device,
            dtype=dtype,
            attn_implementation=attn_implementation,
            generation_mode=generation_mode,
            generation_language=generation_language,
            generation_speaker=generation_speaker,
            generation_instruct=generation_instruct,
            generation_max_new_tokens=generation_max_new_tokens,
        )
    )
    cfg = BackboneFeatureConfig(layers=layers, pooling=pooling, strict=strict)

    with BackboneFeatureExtractor(adapter, cfg) as extractor:
        extractor.process(manifest, texts, output_dir)
