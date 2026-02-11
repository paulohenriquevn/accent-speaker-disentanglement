"""Utilities for generating synthetic manifests for tests."""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

from .manifest import ManifestEntry
from ..utils.io import save_jsonl


@dataclass
class SyntheticDatasetConfig:
    accents: List[str]
    speakers_per_accent: int = 3
    texts: int = 5
    sources: tuple[str, ...] = ("real", "syn_S", "syn_rand")


def build_entries(cfg: SyntheticDatasetConfig) -> List[ManifestEntry]:
    entries: List[ManifestEntry] = []
    for accent in cfg.accents:
        for spk in range(cfg.speakers_per_accent):
            speaker = f"spk{accent}{spk:02d}"
            for text_idx in range(cfg.texts):
                text_id = f"t{text_idx:02d}"
                for source in cfg.sources:
                    utt_id = f"{speaker}_{accent}_{text_id}_{source}"
                    entries.append(
                        ManifestEntry(
                            utt_id=utt_id,
                            path=f"wav/{speaker}/{text_id}_{source}.wav",
                            speaker=speaker,
                            accent=accent,
                            text_id=text_id,
                            source=source,
                        )
                    )
    random.shuffle(entries)
    return entries


def dump_manifest(path: str | Path, cfg: SyntheticDatasetConfig) -> None:
    entries = build_entries(cfg)
    save_jsonl(path, [e.__dict__ for e in entries])
