"""Manifest utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, List, Sequence

import pandas as pd

from ..utils.io import load_jsonl


@dataclass(frozen=True)
class ManifestEntry:
    utt_id: str
    path: str
    speaker: str
    accent: str
    text_id: str
    source: str


class Manifest:
    """Thin wrapper with helpers for filtering and tabular conversion."""

    def __init__(self, entries: Sequence[ManifestEntry]):
        if not entries:
            raise ValueError("Manifest is empty")
        self._entries: List[ManifestEntry] = list(entries)

    @classmethod
    def from_jsonl(cls, path: str | Path) -> "Manifest":
        entries = [ManifestEntry(**row) for row in load_jsonl(path)]
        return cls(entries)

    def __iter__(self) -> Iterator[ManifestEntry]:
        return iter(self._entries)

    def __len__(self) -> int:
        return len(self._entries)

    def filter(self, *, source: str | None = None, accents: Sequence[str] | None = None) -> "Manifest":
        items = self._entries
        if source is not None:
            items = [e for e in items if e.source == source]
        if accents is not None:
            accents = list(accents)
            items = [e for e in items if e.accent in accents]
        return Manifest(items)

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([e.__dict__ for e in self._entries])

    def speakers(self) -> List[str]:
        return sorted({e.speaker for e in self._entries})

    def accents(self) -> List[str]:
        return sorted({e.accent for e in self._entries})
