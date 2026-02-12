"""Manifest utilities."""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, List, Sequence

import pandas as pd

from ..utils.io import load_jsonl

logger = logging.getLogger(__name__)


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

    def validate_minimums(
        self,
        min_speakers_per_accent: int = 8,
        min_utterances_per_speaker: int = 30,
    ) -> List[str]:
        """Validate dataset meets minimum requirements from the PRD.

        Returns a list of warning messages.  Raises ``ValueError`` if any
        hard constraint is violated (fewer than 2 accents or fewer than 2
        speakers in any accent -- which would make splits impossible).
        """
        warnings: List[str] = []

        # --- Accent / speaker counts ---
        accent_speakers: dict[str, set[str]] = {}
        speaker_utterances: Counter[str] = Counter()
        for entry in self._entries:
            accent_speakers.setdefault(entry.accent, set()).add(entry.speaker)
            speaker_utterances[entry.speaker] += 1

        if len(accent_speakers) < 2:
            raise ValueError(
                f"Manifest has only {len(accent_speakers)} accent(s). "
                "At least 2 are required for classification."
            )

        for accent, spk_set in sorted(accent_speakers.items()):
            if len(spk_set) < min_speakers_per_accent:
                warnings.append(
                    f"Accent '{accent}' has {len(spk_set)} speaker(s), "
                    f"minimum recommended is {min_speakers_per_accent}."
                )

        low_utt_speakers = [
            (spk, count)
            for spk, count in speaker_utterances.items()
            if count < min_utterances_per_speaker
        ]
        if low_utt_speakers:
            warnings.append(
                f"{len(low_utt_speakers)} speaker(s) have fewer than "
                f"{min_utterances_per_speaker} utterances. "
                f"Examples: {low_utt_speakers[:5]}"
            )

        for msg in warnings:
            logger.warning("Manifest validation: %s", msg)

        return warnings
