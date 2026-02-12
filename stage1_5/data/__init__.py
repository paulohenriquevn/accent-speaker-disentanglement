"""Data loading helpers for Stage 1.5."""

from .manifest import ManifestEntry, Manifest
from .splits import SpeakerDisjointSplitter, SplitIndices, text_disjoint_split
from .synthetic import SyntheticDatasetConfig, build_entries, dump_manifest
from .dataset_builder import download_archive, build_manifest_from_csv, build_manifest_from_coraa

__all__ = [
    "ManifestEntry",
    "Manifest",
    "SpeakerDisjointSplitter",
    "SplitIndices",
    "text_disjoint_split",
    "SyntheticDatasetConfig",
    "build_entries",
    "dump_manifest",
    "download_archive",
    "build_manifest_from_csv",
    "build_manifest_from_coraa",
]
