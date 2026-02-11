"""Acoustic feature extraction (MFCC, F0, speaking rate)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Mapping

import librosa
import numpy as np
from tqdm import tqdm

from ..data import Manifest
from ..utils.io import ensure_dir
from .storage import save_npz_feature


@dataclass
class AcousticFeatureConfig:
    sample_rate: int = 16000
    n_mfcc: int = 40
    hop_length: int = 256
    win_length: int = 1024


class AcousticFeatureExtractor:
    def __init__(self, cfg: AcousticFeatureConfig | None = None):
        self.cfg = cfg or AcousticFeatureConfig()

    def extract_file(self, wav_path: str | Path) -> Dict[str, np.ndarray]:
        y, sr = librosa.load(wav_path, sr=self.cfg.sample_rate)
        duration = len(y) / self.cfg.sample_rate
        if duration == 0:
            raise ValueError(f"Empty audio: {wav_path}")

        mfcc = librosa.feature.mfcc(y=y, sr=self.cfg.sample_rate,
                                    n_mfcc=self.cfg.n_mfcc,
                                    hop_length=self.cfg.hop_length,
                                    n_fft=self.cfg.win_length)
        mfcc_mean = mfcc.mean(axis=1)
        mfcc_std = mfcc.std(axis=1)

        f0 = librosa.yin(y, fmin=60, fmax=400, sr=self.cfg.sample_rate)
        f0 = f0[np.isfinite(f0)]
        if f0.size == 0:
            f0_stats = np.zeros(3)
        else:
            f0_stats = np.array([f0.mean(), f0.std(), np.median(f0)])

        onset_env = librosa.onset.onset_strength(y=y, sr=self.cfg.sample_rate)
        peaks = librosa.util.peak_pick(onset_env, 3, 3, 3, 3, 0.5, 5)
        speaking_rate = len(peaks) / duration

        return {
            "mfcc_mean": mfcc_mean.astype(np.float32),
            "mfcc_std": mfcc_std.astype(np.float32),
            "f0_stats": f0_stats.astype(np.float32),
            "speaking_rate": np.array([speaking_rate], dtype=np.float32),
            "duration": np.array([duration], dtype=np.float32),
        }

    def process_manifest(self, manifest: Manifest, output_dir: str | Path) -> None:
        ensure_dir(output_dir)
        for entry in tqdm(manifest, desc="Acoustic features"):
            feats = self.extract_file(entry.path)
            save_npz_feature(output_dir, entry.utt_id, feats)


def extract_acoustic_cli(manifest_path: Path, output_dir: Path, sample_rate: int = 16000) -> None:
    manifest = Manifest.from_jsonl(manifest_path)
    extractor = AcousticFeatureExtractor(AcousticFeatureConfig(sample_rate=sample_rate))
    extractor.process_manifest(manifest, output_dir)
