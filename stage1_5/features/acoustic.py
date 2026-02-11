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

        frame_length = self.cfg.win_length
        hop = self.cfg.hop_length
        if len(y) < frame_length:
            padded = np.pad(y, (0, frame_length - len(y)))
        else:
            padded = y
        frames = np.lib.stride_tricks.sliding_window_view(padded, frame_length)[::hop]
        if frames.size == 0:
            rms = np.array([0.0], dtype=np.float32)
        else:
            rms = np.sqrt(np.mean(frames ** 2, axis=1))
        threshold = rms.mean() + rms.std()
        local_max = (rms[1:-1] > rms[:-2]) & (rms[1:-1] > rms[2:]) & (rms[1:-1] > threshold)
        peaks = np.where(local_max)[0]
        speaking_rate = len(peaks) / max(duration, 1e-3)

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
