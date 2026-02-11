"""Feature persistence utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import numpy as np


def save_npz_feature(output_dir: str | Path, utt_id: str, features: Mapping[str, np.ndarray | float]) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{utt_id}.npz"
    np.savez(path, **features)
    return path
