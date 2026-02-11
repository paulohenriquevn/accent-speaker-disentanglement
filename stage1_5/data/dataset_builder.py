"""Dataset download + manifest building utilities."""

from __future__ import annotations

import shutil
import urllib.request
from pathlib import Path
from typing import Optional

import pandas as pd

from ..utils.io import save_jsonl, ensure_dir


def download_archive(url: str, output_dir: str | Path, filename: Optional[str] = None,
                     extract: bool = True) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if filename is None:
        filename = url.split("/")[-1].split("?")[0] or "dataset.zip"
    archive_path = output_dir / filename
    urllib.request.urlretrieve(url, archive_path)
    if extract:
        shutil.unpack_archive(archive_path, output_dir)
    return archive_path


def build_manifest_from_csv(metadata_csv: str | Path, audio_root: str | Path,
                            output_path: str | Path, source: str = "real",
                            path_column: Optional[str] = None,
                            relative_path_column: Optional[str] = "rel_path") -> Path:
    audio_root = Path(audio_root)
    output_path = Path(output_path)
    ensure_dir(output_path.parent)

    df = pd.read_csv(metadata_csv)
    required = {"utt_id", "speaker", "accent", "text_id"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Metadata missing columns: {missing}")

    rows = []
    for _, row in df.iterrows():
        if path_column and path_column in df.columns:
            audio_path = Path(row[path_column])
        elif relative_path_column and relative_path_column in df.columns:
            audio_path = audio_root / row[relative_path_column]
        else:
            audio_path = audio_root / row["speaker"] / f"{row['text_id']}.wav"
        rows.append({
            "utt_id": row["utt_id"],
            "path": str(audio_path),
            "speaker": row["speaker"],
            "accent": row["accent"],
            "text_id": row["text_id"],
            "source": source,
        })

    save_jsonl(output_path, rows)
    return output_path
