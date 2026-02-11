from __future__ import annotations

from pathlib import Path

import pandas as pd

from stage1_5.data.dataset_builder import build_manifest_from_csv


def test_build_manifest_from_csv(tmp_path: Path) -> None:
    audio_root = tmp_path / "wav"
    (audio_root / "spk01").mkdir(parents=True)
    dummy_wav = audio_root / "spk01" / "t01.wav"
    dummy_wav.write_bytes(b"00")

    metadata = pd.DataFrame([
        {"utt_id": "spk01_NE_t01", "speaker": "spk01", "accent": "NE", "text_id": "t01"}
    ])
    metadata_path = tmp_path / "metadata.csv"
    metadata.to_csv(metadata_path, index=False)

    output = tmp_path / "manifest.jsonl"
    build_manifest_from_csv(metadata_path, audio_root, output)

    content = output.read_text().strip()
    assert "spk01_NE_t01" in content
    assert str(dummy_wav) in content
