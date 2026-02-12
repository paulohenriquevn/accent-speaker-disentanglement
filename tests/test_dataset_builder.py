from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd

from stage1_5.data.dataset_builder import (
    build_manifest_from_csv,
    build_manifest_from_coraa,
    map_state_to_region,
    STATE_TO_REGION,
)


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


# ---------------------------------------------------------------------------
# STATE_TO_REGION mapping tests
# ---------------------------------------------------------------------------

class TestStateToRegion:
    def test_all_27_states_are_mapped(self) -> None:
        assert len(STATE_TO_REGION) == 27

    def test_known_mappings(self) -> None:
        assert map_state_to_region("São Paulo") == "SE"
        assert map_state_to_region("Bahia") == "NE"
        assert map_state_to_region("Paraná") == "S"
        assert map_state_to_region("Amazonas") == "N"
        assert map_state_to_region("Goiás") == "CO"

    def test_unknown_state_returns_none(self) -> None:
        assert map_state_to_region("Narnia") is None
        assert map_state_to_region("") is None

    def test_valid_region_codes(self) -> None:
        valid_codes = {"N", "NE", "CO", "SE", "S"}
        assert set(STATE_TO_REGION.values()) == valid_codes


# ---------------------------------------------------------------------------
# build_manifest_from_coraa tests
# ---------------------------------------------------------------------------

def _make_fake_hf_dataframe(n: int = 10, **overrides: object) -> pd.DataFrame:
    """Build a DataFrame that mimics CORAA-MUPE-ASR metadata columns."""
    rows = []
    states = ["São Paulo", "Bahia", "Paraná", "Amazonas", "Goiás"]
    for i in range(n):
        state = states[i % len(states)]
        row = {
            "audio_id": i,
            "audio_name": f"interview_{i:03d}",
            "file_path": f"audio/interview_{i:03d}.wav",
            "speaker_type": "R",
            "speaker_code": f"SPK{i // 2:03d}",
            "speaker_gender": "F" if i % 2 == 0 else "M",
            "education": "higher",
            "birth_state": state,
            "birth_country": "Brazil",
            "age": 30 + i,
            "recording_year": 2022,
            "audio_quality": "high" if i % 3 != 0 else "low",
            "start_time": float(i * 10),
            "end_time": float(i * 10 + 5),
            "duration": 5.0,
            "normalized_text": f"texto normalizado {i}",
            "original_text": f"Texto Original {i}.",
            "audio": {
                "path": None,
                "array": np.zeros(16000, dtype=np.float32),
                "sampling_rate": 16000,
            },
        }
        row.update(overrides)
        rows.append(row)
    return pd.DataFrame(rows)


def _mock_load_dataset(df: pd.DataFrame) -> MagicMock:
    """Create a mock that behaves like datasets.load_dataset()."""
    mock_ds = MagicMock()
    mock_ds.to_pandas.return_value = df
    return mock_ds


_PATCH_TARGET = "datasets.load_dataset"


class TestBuildManifestFromCoraa:
    def test_basic_manifest_generation(self, tmp_path: Path) -> None:
        df = _make_fake_hf_dataframe(6)
        output = tmp_path / "manifest.jsonl"
        audio_dir = tmp_path / "wav"

        with patch(_PATCH_TARGET, return_value=_mock_load_dataset(df)):
            result = build_manifest_from_coraa(output, audio_dir)

        assert result == output
        assert output.exists()
        lines = output.read_text().strip().split("\n")
        assert len(lines) == 6

        entry = json.loads(lines[0])
        assert "utt_id" in entry
        assert "path" in entry
        assert "speaker" in entry
        assert "accent" in entry
        assert "text_id" in entry
        assert entry["source"] == "real"
        # accent should be a macro-region code
        assert entry["accent"] in {"N", "NE", "CO", "SE", "S"}

    def test_filters_only_interviewees(self, tmp_path: Path) -> None:
        df = _make_fake_hf_dataframe(4)
        # Mark half as interviewers (no birth_state usable)
        df.loc[0, "speaker_type"] = "P/1"
        df.loc[1, "speaker_type"] = "P/2"
        output = tmp_path / "manifest.jsonl"

        with patch(_PATCH_TARGET, return_value=_mock_load_dataset(df)):
            build_manifest_from_coraa(output, tmp_path / "wav")

        lines = output.read_text().strip().split("\n")
        assert len(lines) == 2  # only the R rows

    def test_region_filter(self, tmp_path: Path) -> None:
        df = _make_fake_hf_dataframe(10)
        output = tmp_path / "manifest.jsonl"

        with patch(_PATCH_TARGET, return_value=_mock_load_dataset(df)):
            build_manifest_from_coraa(output, tmp_path / "wav", regions=["SE", "NE"])

        lines = output.read_text().strip().split("\n")
        for line in lines:
            entry = json.loads(line)
            assert entry["accent"] in {"SE", "NE"}

    def test_duration_filter(self, tmp_path: Path) -> None:
        df = _make_fake_hf_dataframe(4)
        df.loc[0, "duration"] = 0.5  # too short
        df.loc[1, "duration"] = 60.0  # too long
        output = tmp_path / "manifest.jsonl"

        with patch(_PATCH_TARGET, return_value=_mock_load_dataset(df)):
            build_manifest_from_coraa(output, tmp_path / "wav",
                                       min_duration=1.0, max_duration=30.0)

        lines = output.read_text().strip().split("\n")
        assert len(lines) == 2  # rows 2 and 3 pass the filter

    def test_audio_quality_filter(self, tmp_path: Path) -> None:
        df = _make_fake_hf_dataframe(6)
        output = tmp_path / "manifest.jsonl"

        with patch(_PATCH_TARGET, return_value=_mock_load_dataset(df)):
            build_manifest_from_coraa(output, tmp_path / "wav", audio_quality="high")

        lines = output.read_text().strip().split("\n")
        # rows where i % 3 != 0 have quality "high" -> indices 1,2,4,5 -> 4 rows
        assert len(lines) == 4

    def test_max_samples_per_speaker(self, tmp_path: Path) -> None:
        df = _make_fake_hf_dataframe(10)
        # All same speaker to test capping
        df["speaker_code"] = "SPK000"
        output = tmp_path / "manifest.jsonl"

        with patch(_PATCH_TARGET, return_value=_mock_load_dataset(df)):
            build_manifest_from_coraa(output, tmp_path / "wav",
                                       max_samples_per_speaker=3)

        lines = output.read_text().strip().split("\n")
        assert len(lines) == 3

    def test_unmapped_state_is_dropped(self, tmp_path: Path) -> None:
        df = _make_fake_hf_dataframe(2)
        df.loc[0, "birth_state"] = "Unknown State"
        output = tmp_path / "manifest.jsonl"

        with patch(_PATCH_TARGET, return_value=_mock_load_dataset(df)):
            build_manifest_from_coraa(output, tmp_path / "wav")

        lines = output.read_text().strip().split("\n")
        assert len(lines) == 1  # only the valid row

    def test_empty_result_raises_error(self, tmp_path: Path) -> None:
        df = _make_fake_hf_dataframe(2)
        df["speaker_type"] = "P/1"  # no interviewees
        output = tmp_path / "manifest.jsonl"

        import pytest
        with pytest.raises(ValueError, match="No rows remain"):
            with patch(_PATCH_TARGET, return_value=_mock_load_dataset(df)):
                build_manifest_from_coraa(output, tmp_path / "wav")

    def test_text_id_is_unique_per_segment(self, tmp_path: Path) -> None:
        df = _make_fake_hf_dataframe(6)
        output = tmp_path / "manifest.jsonl"

        with patch(_PATCH_TARGET, return_value=_mock_load_dataset(df)):
            build_manifest_from_coraa(output, tmp_path / "wav")

        lines = output.read_text().strip().split("\n")
        text_ids = [json.loads(line)["text_id"] for line in lines]
        assert len(text_ids) == len(set(text_ids)), "text_ids must be unique per segment"

    def test_audio_export_writes_wav(self, tmp_path: Path) -> None:
        """When HF audio has array data and no path on disk, WAV should be exported."""
        df = _make_fake_hf_dataframe(1)
        # Ensure audio path is None so it falls to array export
        df.at[0, "audio"] = {
            "path": None,
            "array": np.zeros(16000, dtype=np.float32),
            "sampling_rate": 16000,
        }
        output = tmp_path / "manifest.jsonl"
        audio_dir = tmp_path / "wav"

        with patch(_PATCH_TARGET, return_value=_mock_load_dataset(df)):
            build_manifest_from_coraa(output, audio_dir)

        lines = output.read_text().strip().split("\n")
        entry = json.loads(lines[0])
        wav_path = Path(entry["path"])
        assert wav_path.exists()
        assert wav_path.suffix == ".wav"
