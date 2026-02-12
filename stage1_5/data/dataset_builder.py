"""Dataset download + manifest building utilities."""

from __future__ import annotations

import logging
import shutil
import urllib.request
from pathlib import Path
from typing import Optional, Sequence

import pandas as pd

from ..utils.io import save_jsonl, ensure_dir

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Brazilian state -> IBGE macro-region mapping
# ---------------------------------------------------------------------------

STATE_TO_REGION: dict[str, str] = {
    # Norte (N)
    "Acre": "N", "Amapá": "N", "Amazonas": "N", "Pará": "N",
    "Rondônia": "N", "Roraima": "N", "Tocantins": "N",
    # Nordeste (NE)
    "Alagoas": "NE", "Bahia": "NE", "Ceará": "NE", "Maranhão": "NE",
    "Paraíba": "NE", "Pernambuco": "NE", "Piauí": "NE",
    "Rio Grande do Norte": "NE", "Sergipe": "NE",
    # Centro-Oeste (CO)
    "Distrito Federal": "CO", "Goiás": "CO", "Mato Grosso": "CO",
    "Mato Grosso do Sul": "CO",
    # Sudeste (SE)
    "Espírito Santo": "SE", "Minas Gerais": "SE",
    "Rio de Janeiro": "SE", "São Paulo": "SE",
    # Sul (S)
    "Paraná": "S", "Rio Grande do Sul": "S", "Santa Catarina": "S",
}


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


def map_state_to_region(state: str) -> str | None:
    """Map a Brazilian state name to its IBGE macro-region code.

    Returns ``None`` when the state is not recognized.
    """
    return STATE_TO_REGION.get(state)


def build_manifest_from_coraa(
    output_path: str | Path,
    audio_dir: str | Path,
    *,
    hf_dataset_name: str = "nilc-nlp/CORAA-MUPE-ASR",
    hf_split: str = "train",
    regions: Sequence[str] | None = None,
    min_duration: float | None = None,
    max_duration: float | None = None,
    audio_quality: str | None = None,
    max_samples_per_speaker: int | None = None,
) -> Path:
    """Build a Stage 1.5 manifest from the CORAA-MUPE-ASR Hugging Face dataset.

    Parameters
    ----------
    output_path:
        Where to write the resulting ``manifest.jsonl``.
    audio_dir:
        Root directory where audio WAV files will be exported.
    hf_dataset_name:
        Hugging Face Hub dataset identifier.
    hf_split:
        Which split to load (default ``"train"``).
    regions:
        If provided, keep only speakers whose ``birth_state`` maps to one of
        these IBGE macro-region codes (e.g. ``["NE", "SE", "S"]``).
    min_duration:
        Minimum segment duration in seconds (inclusive).
    max_duration:
        Maximum segment duration in seconds (inclusive).
    audio_quality:
        Keep only rows matching this ``audio_quality`` value (``"high"`` or
        ``"low"``).
    max_samples_per_speaker:
        Cap the number of segments per speaker to avoid class imbalance.

    Returns
    -------
    Path
        The path where the manifest was written.
    """
    from datasets import load_dataset  # lazy import to avoid heavy startup
    import soundfile as sf
    import numpy as np

    output_path = Path(output_path)
    audio_dir = Path(audio_dir)
    ensure_dir(output_path.parent)
    ensure_dir(audio_dir)

    logger.info("Loading CORAA-MUPE-ASR from Hugging Face (%s, split=%s)...",
                hf_dataset_name, hf_split)
    ds = load_dataset(hf_dataset_name, split=hf_split)

    # Convert metadata columns to pandas for fast filtering, but exclude the
    # ``audio`` column.  When ``.to_pandas()`` serialises Audio features it
    # stores raw bytes (``{"path": ..., "bytes": ...}``) instead of a decoded
    # numpy array, so the audio data is unusable for writing WAV files.
    # We will access the decoded audio via the original HF dataset later.
    metadata_cols = [c for c in ds.column_names if c != "audio"]
    df = ds.select_columns(metadata_cols).to_pandas()  # type: ignore[union-attr]
    df["_hf_index"] = range(len(df))  # preserve original HF row indices

    # --- Filter: only interviewees (R) who have birth_state metadata ---
    df = df[df["speaker_type"] == "R"].copy()
    df = df[df["birth_state"].notna() & (df["birth_state"] != "")].copy()

    # --- Map birth_state -> macro-region ---
    df["region"] = df["birth_state"].map(map_state_to_region)
    unmapped = df["region"].isna()
    if unmapped.any():
        bad_states = df.loc[unmapped, "birth_state"].unique().tolist()
        logger.warning("Dropping %d rows with unmapped birth_state values: %s",
                        unmapped.sum(), bad_states)
        df = df[~unmapped].copy()

    # --- Optional filters ---
    if regions is not None:
        regions_set = set(regions)
        df = df[df["region"].isin(regions_set)].copy()
        logger.info("After region filter (%s): %d rows", regions_set, len(df))

    if min_duration is not None:
        df = df[df["duration"] >= min_duration].copy()
    if max_duration is not None:
        df = df[df["duration"] <= max_duration].copy()
    if min_duration is not None or max_duration is not None:
        logger.info("After duration filter [%s, %s]: %d rows",
                     min_duration, max_duration, len(df))

    if audio_quality is not None:
        df = df[df["audio_quality"] == audio_quality].copy()
        logger.info("After audio_quality='%s' filter: %d rows",
                     audio_quality, len(df))

    if max_samples_per_speaker is not None:
        df = (
            df.groupby("speaker_code", group_keys=False)
            .apply(lambda g: g.head(max_samples_per_speaker))
        )
        logger.info("After max_samples_per_speaker=%d: %d rows",
                      max_samples_per_speaker, len(df))

    if df.empty:
        raise ValueError(
            "No rows remain after filtering. Check your filter parameters "
            "(regions, duration, audio_quality)."
        )

    # --- Build manifest rows & export audio ---
    # Select only the rows that survived filtering from the original HF dataset
    # so that Audio decoding works correctly (returns ``array`` + ``sampling_rate``).
    filtered_indices = df["_hf_index"].tolist()
    ds_filtered = ds.select(filtered_indices)

    rows: list[dict[str, str]] = []
    for df_row, hf_row in zip(df.itertuples(), ds_filtered):
        speaker = str(df_row.speaker_code)
        region = str(df_row.region)
        audio_name = str(df_row.audio_name)
        start_time = str(df_row.start_time).replace(".", "_")
        text_id = f"{audio_name}_{start_time}"
        utt_id = f"{speaker}_{region}_{text_id}"

        # Audio export path: audio_dir/<speaker>/<utt_id>.wav
        speaker_dir = audio_dir / speaker
        speaker_dir.mkdir(parents=True, exist_ok=True)
        wav_path = speaker_dir / f"{utt_id}.wav"

        # Export audio: the HF dataset decodes the audio column on access,
        # giving us {"path": ..., "array": np.ndarray, "sampling_rate": int}.
        if not wav_path.exists():
            audio_data = hf_row["audio"]
            sf.write(
                str(wav_path),
                np.asarray(audio_data["array"]),
                audio_data["sampling_rate"],
            )

        rows.append({
            "utt_id": utt_id,
            "path": str(wav_path),
            "speaker": speaker,
            "accent": region,
            "text_id": text_id,
            "source": "real",
        })

    save_jsonl(output_path, rows)

    # --- Summary ---
    result_df = pd.DataFrame(rows)
    region_counts = result_df["accent"].value_counts().to_dict()
    speaker_counts = result_df["speaker"].nunique()
    logger.info(
        "CORAA-MUPE manifest written to %s: %d utterances, %d speakers, regions=%s",
        output_path, len(rows), speaker_counts, region_counts,
    )

    return output_path
