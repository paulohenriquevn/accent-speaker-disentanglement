"""CLI entrypoints for Stage 1.5 pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer

from .pipelines.run_all import run_pipeline
from .features.cli import app as features_app
from .data.dataset_builder import download_archive, build_manifest_from_csv, build_manifest_from_coraa

app = typer.Typer(add_completion=False, help="Stage 1.5 latent separability toolkit")
app.add_typer(features_app, name="features", help="Feature extraction helpers")

dataset_app = typer.Typer(help="Dataset preparation helpers")


@dataset_app.command("download")
def dataset_download(url: str = typer.Argument(..., help="HTTP/HTTPS URL to dataset archive"),
                     output_dir: Path = typer.Option(Path("data/external"), help="Where to store the archive"),
                     filename: str | None = typer.Option(None, help="Optional custom filename"),
                     extract: bool = typer.Option(True, help="Automatically extract archive")) -> None:
    download_archive(url, output_dir, filename, extract)
    typer.echo(f"Dataset downloaded to {output_dir}")


@dataset_app.command("build-manifest")
def dataset_build_manifest(metadata: Path = typer.Argument(..., help="CSV with utt_id,speaker,accent,text_id"),
                           audio_root: Path = typer.Option(Path("data/wav"), help="Directory containing WAV files"),
                           output: Path = typer.Option(Path("data/manifest.jsonl"), help="Output manifest path"),
                           source: str = typer.Option("real", help="Source label for manifest"),
                           path_column: str | None = typer.Option(None, help="Column containing absolute audio paths"),
                           rel_path_column: str | None = typer.Option("rel_path", help="Column with paths relative to audio_root")) -> None:
    build_manifest_from_csv(metadata, audio_root, output, source, path_column, rel_path_column)
    typer.echo(f"Manifest written to {output}")


@dataset_app.command("build-coraa")
def dataset_build_coraa(
    output: Path = typer.Option(Path("data/manifest.jsonl"), help="Output manifest path"),
    audio_dir: Path = typer.Option(Path("data/wav/coraa"), help="Directory to export audio files"),
    hf_dataset: str = typer.Option("nilc-nlp/CORAA-MUPE-ASR", help="Hugging Face dataset identifier"),
    hf_split: str = typer.Option("train", help="Dataset split to load"),
    regions: str | None = typer.Option(None, help="Comma-separated macro-region codes (e.g. NE,SE,S)"),
    min_duration: float | None = typer.Option(None, help="Minimum segment duration in seconds"),
    max_duration: float | None = typer.Option(None, help="Maximum segment duration in seconds"),
    audio_quality: str | None = typer.Option(None, help="Filter by audio quality (high or low)"),
    max_samples_per_speaker: int | None = typer.Option(None, help="Max segments per speaker"),
) -> None:
    """Build a Stage 1.5 manifest from the CORAA-MUPE-ASR dataset (Hugging Face)."""
    region_list = [r.strip() for r in regions.split(",")] if regions else None
    build_manifest_from_coraa(
        output_path=output,
        audio_dir=audio_dir,
        hf_dataset_name=hf_dataset,
        hf_split=hf_split,
        regions=region_list,
        min_duration=min_duration,
        max_duration=max_duration,
        audio_quality=audio_quality,
        max_samples_per_speaker=max_samples_per_speaker,
    )
    typer.echo(f"CORAA-MUPE manifest written to {output}")


app.add_typer(dataset_app, name="dataset")


@app.command()
def run(config: Path = typer.Argument(Path("config/stage1_5.yaml"), exists=True),
        output: Optional[Path] = typer.Option(None, help="Optional path to dump metrics JSON")) -> None:
    """Execute the full Stage 1.5 experiment pipeline."""

    report = run_pipeline(config)
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    typer.echo("Pipeline finished. Decision: %s" % report["decision"])


if __name__ == "__main__":
    app()
