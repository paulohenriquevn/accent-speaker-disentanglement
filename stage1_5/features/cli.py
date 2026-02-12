"""Typer CLI for feature extraction."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import typer

from .acoustic import extract_acoustic_cli
from .ecapa import extract_ecapa_cli
from .ssl import extract_ssl_cli
from .backbone import extract_backbone_cli

app = typer.Typer(add_completion=False)


@app.command()
def acoustic(
    manifest: Path,
    output: Path,
    sample_rate: int = 16000,
    max_per_speaker: Optional[int] = typer.Option(None, help="Limit utterances per speaker (stratified subsample)"),
) -> None:
    extract_acoustic_cli(manifest, output, sample_rate, max_per_speaker=max_per_speaker)


@app.command()
def ecapa(
    manifest: Path,
    output: Path,
    device: str = "cpu",
    max_per_speaker: Optional[int] = typer.Option(None, help="Limit utterances per speaker (stratified subsample)"),
) -> None:
    extract_ecapa_cli(manifest, output, device, max_per_speaker=max_per_speaker)


@app.command()
def ssl(
    manifest: Path,
    output: Path,
    model: str = "wavlm_large",
    layers: List[int] | None = typer.Option(None, help="Layer indices to extract"),
    device: str = "cpu",
    torch_dtype: str | None = typer.Option(None, help="float16|bfloat16|float32"),
    pooling: str = typer.Option("mean", help="mean|max|mean_std"),
    max_per_speaker: Optional[int] = typer.Option(None, help="Limit utterances per speaker (stratified subsample)"),
) -> None:
    extract_ssl_cli(manifest, output, model, layers, device, torch_dtype, pooling, max_per_speaker=max_per_speaker)


@app.command()
def backbone(manifest: Path, text_json: Path, output: Path, checkpoint: str,
             layers: List[str] = typer.Option(..., help="Layer names to capture"),
             device: str = "cpu",
             dtype: str | None = typer.Option(None, help="float16|bfloat16|float32"),
             attn_implementation: str | None = typer.Option(None, help="flash-attn2|flash-attn3|eager"),
             generation_mode: str = "custom_voice",
             generation_language: str = "Portuguese",
             generation_speaker: str = "ryan",
             generation_instruct: str | None = None,
             generation_max_new_tokens: int = 256,
             pooling: str = typer.Option("mean", help="mean|max|mean_std"),
             strict: bool = True,
             max_per_speaker: Optional[int] = typer.Option(None, help="Limit utterances per speaker (stratified subsample)"),
             ) -> None:
    extract_backbone_cli(
        manifest_path=manifest,
        text_json=text_json,
        output_dir=output,
        checkpoint=checkpoint,
        layers=layers,
        device=device,
        dtype=dtype,
        attn_implementation=attn_implementation,
        generation_mode=generation_mode,
        generation_language=generation_language,
        generation_speaker=generation_speaker,
        generation_instruct=generation_instruct,
        generation_max_new_tokens=generation_max_new_tokens,
        pooling=pooling,
        strict=strict,
        max_per_speaker=max_per_speaker,
    )


if __name__ == "__main__":
    app()
