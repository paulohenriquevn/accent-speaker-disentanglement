"""Typer CLI for feature extraction."""

from __future__ import annotations

from pathlib import Path
from typing import List

import typer

from .acoustic import extract_acoustic_cli
from .ecapa import extract_ecapa_cli
from .ssl import extract_ssl_cli
from .backbone import extract_backbone_cli

app = typer.Typer(add_completion=False)


@app.command()
def acoustic(manifest: Path, output: Path, sample_rate: int = 16000) -> None:
    extract_acoustic_cli(manifest, output, sample_rate)


@app.command()
def ecapa(manifest: Path, output: Path, device: str = "cpu") -> None:
    extract_ecapa_cli(manifest, output, device)


@app.command()
def ssl(
    manifest: Path,
    output: Path,
    model: str = "wavlm_large",
    layers: List[int] | None = typer.Option(None, help="Layer indices to extract"),
    device: str = "cpu",
    torch_dtype: str | None = typer.Option(None, help="float16|bfloat16|float32"),
    pooling: str = typer.Option("mean", help="mean|max|mean_std"),
) -> None:
    extract_ssl_cli(manifest, output, model, layers, device, torch_dtype, pooling)


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
             strict: bool = True) -> None:
    extract_backbone_cli(
        manifest,
        text_json,
        output,
        checkpoint,
        layers,
        device,
        dtype,
        attn_implementation,
        generation_mode,
        generation_language,
        generation_speaker,
        generation_instruct,
        generation_max_new_tokens,
        pooling,
        strict,
    )


if __name__ == "__main__":
    app()
