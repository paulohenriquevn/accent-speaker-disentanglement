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
def ssl(manifest: Path, output: Path, model: str = "wavlm_large") -> None:
    extract_ssl_cli(manifest, output, model)


@app.command()
def backbone(manifest: Path, text_json: Path, output: Path, checkpoint: str,
             layers: List[str] = typer.Option(..., help="Layer names to capture")) -> None:
    extract_backbone_cli(manifest, text_json, output, checkpoint, layers)


if __name__ == "__main__":
    app()
