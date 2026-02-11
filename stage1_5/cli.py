"""CLI entrypoints for Stage 1.5 pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer

from .pipelines.run_all import run_pipeline
from .features.cli import app as features_app

app = typer.Typer(add_completion=False, help="Stage 1.5 latent separability toolkit")
app.add_typer(features_app, name="features", help="Feature extraction helpers")


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
