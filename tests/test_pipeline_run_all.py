from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from stage1_5.pipelines.run_all import run_pipeline


def _write_manifest(path: Path) -> None:
    rows = []
    accents = ["NE", "SE"]
    speakers = {
        "NE": [f"spk_ne_{idx}" for idx in range(1, 7)],
        "SE": [f"spk_se_{idx}" for idx in range(1, 7)],
    }
    for accent in accents:
        for speaker in speakers[accent]:
            for text_id in ["t1", "t2"]:
                rows.append(
                    {
                        "utt_id": f"{speaker}_{text_id}",
                        "path": "",
                        "speaker": speaker,
                        "accent": accent,
                        "text_id": text_id,
                        "source": "real",
                    }
                )
    path.write_text("\n".join(json.dumps(r) for r in rows))


def _write_features(features_dir: Path, manifest_path: Path) -> None:
    features_dir.mkdir(parents=True, exist_ok=True)
    for line in manifest_path.read_text().splitlines():
        row = json.loads(line)
        utt_id = row["utt_id"]
        accent = row["accent"]
        vector = np.array([0.0, 0.0], dtype=np.float32) if accent == "NE" else np.array([10.0, 10.0], dtype=np.float32)
        np.savez(features_dir / f"{utt_id}.npz", rep=vector)


def test_run_pipeline_end_to_end(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.jsonl"
    _write_manifest(manifest_path)

    features_dir = tmp_path / "features"
    _write_features(features_dir, manifest_path)

    config_path = tmp_path / "config.yaml"
    report_path = tmp_path / "report" / "stage1_5_report.md"
    metrics_path = tmp_path / "analysis" / "metrics.csv"
    heatmap_dir = tmp_path / "analysis" / "figures"

    config = {
        "experiment": {
            "seed": 1,
            "leakage_margin_pp": 7,
            "leakage_conditional_margin_pp": 12,
            "min_f1_go": 0.55,
            "min_f1_conditional": 0.45,
            "text_drop_tolerance_pp": 10,
        },
        "paths": {
            "manifest": str(manifest_path),
            "report": str(report_path),
        },
        "analysis": {
            "metrics_csv": str(metrics_path),
            "heatmap_dir": str(heatmap_dir),
        },
        "probes": {
            "max_iter": 400,
            "test_size": 0.5,
            "speaker_disjoint": False,
            "feature_spaces": [
                {
                    "name": "backbone",
                    "root": str(features_dir),
                    "keys": ["rep"],
                    "combine": False,
                    "target": "accent",
                }
            ],
        },
    }
    config_path.write_text(json.dumps(config))

    summary = run_pipeline(config_path)
    assert summary["decision"] in {"GO", "GO_CONDITIONAL"}
    assert Path(summary["report_path"]).exists()
    assert Path(summary["metrics_path"]).exists()
