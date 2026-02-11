from __future__ import annotations

import numpy as np
from pathlib import Path

from stage1_5.data import Manifest, ManifestEntry
from stage1_5.probes.dataset import FeaturePlan
from stage1_5.probes.trainer import ProbeRunner, ProbeConfig


def test_probe_runner_outputs_metrics(tmp_path: Path) -> None:
    entries: list[ManifestEntry] = []
    feats_dir = tmp_path / "feats"
    feats_dir.mkdir(parents=True, exist_ok=True)
    accents = ["NE", "SE", "S"]
    for accent_idx, accent in enumerate(accents):
        for speaker_idx in range(3):
            speaker = f"{accent}_spk{speaker_idx}"
            for text_idx in range(5):
                utt_id = f"{speaker}_t{text_idx}"
                entries.append(ManifestEntry(utt_id, "", speaker, accent, f"t{text_idx}", "real"))
                vector = np.array([accent_idx, accent_idx * 10 + speaker_idx], dtype=np.float32)
                np.savez(feats_dir / f"{utt_id}.npz", rep=vector)

    manifest = Manifest(entries)
    plan = FeaturePlan(name="dummy", root=str(feats_dir), keys=["rep"], combine=True)
    runner = ProbeRunner(manifest, [plan], ProbeConfig(max_iter=300, test_size=0.3, seed=1))
    results = runner.run()
    assert len(results) == 1
    res = results[0]
    assert res.accent_f1 > 0.8
    assert res.speaker_acc > 0.4
