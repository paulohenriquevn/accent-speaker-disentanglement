from __future__ import annotations

import json
from pathlib import Path

from stage1_5.data import Manifest


def test_manifest_loading(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.jsonl"
    rows = [
        {"utt_id": "u1", "path": "a.wav", "speaker": "s1", "accent": "NE", "text_id": "t1", "source": "real"},
        {"utt_id": "u2", "path": "b.wav", "speaker": "s2", "accent": "SE", "text_id": "t1", "source": "real"},
    ]
    manifest_path.write_text("\n".join(json.dumps(r) for r in rows))

    manifest = Manifest.from_jsonl(manifest_path)
    assert len(manifest) == 2
    df = manifest.to_dataframe()
    assert set(df["accent"]) == {"NE", "SE"}
