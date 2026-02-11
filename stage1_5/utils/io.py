"""IO helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Iterator, Mapping, Any


def load_jsonl(path: str | Path) -> Iterator[Mapping[str, Any]]:
    with Path(path).open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def save_jsonl(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p
