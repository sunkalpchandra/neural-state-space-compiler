"""JSON/JSONL helpers with numpy/torch-safe encoding."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


class NumpyEncoder(json.JSONEncoder):
    def default(self, o: Any) -> Any:  # noqa: D401
        if isinstance(o, (np.floating, np.integer, np.bool_)):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, Path):
            return str(o)
        try:
            import torch

            if isinstance(o, torch.Tensor):
                return o.detach().cpu().tolist()
        except Exception:
            pass
        return super().default(o)


def save_json(obj: Any, path: str | Path, indent: int = 2) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=indent, cls=NumpyEncoder)


def load_json(path: str | Path) -> Any:
    with open(path) as f:
        return json.load(f)


def append_jsonl(obj: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(obj, cls=NumpyEncoder) + "\n")


def read_jsonl(path: str | Path) -> list[Any]:
    path = Path(path)
    if not path.exists():
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]
