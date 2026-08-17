"""Checkpoint format: a directory containing

    model.pt        torch state_dict of the LatentModel
    config.yaml     model config (encoder/decoder/dynamics/latent_dim/obs_dim)
    metadata.json   free-form metadata (metrics, seed, git commit, norm stats, ...)

``load_checkpoint`` rebuilds the model from ``config.yaml`` via the registries.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from nssc.models.builder import build_latent_model
from nssc.models.latent_model import LatentModel
from nssc.utils.config import load_yaml, save_yaml
from nssc.utils.io import load_json, save_json


def save_checkpoint(model: LatentModel, path: str | Path, metadata: dict[str, Any] | None = None
                    ) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path / "model.pt")
    save_yaml(model.config, path / "config.yaml")
    save_json(metadata or {}, path / "metadata.json")
    return path


def load_checkpoint(path: str | Path, map_location: str | torch.device = "cpu"
                    ) -> tuple[LatentModel, dict[str, Any]]:
    path = Path(path)
    cfg = load_yaml(path / "config.yaml")
    model = build_latent_model(cfg, obs_dim=int(cfg["obs_dim"]))
    state = torch.load(path / "model.pt", map_location=map_location, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    meta = load_json(path / "metadata.json") if (path / "metadata.json").exists() else {}
    return model, meta
