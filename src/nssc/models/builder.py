"""Build a LatentModel from a model config dict.

Config schema::

    latent_dim: 8
    encoder: {name: mlp, kwargs: {hidden_dims: [128, 128]}}
    decoder: {name: mlp, kwargs: {}}
    dynamics: {name: residual_mlp, kwargs: {hidden_dims: [128, 128]}}
"""

from __future__ import annotations

from typing import Any

from nssc.models.latent_model import LatentModel
from nssc.utils.registry import DECODERS, DYNAMICS, ENCODERS


def _ensure_registries() -> None:
    import nssc.dynamics  # noqa: F401
    import nssc.representations  # noqa: F401


def _spec(node: Any) -> tuple[str, dict[str, Any]]:
    if isinstance(node, str):
        return node, {}
    return node["name"], dict(node.get("kwargs", {}) or {})


def build_latent_model(cfg: dict[str, Any], obs_dim: int) -> LatentModel:
    _ensure_registries()
    d = int(cfg["latent_dim"])
    enc_name, enc_kw = _spec(cfg.get("encoder", "mlp"))
    dec_name, dec_kw = _spec(cfg.get("decoder", enc_name if enc_name in DECODERS else "mlp"))
    dyn_name, dyn_kw = _spec(cfg.get("dynamics", "residual_mlp"))
    encoder = ENCODERS.build(enc_name, obs_dim=obs_dim, latent_dim=d, **enc_kw)
    decoder = DECODERS.build(dec_name, latent_dim=d, obs_dim=obs_dim, **dec_kw)
    dynamics = DYNAMICS.build(dyn_name, latent_dim=d, **dyn_kw)
    if enc_name == "pca" and dec_name == "pca" and hasattr(decoder, "tie"):
        decoder.tie(encoder)
    return LatentModel(encoder, dynamics, decoder, config={**cfg, "obs_dim": obs_dim})


def model_name(cfg: dict[str, Any]) -> str:
    e, _ = _spec(cfg.get("encoder", "mlp"))
    dy, _ = _spec(cfg.get("dynamics", "residual_mlp"))
    return f"{e}+{dy}@d{cfg['latent_dim']}"
