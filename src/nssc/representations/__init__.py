"""Encoders and decoders mapping observations x ∈ R^D ↔ latent z ∈ R^d.

Importing this package populates the ``ENCODERS`` / ``DECODERS`` registries.
"""

from __future__ import annotations

from typing import Any

from nssc.representations import (  # noqa: F401  (import for registration side effects)
    linear,
    mlp,
    multiscale,
    pca,
    rnn,
    ssm,
    tcn,
)
from nssc.representations.base import Decoder, Encoder
from nssc.representations.linear import LinearDecoder, LinearEncoder
from nssc.representations.mlp import MLPDecoder, MLPEncoder
from nssc.representations.multiscale import MultiScaleEncoder
from nssc.representations.pca import PCADecoder, PCAEncoder, explained_variance_curve
from nssc.representations.rnn import GRUEncoder, LSTMEncoder
from nssc.representations.ssm import SSMEncoder
from nssc.representations.tcn import TemporalConvEncoder
from nssc.utils.registry import DECODERS, ENCODERS


def build_encoder(key: str, obs_dim: int, latent_dim: int, **kwargs: Any) -> Encoder:
    return ENCODERS.build(key, obs_dim=obs_dim, latent_dim=latent_dim, **kwargs)


def build_decoder(key: str, latent_dim: int, obs_dim: int, **kwargs: Any) -> Decoder:
    return DECODERS.build(key, latent_dim=latent_dim, obs_dim=obs_dim, **kwargs)


__all__ = [
    "Decoder",
    "Encoder",
    "GRUEncoder",
    "LSTMEncoder",
    "LinearDecoder",
    "LinearEncoder",
    "MLPDecoder",
    "MLPEncoder",
    "MultiScaleEncoder",
    "PCADecoder",
    "PCAEncoder",
    "SSMEncoder",
    "TemporalConvEncoder",
    "build_decoder",
    "build_encoder",
    "explained_variance_curve",
]
