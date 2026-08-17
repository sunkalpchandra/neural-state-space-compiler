"""Pointwise MLP encoder/decoder."""

from __future__ import annotations

from collections.abc import Sequence

from torch import Tensor, nn

from nssc.representations.base import Decoder, Encoder
from nssc.utils.registry import DECODERS, ENCODERS

_ACTS: dict[str, type[nn.Module]] = {
    "gelu": nn.GELU,
    "tanh": nn.Tanh,
    "relu": nn.ReLU,
    "silu": nn.SiLU,
    "elu": nn.ELU,
}


def get_activation(name: str) -> nn.Module:
    if name not in _ACTS:
        raise ValueError(f"unknown activation '{name}'; choose from {sorted(_ACTS)}")
    return _ACTS[name]()


def build_mlp(in_dim: int, out_dim: int, hidden_dims: Sequence[int] = (128, 128),
              activation: str = "gelu", layernorm: bool = False,
              dropout: float = 0.0) -> nn.Sequential:
    """``in_dim -> hidden... -> out_dim`` MLP applied to the last axis."""
    layers: list[nn.Module] = []
    prev = in_dim
    for h in hidden_dims:
        layers.append(nn.Linear(prev, h))
        if layernorm:
            layers.append(nn.LayerNorm(h))
        layers.append(get_activation(activation))
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        prev = h
    layers.append(nn.Linear(prev, out_dim))
    return nn.Sequential(*layers)


@ENCODERS.register("mlp")
class MLPEncoder(Encoder):
    is_causal = True
    is_pointwise = True
    requires_fit = False

    def __init__(self, obs_dim: int, latent_dim: int, hidden_dims: Sequence[int] = (128, 128),
                 activation: str = "gelu", layernorm: bool = False, dropout: float = 0.0,
                 **_: object) -> None:
        super().__init__(obs_dim, latent_dim)
        self.net = build_mlp(obs_dim, latent_dim, tuple(hidden_dims), activation, layernorm,
                             dropout)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


@DECODERS.register("mlp")
class MLPDecoder(Decoder):
    def __init__(self, latent_dim: int, obs_dim: int, hidden_dims: Sequence[int] = (128, 128),
                 activation: str = "gelu", layernorm: bool = False, dropout: float = 0.0,
                 **_: object) -> None:
        super().__init__(latent_dim, obs_dim)
        self.net = build_mlp(latent_dim, obs_dim, tuple(hidden_dims), activation, layernorm,
                             dropout)

    def forward(self, z: Tensor) -> Tensor:
        return self.net(z)
