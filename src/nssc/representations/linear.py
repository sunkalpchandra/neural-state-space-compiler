"""Linear (single ``nn.Linear``) encoder/decoder — the linear autoencoder baseline."""

from __future__ import annotations

from torch import Tensor, nn

from nssc.representations.base import Decoder, Encoder
from nssc.utils.registry import DECODERS, ENCODERS


@ENCODERS.register("linear")
class LinearEncoder(Encoder):
    is_causal = True
    is_pointwise = True
    requires_fit = False

    def __init__(self, obs_dim: int, latent_dim: int, bias: bool = True, **_: object) -> None:
        super().__init__(obs_dim, latent_dim)
        self.proj = nn.Linear(obs_dim, latent_dim, bias=bias)

    def forward(self, x: Tensor) -> Tensor:
        return self.proj(x)


@DECODERS.register("linear")
class LinearDecoder(Decoder):
    def __init__(self, latent_dim: int, obs_dim: int, bias: bool = True, **_: object) -> None:
        super().__init__(latent_dim, obs_dim)
        self.proj = nn.Linear(latent_dim, obs_dim, bias=bias)

    def forward(self, z: Tensor) -> Tensor:
        return self.proj(z)
