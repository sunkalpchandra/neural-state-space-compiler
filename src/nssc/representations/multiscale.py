"""Multi-scale (slow / fast) causal encoder.

The latent is split as ``z = [z_slow, z_fast]``. ``z_fast`` comes from a base
causal encoder on the raw input. ``z_slow`` comes from a separate small causal
encoder that sees only a *causally smoothed and strided* version of the input:
a left-padded moving average of window ``slow_window``, sampled every
``slow_window`` steps (at block starts, so nothing from the future leaks in) and
zero-order-hold upsampled back to length ``T``. ``z_slow`` is therefore
piecewise-constant over blocks and varies slowly by construction.

Both branches are causal, so the whole encoder is causal.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor

from nssc.representations.base import Encoder
from nssc.utils.registry import ENCODERS


def causal_moving_average(x: Tensor, window: int) -> Tensor:
    """Left-padded moving average over time. ``x``: (B,T,D) -> (B,T,D)."""
    if window <= 1:
        return x
    xt = x.transpose(1, 2)  # (B,D,T)
    xt = F.pad(xt, (window - 1, 0), mode="replicate")
    return F.avg_pool1d(xt, kernel_size=window, stride=1).transpose(1, 2)


def stride_and_hold(x: Tensor, stride: int) -> tuple[Tensor, int]:
    """Sample every ``stride``-th step (starting at t=0). Returns (x_strided, T)."""
    return x[:, ::stride], x.shape[1]


def hold_upsample(x: Tensor, stride: int, length: int) -> Tensor:
    """Zero-order-hold upsample (B,T',D) -> (B,length,D)."""
    return x.repeat_interleave(stride, dim=1)[:, :length]


@ENCODERS.register("multiscale")
class MultiScaleEncoder(Encoder):
    is_causal = True
    is_pointwise = False
    requires_fit = False

    def __init__(self, obs_dim: int, latent_dim: int, slow_dim: int | None = None,
                 slow_window: int = 8, base: str = "tcn", base_kwargs: dict[str, Any] | None = None,
                 slow_encoder: str = "tcn", slow_kwargs: dict[str, Any] | None = None,
                 **_: object) -> None:
        super().__init__(obs_dim, latent_dim)
        if slow_dim is None:
            slow_dim = max(1, latent_dim // 2)
        if not 0 < slow_dim < latent_dim:
            raise ValueError("slow_dim must satisfy 0 < slow_dim < latent_dim")
        self.slow_dim = slow_dim
        self.fast_dim = latent_dim - slow_dim
        self.slow_window = int(slow_window)
        fast_cls = ENCODERS.get(base)
        slow_cls = ENCODERS.get(slow_encoder)
        self.fast = fast_cls(obs_dim, self.fast_dim, **(base_kwargs or {}))
        default_slow = {"channels": 32, "n_layers": 2} if slow_encoder == "tcn" else {}
        self.slow = slow_cls(obs_dim, self.slow_dim, **{**default_slow, **(slow_kwargs or {})})
        for enc, name in ((self.fast, base), (self.slow, slow_encoder)):
            if not enc.is_causal:
                raise ValueError(f"MultiScaleEncoder requires causal sub-encoders; '{name}' is not")

    def split(self, z: Tensor) -> tuple[Tensor, Tensor]:
        """``z`` (B,T,d) -> (z_slow (B,T,slow_dim), z_fast (B,T,fast_dim))."""
        return z[..., : self.slow_dim], z[..., self.slow_dim :]

    def encode_slow(self, x: Tensor) -> Tensor:
        length = x.shape[1]
        s = causal_moving_average(x, self.slow_window)
        s_strided, _ = stride_and_hold(s, self.slow_window)
        z = self.slow(s_strided)
        return hold_upsample(z, self.slow_window, length)

    def encode_fast(self, x: Tensor) -> Tensor:
        return self.fast(x)

    def forward(self, x: Tensor) -> Tensor:
        z_slow = self.encode_slow(x)
        z_fast = self.encode_fast(x)
        return torch.cat([z_slow, z_fast], dim=-1)
