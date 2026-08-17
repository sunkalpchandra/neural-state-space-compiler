"""Causal dilated temporal convolutional encoder (TCN).

Each residual block: LeftPad -> Conv1d(dilation) -> act -> Conv1d(k=1) -> +skip.
Left padding only, so ``z_t`` depends on ``x_{≤t}``. Receptive field is
``1 + (k-1) * sum(dilations)``.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch.nn.functional as F
from torch import Tensor, nn

from nssc.representations.base import Encoder
from nssc.representations.mlp import get_activation
from nssc.utils.registry import ENCODERS


class CausalConv1d(nn.Module):
    """1D convolution over (B,C,T) with left-only padding (causal)."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, dilation: int = 1) -> None:
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, dilation=dilation)

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(F.pad(x, (self.pad, 0)))


class TCNBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dilation: int, activation: str,
                 dropout: float) -> None:
        super().__init__()
        self.conv = CausalConv1d(channels, channels, kernel_size, dilation)
        self.act = get_activation(activation)
        self.mix = nn.Conv1d(channels, channels, 1)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:  # (B,C,T)
        h = self.drop(self.mix(self.act(self.conv(x))))
        return x + h


@ENCODERS.register("tcn")
class TemporalConvEncoder(Encoder):
    is_causal = True
    is_pointwise = False
    requires_fit = False

    def __init__(self, obs_dim: int, latent_dim: int, channels: int = 64, kernel_size: int = 3,
                 n_layers: int = 4, dilations: Sequence[int] | None = None,
                 activation: str = "gelu", dropout: float = 0.0, **_: object) -> None:
        super().__init__(obs_dim, latent_dim)
        if dilations is None:
            dilations = [2**i for i in range(n_layers)]
        dilations = list(dilations)
        if len(dilations) != n_layers:
            raise ValueError("len(dilations) must equal n_layers")
        self.kernel_size = kernel_size
        self.dilations = dilations
        self.inp = nn.Conv1d(obs_dim, channels, 1)
        self.blocks = nn.ModuleList(
            TCNBlock(channels, kernel_size, d, activation, dropout) for d in dilations
        )
        self.act = get_activation(activation)
        self.out = nn.Conv1d(channels, latent_dim, 1)

    @property
    def receptive_field(self) -> int:
        return 1 + (self.kernel_size - 1) * sum(self.dilations)

    def forward(self, x: Tensor) -> Tensor:
        h = self.inp(x.transpose(1, 2))  # (B,C,T)
        for blk in self.blocks:
            h = blk(h)
        return self.out(self.act(h)).transpose(1, 2)
