"""Causal dilated temporal-convolution forecaster (WaveNet/TCN style).

Reuses :class:`nssc.representations.tcn.TCNBlock`. Receptive field
``1 + (k-1) * sum(dilations)``; pass ``min_receptive_field`` (e.g. the context
length) to append doubling-dilation layers until the field covers it. Positions
outside the receptive field are exactly ignored, so ``max_context`` is set to
the receptive field (truncation in ``forecast`` is lossless).
"""

from __future__ import annotations

from collections.abc import Sequence

from torch import Tensor, nn

from nssc.baselines.base import SequenceForecaster
from nssc.representations.mlp import get_activation
from nssc.representations.tcn import TCNBlock
from nssc.utils.registry import BASELINES


@BASELINES.register("tcn")
class TCNForecaster(SequenceForecaster):
    def __init__(self, obs_dim: int, channels: int = 64, kernel_size: int = 3, n_layers: int = 4,
                 dilations: Sequence[int] | None = None, activation: str = "gelu",
                 dropout: float = 0.0, min_receptive_field: int | None = None, **kw: object) -> None:
        super().__init__(obs_dim, **kw)
        dil = list(dilations) if dilations is not None else [2**i for i in range(n_layers)]
        if len(dil) != n_layers:
            raise ValueError("len(dilations) must equal n_layers")
        if min_receptive_field:
            while 1 + (kernel_size - 1) * sum(dil) < min_receptive_field:
                dil.append(dil[-1] * 2 if dil else 1)
        self.channels, self.kernel_size, self.dilations = int(channels), int(kernel_size), dil
        self.activation, self.dropout = activation, float(dropout)
        self.inp = nn.Conv1d(obs_dim, self.channels, 1)
        self.blocks = nn.ModuleList(
            TCNBlock(self.channels, self.kernel_size, d, activation, self.dropout) for d in dil)
        self.act = get_activation(activation)
        self.max_context = self.receptive_field
        self._build_heads()

    @property
    def receptive_field(self) -> int:
        return 1 + (self.kernel_size - 1) * sum(self.dilations)

    @property
    def feature_dim(self) -> int:
        return self.channels

    def backbone(self, x: Tensor) -> Tensor:
        h = self.inp(x.transpose(1, 2))
        for blk in self.blocks:
            h = blk(h)
        return self.act(h).transpose(1, 2)

    def config(self) -> dict:
        return {**super().config(), "channels": self.channels, "kernel_size": self.kernel_size,
                "n_layers": len(self.dilations), "dilations": list(self.dilations),
                "activation": self.activation, "dropout": self.dropout}
