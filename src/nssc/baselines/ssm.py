"""Diagonal linear state-space (S4D-lite) forecaster.

Stacks :class:`nssc.representations.ssm.SSMBlock` (the same building block as
the ``ssm`` encoder) behind an input projection and in front of a linear
next-step head. Exactly causal; the recurrence is a chunked parallel scan.
"""

from __future__ import annotations

from torch import Tensor, nn

from nssc.baselines.base import SequenceForecaster
from nssc.representations.ssm import SSMBlock
from nssc.utils.registry import BASELINES


@BASELINES.register("ssm")
class SSMForecaster(SequenceForecaster):
    def __init__(self, obs_dim: int, d_model: int = 64, d_state: int = 16, n_layers: int = 2,
                 expand: int = 2, dropout: float = 0.0, chunk: int = 32, **kw: object) -> None:
        super().__init__(obs_dim, **kw)
        self.d_model, self.d_state, self.n_layers = int(d_model), int(d_state), int(n_layers)
        self.expand, self.dropout, self.chunk = int(expand), float(dropout), int(chunk)
        self.inp = nn.Linear(obs_dim, self.d_model)
        self.blocks = nn.ModuleList(
            SSMBlock(self.d_model, self.d_state, self.expand, self.dropout, self.chunk)
            for _ in range(self.n_layers))
        self.norm = nn.LayerNorm(self.d_model)
        self._build_heads()

    @property
    def feature_dim(self) -> int:
        return self.d_model

    def backbone(self, x: Tensor) -> Tensor:
        h = self.inp(x)
        for blk in self.blocks:
            h = blk(h)
        return self.norm(h)

    def config(self) -> dict:
        return {**super().config(), "d_model": self.d_model, "d_state": self.d_state,
                "n_layers": self.n_layers, "expand": self.expand, "dropout": self.dropout,
                "chunk": self.chunk}
