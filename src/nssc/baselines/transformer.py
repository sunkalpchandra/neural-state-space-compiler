"""Decoder-only (causal) transformer forecaster.

``nn.TransformerEncoder`` with a causal attention mask, learned positional
embeddings up to ``max_len`` (default 512), pre-LayerNorm blocks and a linear
next-step head applied at every position. Contexts longer than ``max_len`` are
truncated to the most recent ``max_len`` steps.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from nssc.baselines.base import SequenceForecaster
from nssc.utils.registry import BASELINES


@BASELINES.register("transformer")
class TransformerForecaster(SequenceForecaster):
    def __init__(self, obs_dim: int, d_model: int = 64, n_heads: int = 4, n_layers: int = 2,
                 dim_feedforward: int | None = None, dropout: float = 0.0, max_len: int = 512,
                 **kw: object) -> None:
        super().__init__(obs_dim, **kw)
        if d_model % n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        self.d_model, self.n_heads, self.n_layers = int(d_model), int(n_heads), int(n_layers)
        self.dim_feedforward = int(dim_feedforward or 4 * d_model)
        self.dropout, self.max_len = float(dropout), int(max_len)
        self.max_context = self.max_len
        self.inp = nn.Linear(obs_dim, self.d_model)
        self.pos = nn.Embedding(self.max_len, self.d_model)
        layer = nn.TransformerEncoderLayer(self.d_model, self.n_heads, self.dim_feedforward,
                                           dropout=self.dropout, activation="gelu",
                                           batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, self.n_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(self.d_model)
        self._build_heads()

    @property
    def feature_dim(self) -> int:
        return self.d_model

    def backbone(self, x: Tensor) -> Tensor:
        b, t, _ = x.shape
        if t > self.max_len:
            raise ValueError(f"sequence length {t} exceeds max_len {self.max_len}")
        pos = torch.arange(t, device=x.device)
        h = self.inp(x) + self.pos(pos)[None]
        mask = nn.Transformer.generate_square_subsequent_mask(t, device=x.device)
        h = self.encoder(h, mask=mask, is_causal=True)
        return self.norm(h)

    def config(self) -> dict:
        return {**super().config(), "d_model": self.d_model, "n_heads": self.n_heads,
                "n_layers": self.n_layers, "dim_feedforward": self.dim_feedforward,
                "dropout": self.dropout, "max_len": self.max_len}
