"""GRU / LSTM autoregressive forecasters.

``predict_next`` runs the RNN over the full context and projects the last hidden
state. ``forecast`` (recursive mode) carries the hidden state across steps
instead of re-reading the growing context, which is mathematically identical.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from nssc.baselines.base import SequenceForecaster
from nssc.utils.registry import BASELINES


class _RNNForecaster(SequenceForecaster):
    _cell: type[nn.RNNBase]

    def __init__(self, obs_dim: int, hidden: int = 64, n_layers: int = 1, dropout: float = 0.0,
                 **kw: object) -> None:
        super().__init__(obs_dim, **kw)
        self.hidden, self.n_layers, self.dropout = int(hidden), int(n_layers), float(dropout)
        self.rnn = self._cell(obs_dim, self.hidden, num_layers=self.n_layers, batch_first=True,
                              dropout=self.dropout if self.n_layers > 1 else 0.0)
        self._build_heads()

    @property
    def feature_dim(self) -> int:
        return self.hidden

    def backbone(self, x: Tensor) -> Tensor:
        h, _ = self.rnn(x)
        return h

    def forecast(self, x_context: Tensor, horizon: int) -> Tensor:
        if self.mode == "direct":
            return super().forecast(x_context, horizon)
        h, state = self.rnn(x_context)
        x_next = self.head(h[:, -1])
        outs = [x_next]
        for _ in range(horizon - 1):
            h, state = self.rnn(x_next.unsqueeze(1), state)
            x_next = self.head(h[:, -1])
            outs.append(x_next)
        return torch.stack(outs, dim=1)

    def config(self) -> dict:
        return {**super().config(), "hidden": self.hidden, "n_layers": self.n_layers,
                "dropout": self.dropout}


@BASELINES.register("gru")
class GRUForecaster(_RNNForecaster):
    _cell = nn.GRU


@BASELINES.register("lstm")
class LSTMForecaster(_RNNForecaster):
    _cell = nn.LSTM
