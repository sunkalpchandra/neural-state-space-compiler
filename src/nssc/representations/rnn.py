"""Recurrent (GRU / LSTM) causal encoders with a linear output projection."""

from __future__ import annotations

from torch import Tensor, nn

from nssc.representations.base import Encoder
from nssc.utils.registry import ENCODERS


class _RNNEncoder(Encoder):
    is_causal = True
    is_pointwise = False
    requires_fit = False
    _cell: type[nn.RNNBase]

    def __init__(self, obs_dim: int, latent_dim: int, hidden: int = 64, n_layers: int = 1,
                 dropout: float = 0.0, **_: object) -> None:
        super().__init__(obs_dim, latent_dim)
        self.rnn = self._cell(obs_dim, hidden, num_layers=n_layers, batch_first=True,
                              dropout=dropout if n_layers > 1 else 0.0)
        self.out = nn.Linear(hidden, latent_dim)

    def forward(self, x: Tensor) -> Tensor:
        h, _ = self.rnn(x)
        return self.out(h)


@ENCODERS.register("gru")
class GRUEncoder(_RNNEncoder):
    _cell = nn.GRU


@ENCODERS.register("lstm")
class LSTMEncoder(_RNNEncoder):
    _cell = nn.LSTM
