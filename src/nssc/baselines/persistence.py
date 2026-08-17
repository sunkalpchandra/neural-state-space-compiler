"""Trivial baselines: persistence (repeat last value) and training-set mean."""

from __future__ import annotations

import torch
from torch import Tensor

from nssc.baselines.base import SequenceForecaster
from nssc.utils.registry import BASELINES


@BASELINES.register("persistence")
class PersistenceBaseline(SequenceForecaster):
    """``x̂_{t+k} = x_t`` for all ``k``. Zero parameters."""

    def __init__(self, obs_dim: int, **kw: object) -> None:
        super().__init__(obs_dim, **kw)

    def predict_next_sequence(self, x: Tensor) -> Tensor:
        return x

    def predict_next(self, x: Tensor) -> Tensor:
        return x[:, -1]

    def predict_direct(self, x_context: Tensor) -> Tensor:
        h = self.direct_horizon or 1
        return x_context[:, -1:].expand(-1, h, -1)

    def forecast(self, x_context: Tensor, horizon: int) -> Tensor:
        return x_context[:, -1:].expand(-1, horizon, -1).clone()


@BASELINES.register("mean")
class MeanBaseline(SequenceForecaster):
    """``x̂ = mean_train`` (per dimension). The mean is set by :meth:`fit`; zero trainable params."""

    def __init__(self, obs_dim: int, **kw: object) -> None:
        super().__init__(obs_dim, **kw)
        self.register_buffer("mean", torch.zeros(obs_dim))

    @torch.no_grad()
    def fit(self, x: Tensor) -> None:
        """``x``: (N,T,D) or (B,L,D) training data → per-dim mean."""
        self.mean.copy_(x.reshape(-1, self.obs_dim).float().mean(0).to(self.mean.device))

    def predict_next_sequence(self, x: Tensor) -> Tensor:
        return self.mean.expand(x.shape[0], x.shape[1], -1).clone()

    def predict_next(self, x: Tensor) -> Tensor:
        return self.mean.expand(x.shape[0], -1).clone()

    def predict_direct(self, x_context: Tensor) -> Tensor:
        return self.mean.expand(x_context.shape[0], self.direct_horizon or 1, -1).clone()

    def forecast(self, x_context: Tensor, horizon: int) -> Tensor:
        return self.mean.expand(x_context.shape[0], horizon, -1).clone()
