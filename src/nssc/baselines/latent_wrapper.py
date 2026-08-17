"""Adapt a :class:`nssc.models.LatentModel` to the :class:`SequenceForecaster` interface.

Lets :func:`nssc.baselines.evaluate.evaluate_forecaster` score compiled latent
models and sequence baselines with the identical protocol. Not registered in
``BASELINES`` (it is not a candidate to build from a config).
"""

from __future__ import annotations

from torch import Tensor

from nssc.baselines.base import SequenceForecaster
from nssc.models.latent_model import LatentModel


class LatentModelForecaster(SequenceForecaster):
    registry_key = "latent"

    def __init__(self, model: LatentModel) -> None:
        super().__init__(model.obs_dim, mode="recursive")
        self.model = model

    @property
    def latent_dim(self) -> int:
        return self.model.latent_dim

    def predict_next_sequence(self, x: Tensor) -> Tensor:
        z = self.model.encode(x)
        return self.model.decode(self.model.dynamics.step_sequence(z))

    def predict_next(self, x: Tensor) -> Tensor:
        z = self.model.encode(x)[:, -1]
        return self.model.decode(self.model.dynamics.step(z))

    def forecast(self, x_context: Tensor, horizon: int) -> Tensor:
        x_hat, _ = self.model.rollout(x_context, horizon)
        return x_hat

    def num_parameters(self) -> int:
        return int(self.model.num_parameters()["total"])
