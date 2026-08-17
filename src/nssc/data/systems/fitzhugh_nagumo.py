"""FitzHugh-Nagumo excitable / slow-fast neuron model, ``d = 2``."""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np

from nssc.data.systems.base import DynamicalSystem
from nssc.utils.registry import SYSTEMS


@SYSTEMS.register("fitzhugh_nagumo")
class FitzHughNagumo(DynamicalSystem):
    """``v' = v - v^3/3 - w + I, w' = eps (v + a - b w)``; ``I=0.5`` oscillatory."""

    name: ClassVar[str] = "fitzhugh_nagumo"
    state_dim = 2
    default_params: ClassVar[dict[str, Any]] = {"a": 0.7, "b": 0.8, "eps": 0.08, "I": 0.5}
    default_dt: ClassVar[float] = 0.1
    default_transient: ClassVar[int] = 0

    def f(self, t: float, x: np.ndarray, params: dict[str, Any]) -> np.ndarray:
        a, b, e, cur = params["a"], params["b"], params["eps"], params["I"]
        v, w = x[:, 0], x[:, 1]
        return np.stack([v - v**3 / 3.0 - w + cur, e * (v + a - b * w)], axis=1)

    def sample_initial(self, rng: np.random.Generator, n: int) -> np.ndarray:
        v = rng.uniform(-2.0, 2.0, size=n)
        w = rng.uniform(-1.0, 1.5, size=n)
        return np.stack([v, w], axis=1)
