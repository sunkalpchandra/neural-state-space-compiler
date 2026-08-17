"""Latent transition operators z_{t+1} = F_θ(z_t, u_t).

Import this package to populate the DYNAMICS registry.
"""

from __future__ import annotations

from typing import Any

from nssc.dynamics.base import Dynamics
from nssc.dynamics.gaussian import GaussianDynamics
from nssc.dynamics.koopman import KoopmanDynamics
from nssc.dynamics.linear import AffineDynamics, LinearDynamics
from nssc.dynamics.mlp import MLPDynamics, ResidualMLPDynamics
from nssc.dynamics.multiscale import MultiScaleDynamics
from nssc.dynamics.neural_ode import NeuralODEDynamics
from nssc.dynamics.ssm import SSMDynamics
from nssc.utils.registry import DYNAMICS


def build_dynamics(key: str, latent_dim: int, **kwargs: Any) -> Dynamics:
    """Instantiate a registered dynamics family by key."""
    return DYNAMICS.build(key, latent_dim=latent_dim, **kwargs)


__all__ = [
    "DYNAMICS",
    "Dynamics",
    "AffineDynamics",
    "GaussianDynamics",
    "KoopmanDynamics",
    "LinearDynamics",
    "MLPDynamics",
    "MultiScaleDynamics",
    "NeuralODEDynamics",
    "ResidualMLPDynamics",
    "SSMDynamics",
    "build_dynamics",
]
