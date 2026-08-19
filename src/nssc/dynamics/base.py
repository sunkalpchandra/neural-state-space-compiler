"""Base class for latent dynamics.

All dynamics expose a discrete-time ``step`` (one transition), a batched
``rollout`` (recursive application), and a ``jacobian`` for stability analysis.
Continuous-time models (neural ODE) still expose ``step`` with a fixed ``dt``.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class Dynamics(nn.Module):
    is_linear: bool = False
    is_stochastic: bool = False

    def __init__(self, latent_dim: int, control_dim: int = 0) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.control_dim = control_dim

    # ------------------------------------------------------------------ core
    def step(self, z: Tensor, u: Tensor | None = None) -> Tensor:
        """One transition. ``z``: (B, d) → (B, d)."""
        raise NotImplementedError

    def forward(self, z: Tensor, u: Tensor | None = None) -> Tensor:
        return self.step(z, u)

    def rollout(self, z0: Tensor, horizon: int, u: Tensor | None = None) -> Tensor:
        """Recursive rollout. ``z0``: (B, d) → (B, horizon, d) of z_1..z_H."""
        zs = []
        z = z0
        for k in range(horizon):
            uk = None if u is None else u[:, k]
            z = self.step(z, uk)
            zs.append(z)
        return torch.stack(zs, dim=1)

    def step_sequence(self, z: Tensor, u: Tensor | None = None) -> Tensor:
        """Teacher-forced one-step predictions for a whole sequence.

        ``z``: (B, T, d) → (B, T, d) where output[:, t] = F(z[:, t]).
        """
        B, T, d = z.shape
        flat = self.step(z.reshape(B * T, d), None if u is None else u.reshape(B * T, -1))
        return flat.reshape(B, T, d)

    # ------------------------------------------------------------- analysis
    def jacobian(self, z: Tensor, u: Tensor | None = None) -> Tensor:
        """Batched Jacobian ∂F/∂z at ``z``: (B, d) → (B, d, d). Autograd default."""
        z = z.detach().requires_grad_(True)

        def f(zi: Tensor) -> Tensor:
            return self.step(zi.unsqueeze(0), None if u is None else u[:1]).squeeze(0)

        jac = torch.func.vmap(torch.func.jacrev(f))(z)
        return jac

    def num_parameters(self) -> int:
        """Trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def num_stored(self) -> int:
        """Everything the checkpoint must store: trainable parameters + buffers.

        A PCA encoder has zero trainable parameters but a ``D×d`` component matrix in a buffer;
        counting only ``num_parameters`` made it look free to the compiler's complexity term.
        """
        return (sum(p.numel() for p in self.parameters())
                + sum(b.numel() for b in self.buffers()))

    def extra_losses(self) -> dict[str, Tensor]:
        """Optional model-specific regularisers (e.g. Koopman consistency)."""
        return {}
