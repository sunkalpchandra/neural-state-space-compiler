"""LatentModel: encoder + latent dynamics + decoder.

Provides the three evaluation modes used throughout the project:

* ``reconstruct(x)``           x̂_t = D(E(x))_t
* ``predict_teacher_forced(x)`` x̂_{t+1} = D(F(E(x)_t))   (one-step, ground truth latents)
* ``rollout(x_context, H)``     encode context, recurse F for H steps, decode  (recursive)
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn

from nssc.dynamics.base import Dynamics
from nssc.representations.base import Decoder, Encoder


class LatentModel(nn.Module):
    def __init__(self, encoder: Encoder, dynamics: Dynamics, decoder: Decoder,
                 config: dict[str, Any] | None = None) -> None:
        super().__init__()
        assert encoder.latent_dim == dynamics.latent_dim == decoder.latent_dim
        self.encoder = encoder
        self.dynamics = dynamics
        self.decoder = decoder
        self.config = dict(config or {})

    @property
    def latent_dim(self) -> int:
        return self.encoder.latent_dim

    @property
    def obs_dim(self) -> int:
        return self.encoder.obs_dim

    def encode(self, x: Tensor) -> Tensor:
        return self.encoder(x)

    def decode(self, z: Tensor) -> Tensor:
        return self.decoder(z)

    def reconstruct(self, x: Tensor) -> Tensor:
        return self.decode(self.encode(x))

    def predict_teacher_forced(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Returns (x̂_{2:T}, z_{1:T}, ẑ_{2:T}) with ẑ_{t+1} = F(z_t)."""
        z = self.encode(x)
        z_next_hat = self.dynamics.step_sequence(z[:, :-1])
        x_next_hat = self.decode(z_next_hat)
        return x_next_hat, z, z_next_hat

    def rollout(self, x_context: Tensor, horizon: int, u: Tensor | None = None
                ) -> tuple[Tensor, Tensor]:
        """Encode context, take last latent, roll forward ``horizon`` steps.

        Returns (x̂ (B,H,D), ẑ (B,H,d)) — predictions for the H steps *after* the context.
        """
        z = self.encode(x_context)
        z0 = z[:, -1]
        z_roll = self.dynamics.rollout(z0, horizon, u)
        return self.decode(z_roll), z_roll

    def rollout_from_latent(self, z0: Tensor, horizon: int, u: Tensor | None = None
                            ) -> tuple[Tensor, Tensor]:
        z_roll = self.dynamics.rollout(z0, horizon, u)
        return self.decode(z_roll), z_roll

    def num_parameters(self) -> dict[str, int]:
        """Trainable counts per component, plus ``total_stored`` = parameters + buffers.

        ``total`` is what a "parameter count" usually means (trainable); ``total_stored`` is the
        honest size of the compiled artefact and is what the compiler's complexity term uses, so a
        PCA encoder is not treated as free (review finding: its components live in buffers).
        """
        return {
            "encoder": self.encoder.num_parameters(),
            "dynamics": self.dynamics.num_parameters(),
            "decoder": self.decoder.num_parameters(),
            "total": sum(p.numel() for p in self.parameters() if p.requires_grad),
            "encoder_stored": self.encoder.num_stored(),
            "dynamics_stored": self.dynamics.num_stored(),
            "decoder_stored": self.decoder.num_stored(),
            "total_stored": (sum(p.numel() for p in self.parameters())
                             + sum(b.numel() for b in self.buffers())),
        }

    @torch.no_grad()
    def latent_trajectory(self, x: Tensor) -> Tensor:
        return self.encode(x)
