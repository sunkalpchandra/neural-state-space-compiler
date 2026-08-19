"""Base classes for encoders and decoders.

Shape convention: observations ``x`` are ``(batch, time, obs_dim)``; latents
``z`` are ``(batch, time, latent_dim)``. Encoders may be causal (depend on
``x_{≤t}``) or pointwise; ``is_causal`` documents which. Non-neural encoders
(PCA) still subclass ``nn.Module`` so they serialise through ``state_dict``.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class Encoder(nn.Module):
    is_causal: bool = True
    is_pointwise: bool = True  # z_t depends only on x_t
    requires_fit: bool = False  # e.g. PCA: closed-form fit before/instead of SGD

    def __init__(self, obs_dim: int, latent_dim: int) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.latent_dim = latent_dim

    def forward(self, x: Tensor) -> Tensor:  # (B,T,D) -> (B,T,d)
        raise NotImplementedError

    def encode(self, x: Tensor) -> Tensor:
        return self.forward(x)

    @torch.no_grad()
    def fit(self, x: Tensor) -> None:  # closed-form fit hook (PCA); default no-op
        return None

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


class Decoder(nn.Module):
    def __init__(self, latent_dim: int, obs_dim: int) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.obs_dim = obs_dim

    def forward(self, z: Tensor) -> Tensor:  # (B,T,d) -> (B,T,D)
        raise NotImplementedError

    def decode(self, z: Tensor) -> Tensor:
        return self.forward(z)

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
