"""PCA encoder/decoder (closed-form linear subspace baseline).

``PCAEncoder.fit(x)`` computes the mean and top-``latent_dim`` principal
components of the flattened ``(B*T, D)`` observations via ``torch.linalg.svd``.
Forward is ``(x - mean) @ components.T``. It has no trainable parameters; the
fitted statistics live in buffers so they serialise through ``state_dict``.
"""

from __future__ import annotations

import torch
from torch import Tensor

from nssc.representations.base import Decoder, Encoder
from nssc.utils.registry import DECODERS, ENCODERS


def explained_variance_curve(x: Tensor) -> Tensor:
    """Cumulative explained-variance ratio per number of components.

    ``x``: (B,T,D) or (N,D). Returns a tensor of shape (D,) whose k-th entry is
    the fraction of variance captured by the top-(k+1) principal components.
    """
    flat = x.reshape(-1, x.shape[-1]).to(torch.float32)
    flat = flat - flat.mean(0, keepdim=True)
    s = torch.linalg.svdvals(flat)
    var = s**2
    total = var.sum().clamp_min(1e-12)
    curve = torch.cumsum(var, 0) / total
    d = x.shape[-1]
    if curve.numel() < d:  # rank-deficient (N < D): pad with 1.0
        curve = torch.cat([curve, torch.ones(d - curve.numel(), dtype=curve.dtype)])
    return curve


@ENCODERS.register("pca")
class PCAEncoder(Encoder):
    is_causal = True
    is_pointwise = True
    requires_fit = True

    def __init__(self, obs_dim: int, latent_dim: int, center: bool = True, **_: object) -> None:
        super().__init__(obs_dim, latent_dim)
        self.center = center
        self.register_buffer("mean", torch.zeros(obs_dim))
        self.register_buffer("components", torch.eye(latent_dim, obs_dim))
        self.register_buffer("explained_variance_ratio", torch.zeros(latent_dim))
        self.register_buffer("fitted", torch.tensor(False))

    @torch.no_grad()
    def fit(self, x: Tensor) -> None:
        """Fit mean + components on ``x`` of shape (B,T,D) or (N,D)."""
        flat = x.reshape(-1, self.obs_dim).to(self.mean.dtype)
        mean = flat.mean(0) if self.center else torch.zeros_like(self.mean)
        centered = flat - mean
        _, s, vh = torch.linalg.svd(centered, full_matrices=False)
        d = self.latent_dim
        comps = vh[:d]
        if comps.shape[0] < d:  # fewer samples than latent dims: pad with zeros
            pad = torch.zeros(d - comps.shape[0], self.obs_dim, dtype=comps.dtype)
            comps = torch.cat([comps, pad], 0)
        var = s**2
        evr = var[:d] / var.sum().clamp_min(1e-12)
        if evr.numel() < d:
            evr = torch.cat([evr, torch.zeros(d - evr.numel(), dtype=evr.dtype)])
        self.mean.copy_(mean)
        self.components.copy_(comps)
        self.explained_variance_ratio.copy_(evr)
        self.fitted.fill_(True)

    def forward(self, x: Tensor) -> Tensor:
        return (x - self.mean) @ self.components.T


@DECODERS.register("pca")
class PCADecoder(Decoder):
    """Linear inverse of :class:`PCAEncoder`: ``x̂ = z @ components + mean``."""

    def __init__(self, latent_dim: int, obs_dim: int, **_: object) -> None:
        super().__init__(latent_dim, obs_dim)
        self.register_buffer("mean", torch.zeros(obs_dim))
        self.register_buffer("components", torch.eye(latent_dim, obs_dim))

    @torch.no_grad()
    def tie(self, encoder: PCAEncoder) -> PCADecoder:
        self.mean.copy_(encoder.mean)
        self.components.copy_(encoder.components)
        return self

    def forward(self, z: Tensor) -> Tensor:
        return z @ self.components + self.mean
