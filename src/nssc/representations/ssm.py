"""Lightweight diagonal linear state-space encoder (S4D / Mamba-lite style).

Each layer holds, per model channel ``h`` (``d_model`` channels) and per state
``n`` (``d_state`` states), a *real diagonal* continuous-time system

    ẋ = A x + B u,   y = C x + D u,   A = -exp(log_a) < 0   (stable by construction)

discretised with zero-order hold at a learnable step Δ (per channel):

    Ā = exp(Δ A),   B̄ = (Ā - 1) / A · B .

The recurrence ``x_t = Ā x_{t-1} + B̄ u_t`` is evaluated with a *chunked scan*:
inside each chunk of length ``L`` the response is computed in parallel from the
materialised powers ``Ā^k`` (a small ``L×L`` lower-triangular kernel), and the
chunk-final states are carried sequentially across ``T / L`` chunks. This is
exactly causal: ``y_t`` depends only on ``u_{≤t}``.

Each SSM layer is followed by GLU mixing and wrapped in a pre-LayerNorm
residual block; ``n_layers`` blocks are stacked; a linear head projects to
``latent_dim``.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from nssc.representations.base import Encoder
from nssc.utils.registry import ENCODERS


def chunked_diagonal_scan(a_bar: Tensor, bu: Tensor, chunk: int = 32) -> Tensor:
    """Causal linear recurrence ``x_t = a_bar * x_{t-1} + bu_t`` with constant diagonal ``a_bar``.

    ``a_bar``: (H,N) in (0,1);  ``bu``: (B,T,H,N).  Returns ``x``: (B,T,H,N).
    """
    b, t, h, n = bu.shape
    length = min(chunk, t)
    n_chunks = math.ceil(t / length)
    pad = n_chunks * length - t
    if pad:
        bu = F.pad(bu, (0, 0, 0, 0, 0, pad))
    bu = bu.reshape(b, n_chunks, length, h, n)

    # powers[k] = a_bar^k for k = 0..L  (L+1 entries; index L is the chunk-carry factor)
    ks = torch.arange(length + 1, device=bu.device)
    log_a = torch.log(a_bar.clamp_min(1e-30))  # (H,N)
    powers = torch.exp(ks.to(bu.dtype)[:, None, None] * log_a)  # (L+1,H,N)

    # intra-chunk kernel K[t,s] = a_bar^(t-s) for t >= s else 0
    idx = ks[:length, None] - ks[None, :length]  # (L,L) long
    mask = idx >= 0
    kern = powers[idx.clamp_min(0)] * mask[:, :, None, None]  # (L,L,H,N)
    local = torch.einsum("tshn,bcshn->bcthn", kern, bu)  # (B,C,L,H,N)

    # sequential carry across chunks: state entering chunk c
    carry = torch.zeros(b, h, n, device=bu.device, dtype=bu.dtype)
    a_pow_l = powers[length]  # (H,N)
    carries = []
    for c in range(n_chunks):
        carries.append(carry)
        carry = a_pow_l * carry + local[:, c, -1]
    carry_in = torch.stack(carries, 1)  # (B,C,H,N)
    # contribution of carry to position t within chunk is a_bar^(t+1) * carry_in
    x = local + powers[1 : length + 1][None, None] * carry_in[:, :, None]
    x = x.reshape(b, n_chunks * length, h, n)
    return x[:, :t]


class DiagonalSSM(nn.Module):
    """Per-channel real-diagonal SSM layer, (B,T,H) -> (B,T,H)."""

    def __init__(self, d_model: int, d_state: int = 16, dt_min: float = 1e-3,
                 dt_max: float = 1e-1, chunk: int = 32) -> None:
        super().__init__()
        self.d_model, self.d_state, self.chunk = d_model, d_state, chunk
        # S4D-real style init: A_n = -(n+1)
        a_init = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(d_model, 1)
        self.log_a = nn.Parameter(torch.log(a_init))
        self.B = nn.Parameter(torch.ones(d_model, d_state))
        self.C = nn.Parameter(torch.randn(d_model, d_state) / math.sqrt(d_state))
        self.D = nn.Parameter(torch.ones(d_model))
        log_dt = torch.rand(d_model) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        self.log_dt = nn.Parameter(log_dt)

    def discretize(self) -> tuple[Tensor, Tensor]:
        a = -torch.exp(self.log_a)  # (H,N) < 0
        dt = torch.exp(self.log_dt)[:, None]  # (H,1)
        a_bar = torch.exp(dt * a)
        b_bar = (a_bar - 1.0) / a * self.B
        return a_bar, b_bar

    def forward(self, u: Tensor) -> Tensor:  # (B,T,H)
        a_bar, b_bar = self.discretize()
        bu = u.unsqueeze(-1) * b_bar  # (B,T,H,N)
        x = chunked_diagonal_scan(a_bar, bu, self.chunk)
        y = torch.einsum("bthn,hn->bth", x, self.C)
        return y + u * self.D


class SSMBlock(nn.Module):
    def __init__(self, d_model: int, d_state: int, expand: int, dropout: float, chunk: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.ssm = DiagonalSSM(d_model, d_state, chunk=chunk)
        self.glu_in = nn.Linear(d_model, 2 * expand * d_model)
        self.glu_out = nn.Linear(expand * d_model, d_model)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, h: Tensor) -> Tensor:
        y = self.ssm(self.norm(h))
        y = self.glu_out(F.glu(self.glu_in(F.gelu(y)), dim=-1))
        return h + self.drop(y)


@ENCODERS.register("ssm")
class SSMEncoder(Encoder):
    is_causal = True
    is_pointwise = False
    requires_fit = False

    def __init__(self, obs_dim: int, latent_dim: int, d_model: int = 64, d_state: int = 16,
                 n_layers: int = 2, expand: int = 2, dropout: float = 0.0, chunk: int = 32,
                 **_: object) -> None:
        super().__init__(obs_dim, latent_dim)
        self.inp = nn.Linear(obs_dim, d_model)
        self.blocks = nn.ModuleList(
            SSMBlock(d_model, d_state, expand, dropout, chunk) for _ in range(n_layers)
        )
        self.norm = nn.LayerNorm(d_model)
        self.out = nn.Linear(d_model, latent_dim)

    def forward(self, x: Tensor) -> Tensor:
        h = self.inp(x)
        for blk in self.blocks:
            h = blk(h)
        return self.out(self.norm(h))
