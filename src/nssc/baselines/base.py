"""Common interface for "large sequence model" comparators and trivial baselines.

A :class:`SequenceForecaster` maps an observation history directly to future
observations, with **no latent state-space bottleneck**. It is the comparator
for the central hypothesis (compiled latent SSM vs. plain sequence model).

Two prediction modes (fixed at construction):

* ``recursive`` — the model predicts one step ahead; :meth:`forecast` appends the
  prediction to the context and repeats (autoregressive rollout).
* ``direct``    — a second head predicts ``direct_horizon`` steps at once from
  the context; horizons longer than that are covered by block-wise recursion.

Neural subclasses only implement :meth:`backbone` — causal per-position features
``(B,T,F)`` — plus ``feature_dim``; the base class supplies the heads and
everything else. Non-parametric baselines override the ``predict_*`` methods.

Shapes: ``x`` is ``(B, T, D)`` (batch, time, obs_dim) unless stated otherwise.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class SequenceForecaster(nn.Module):
    """Base forecaster: ``x_{≤t} → x̂_{t+1}`` (and optionally ``x̂_{t+1:t+H}``)."""

    registry_key: str = "base"
    #: longest history the model can consume in one call (None = unlimited). ``forecast``
    #: truncates the growing context to the last ``max_context`` steps.
    max_context: int | None = None

    def __init__(self, obs_dim: int, mode: str = "recursive", direct_horizon: int | None = None,
                 **_: object) -> None:
        super().__init__()
        if mode not in ("recursive", "direct"):
            raise ValueError(f"mode must be 'recursive' or 'direct', got {mode!r}")
        if mode == "direct" and not direct_horizon:
            raise ValueError("direct mode needs direct_horizon >= 1")
        self.obs_dim = int(obs_dim)
        self.mode = mode
        self.direct_horizon = int(direct_horizon) if direct_horizon else None
        self.head: nn.Module | None = None
        self.direct_head: nn.Module | None = None

    # ------------------------------------------------------------ to override
    @property
    def feature_dim(self) -> int:
        raise NotImplementedError

    def backbone(self, x: Tensor) -> Tensor:
        """Causal features ``(B,T,F)``: position ``t`` depends on ``x_{≤t}`` only."""
        raise NotImplementedError

    def _build_heads(self) -> None:
        """Call at the end of a neural subclass ``__init__`` (needs ``feature_dim``)."""
        f = self.feature_dim
        self.head = nn.Linear(f, self.obs_dim)
        if self.mode == "direct":
            self.direct_head = nn.Linear(f, self.direct_horizon * self.obs_dim)

    # ---------------------------------------------------------------- predict
    def predict_next_sequence(self, x: Tensor) -> Tensor:
        """``(B,T,D) → (B,T,D)``: entry ``t`` is the prediction of ``x_{t+1}`` from ``x_{≤t}``."""
        return self.head(self.backbone(x))

    def predict_next(self, x: Tensor) -> Tensor:
        """``(B,T,D) → (B,D)``: prediction of the step following the window."""
        return self.predict_next_sequence(x)[:, -1]

    def predict_direct(self, x_context: Tensor) -> Tensor:
        """``(B,C,D) → (B,direct_horizon,D)`` (direct mode only)."""
        if self.direct_head is None:
            raise RuntimeError("predict_direct requires mode='direct'")
        f = self.backbone(x_context)[:, -1]
        return self.direct_head(f).view(x_context.shape[0], self.direct_horizon, self.obs_dim)

    def _trim(self, x: Tensor) -> Tensor:
        return x if self.max_context is None or x.shape[1] <= self.max_context else x[:, -self.max_context:]

    def forecast(self, x_context: Tensor, horizon: int) -> Tensor:
        """``(B,C,D), H → (B,H,D)``: predictions for the ``H`` steps after the context.

        Recursive mode: one step at a time. Direct mode: ``direct_horizon``-blocks,
        each appended to the context before predicting the next block.
        """
        ctx = x_context
        outs: list[Tensor] = []
        n = 0
        while n < horizon:
            if self.mode == "direct":
                blk = self.predict_direct(self._trim(ctx))[:, : horizon - n]
            else:
                blk = self.predict_next(self._trim(ctx)).unsqueeze(1)
            outs.append(blk)
            ctx = torch.cat([ctx, blk], dim=1)
            if self.max_context is not None and ctx.shape[1] > self.max_context:
                ctx = ctx[:, -self.max_context:]
            n += blk.shape[1]
        return torch.cat(outs, dim=1)

    # ------------------------------------------------------------------ misc
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def config(self) -> dict:
        """Constructor kwargs needed to rebuild (subclasses extend)."""
        return {"obs_dim": self.obs_dim, "mode": self.mode, "direct_horizon": self.direct_horizon}
