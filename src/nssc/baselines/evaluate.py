"""Evaluate a :class:`SequenceForecaster` with the same protocol/keys as
:func:`nssc.evaluation.evaluator.evaluate_model` (minus recon/stability, which
have no meaning without a latent state).

Keys: ``recursive/nrmse@k``, ``recursive/nrmse_step@k``, ``recursive/nrmse_mean``,
``recursive/divergence_time``, ``recursive/horizon``, ``recursive/context``,
``curves.recursive_nrmse``, ``teacher_forced/mse|nrmse`` (one step ahead from
ground-truth history at every position, like ``evaluate_model``), ``params/total``,
``latent_dim`` (None, or the wrapped model's), ``latency/step_latency_ms_*`` (one
``predict_next`` call) and, in direct mode, ``direct/nrmse@k`` / ``direct/nrmse_step@k``.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import torch
from torch import Tensor

from nssc.baselines.base import SequenceForecaster
from nssc.metrics import (
    measure_inference_latency,
    mse,
    nrmse,
    rollout_divergence_time,
    rollout_errors,
)
from nssc.metrics.prediction import DEFAULT_HORIZONS, horizon_curve


def _batches(x: Tensor, bs: int):
    for i in range(0, x.shape[0], bs):
        yield x[i : i + bs]


def _teacher_forced(model: SequenceForecaster, x: Tensor) -> Tensor:
    """Predictions of ``x_{t+1}`` for every ``t`` (B,T,D), chunked when ``T > max_context``."""
    T = x.shape[1]
    M = model.max_context
    if M is None or T <= M:
        return model.predict_next_sequence(x)
    out = torch.empty_like(x)
    out[:, :M] = model.predict_next_sequence(x[:, :M])
    step = max(1, M // 2)  # every later position sees >= M - step history
    done = M
    while done < T:
        s = min(done + step, T) - M
        pred = model.predict_next_sequence(x[:, s : s + M])
        out[:, done : s + M] = pred[:, done - s :]
        done = s + M
    return out


@torch.no_grad()
def evaluate_forecaster(model: SequenceForecaster, x: Tensor, context: int = 20,
                        horizons: Sequence[int] = DEFAULT_HORIZONS, sigma: np.ndarray | None = None,
                        device: torch.device | None = None, max_horizon: int | None = None,
                        batch_size: int = 64, divergence_threshold: float = 1.0,
                        latency: bool = True, latency_horizon: int = 50) -> dict[str, Any]:
    """``x``: (N,T,D) held-out trajectories normalised like the training data."""
    device = device or next(iter(model.parameters()), torch.zeros(1)).device
    model.eval().to(device)
    x = x.to(device).float()
    N, T, D = x.shape
    if sigma is None:
        sigma = x.reshape(-1, D).std(0).cpu().numpy()
    H = max_horizon or max(horizons)
    H = min(H, T - context)
    assert H >= 1, "sequence too short for context + horizon"

    tf_pred, roll_pred, direct_pred = [], [], []
    for xb in _batches(x, batch_size):
        tf_pred.append(_teacher_forced(model, xb)[:, :-1])
        roll_pred.append(model.forecast(xb[:, :context], H))
        if model.mode == "direct":
            direct_pred.append(model.predict_direct(xb[:, :context]))
    tf_pred, tf_tgt = torch.cat(tf_pred), x[:, 1:]
    roll_pred = torch.nan_to_num(torch.cat(roll_pred), nan=1e6, posinf=1e6, neginf=-1e6)
    roll_tgt = x[:, context : context + H]

    out: dict[str, Any] = {
        "teacher_forced/mse": mse(tf_pred, tf_tgt),
        "teacher_forced/nrmse": nrmse(tf_pred, tf_tgt, sigma),
        # Position-matched twin of the metric above: these models are only trained on positions
        # t >= context-1, so averaging one-step error from t=0 penalises them for history they
        # never had (review finding R-05). Both are reported; the latent evaluator reports both too.
        "teacher_forced_ctx/nrmse": nrmse(tf_pred[:, max(context - 1, 0):],
                                          tf_tgt[:, max(context - 1, 0):], sigma),
        "recursive/horizon": H,
        "recursive/context": context,
    }
    for k, v in rollout_errors(roll_pred, roll_tgt, horizons, sigma).items():
        out[f"recursive/{k}"] = v
    curve = horizon_curve(roll_pred, roll_tgt, sigma)
    out["recursive/nrmse_mean"] = float(np.mean(curve))
    out["recursive/divergence_time"] = rollout_divergence_time(roll_pred, roll_tgt,
                                                               divergence_threshold, sigma)
    out["curves"] = {"recursive_nrmse": curve.tolist()}
    if direct_pred:
        dp = torch.nan_to_num(torch.cat(direct_pred), nan=1e6, posinf=1e6, neginf=-1e6)
        hd = min(dp.shape[1], T - context)
        dt_ = x[:, context : context + hd]
        for k, v in rollout_errors(dp[:, :hd], dt_, horizons, sigma).items():
            out[f"direct/{k}"] = v
        out["direct/horizon"] = hd
        out["curves"]["direct_nrmse"] = horizon_curve(dp[:, :hd], dt_, sigma).tolist()

    out["params/total"] = int(model.num_parameters())
    out["params/total_stored"] = int(sum(p.numel() for p in model.parameters())
                                     + sum(b.numel() for b in model.buffers()))
    out["latent_dim"] = getattr(model, "latent_dim", None)
    out["mode"] = model.mode
    if latency:
        xc = x[:1, :context]
        out.update({f"latency/step_{k}": v for k, v in
                    measure_inference_latency(lambda: model.predict_next(xc), device=device).items()})
        # Protocol-comparable cost, identical to nssc.evaluation.evaluator: one full forecast of
        # ``latency_horizon`` observation-space steps from the same context (review finding R-49).
        lh = min(latency_horizon, H)
        out["latency/horizon"] = lh
        out.update({f"latency/forecast{lh}_{k}": v for k, v in
                    measure_inference_latency(lambda: model.forecast(xc, lh), n_iters=10,
                                              device=device).items()})
    return out
