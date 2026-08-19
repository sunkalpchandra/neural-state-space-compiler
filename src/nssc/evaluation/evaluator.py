"""Standard evaluation of a LatentModel on held-out trajectories.

Evaluation modes are always labelled explicitly:

* ``recon``            x̂ = D(E(x))
* ``teacher_forced``   x̂_{t+1} = D(F(E(x)_t))   (one-step, ground-truth history)
* ``recursive``        encode ``context`` steps, roll F forward H steps, decode
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
from torch import Tensor

from nssc.metrics import (
    count_parameters,
    estimate_flops_per_step,
    measure_inference_latency,
    mse,
    nrmse,
    rollout_divergence_time,
    rollout_errors,
)
from nssc.metrics.prediction import DEFAULT_HORIZONS, horizon_curve
from nssc.models.latent_model import LatentModel
from nssc.stability import analyze_stability


@dataclass
class EvalConfig:
    context: int = 20
    horizons: tuple[int, ...] = DEFAULT_HORIZONS
    max_horizon: int | None = None  # default: max(horizons) clipped to T - context
    batch_size: int = 64
    stability: bool = True
    stability_horizon: int = 200
    latency: bool = True
    latency_horizon: int = 50  # steps in the comparable end-to-end forecast latency measurement
    divergence_threshold: float = 1.0
    extra: dict[str, Any] = field(default_factory=dict)


def _batches(x: Tensor, bs: int):
    for i in range(0, x.shape[0], bs):
        yield x[i : i + bs]


@torch.no_grad()
def evaluate_model(model: LatentModel, x: Tensor, cfg: EvalConfig | None = None,
                   sigma: np.ndarray | None = None, device: torch.device | None = None,
                   dt: float = 1.0) -> dict[str, Any]:
    """``x``: (N, T, D) held-out, normalised like training data. Returns flat metrics dict
    plus ``curves`` (per-step NRMSE) and ``stability`` sub-dicts."""
    cfg = cfg or EvalConfig()
    device = device or next(iter(model.parameters()), torch.zeros(1)).device
    model.eval().to(device)
    x = x.to(device).float()
    N, T, D = x.shape
    if sigma is None:
        sigma = x.reshape(-1, D).std(0).cpu().numpy()
    H = cfg.max_horizon or max(cfg.horizons)
    H = min(H, T - cfg.context)
    assert H >= 1, "sequence too short for context + horizon"

    recon, tf_pred, tf_tgt, roll_pred, roll_tgt = [], [], [], [], []
    for xb in _batches(x, cfg.batch_size):
        recon.append(model.reconstruct(xb))
        xn, _, _ = model.predict_teacher_forced(xb)
        tf_pred.append(xn)
        tf_tgt.append(xb[:, 1:])
        xr, _ = model.rollout(xb[:, : cfg.context], H)
        roll_pred.append(xr)
        roll_tgt.append(xb[:, cfg.context : cfg.context + H])
    recon = torch.cat(recon)
    tf_pred, tf_tgt = torch.cat(tf_pred), torch.cat(tf_tgt)
    roll_pred, roll_tgt = torch.cat(roll_pred), torch.cat(roll_tgt)
    roll_pred = torch.nan_to_num(roll_pred, nan=1e6, posinf=1e6, neginf=-1e6)

    c = max(cfg.context - 1, 0)  # first position a context-window model has full history for
    out: dict[str, Any] = {
        "recon/mse": mse(recon, x),
        "recon/nrmse": nrmse(recon, x, sigma),
        "teacher_forced/mse": mse(tf_pred, tf_tgt),
        "teacher_forced/nrmse": nrmse(tf_pred, tf_tgt, sigma),
        # Baselines are only *trained* on positions t >= context-1, but the metric above averages
        # from t=0. This second, position-matched metric makes the one-step comparison fair in both
        # directions (review finding R-05); both are reported, neither replaces the other.
        "teacher_forced_ctx/nrmse": nrmse(tf_pred[:, c:], tf_tgt[:, c:], sigma),
        "recursive/horizon": H,
        "recursive/context": cfg.context,
    }
    for k, v in rollout_errors(roll_pred, roll_tgt, cfg.horizons, sigma).items():
        out[f"recursive/{k}"] = v
    out["recursive/nrmse_mean"] = float(np.mean(horizon_curve(roll_pred, roll_tgt, sigma)))
    out["recursive/divergence_time"] = rollout_divergence_time(roll_pred, roll_tgt,
                                                               cfg.divergence_threshold, sigma)
    out["curves"] = {"recursive_nrmse": horizon_curve(roll_pred, roll_tgt, sigma).tolist()}

    # ------------------------------------------------------------ complexity
    counts = model.num_parameters()
    out.update({f"params/{k}": v for k, v in counts.items()})
    out["params/total"] = count_parameters(model)
    out["latent_dim"] = model.latent_dim
    z0 = model.encode(x[:1, : cfg.context])[:, -1]
    out["flops/dynamics_step"] = estimate_flops_per_step(model.dynamics, z0)
    if cfg.latency:
        out.update({f"latency/step_{k}": v for k, v in
                    measure_inference_latency(lambda: model.dynamics.step(z0), device=device).items()})
        xc = x[:1, : cfg.context]
        out.update({f"latency/encode_{k}": v for k, v in
                    measure_inference_latency(lambda: model.encode(xc), n_iters=10, device=device).items()})
        # Protocol-comparable cost: one full forecast of ``cfg.latency_horizon`` observation-space
        # steps from the same context, measured identically for latent models and baselines
        # (review finding R-49: dynamics-step latency alone is not comparable — a latent step is a
        # 3-dim MLP, a baseline step re-runs the whole backbone and also decodes).
        lh = min(cfg.latency_horizon, H)
        out["latency/horizon"] = lh
        out.update({f"latency/forecast{lh}_{k}": v for k, v in
                    measure_inference_latency(lambda: model.rollout(xc, lh), n_iters=10,
                                              device=device).items()})

    # -------------------------------------------------------------- stability
    if cfg.stability:
        with torch.enable_grad():
            rep = analyze_stability(model, x[: min(64, N)], horizon=min(cfg.stability_horizon, 4 * T),
                                    dt=dt)
        out["stability"] = rep.to_dict()
        out["stability/instability_score"] = rep.instability_score
        out["stability/rho_max"] = rep.spectral["rho_max"]
        out["stability/lyapunov_max"] = rep.lyapunov["lyapunov_max_mean"]
        out["stability/frac_blowup"] = rep.norm_growth["frac_blowup"]
        out["stability/verdict"] = rep.verdict
    return out
