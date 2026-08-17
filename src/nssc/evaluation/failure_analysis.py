"""Automated failure categorisation for trained latent models.

Categories (a run may receive several):

    representation_collapse   latent variance concentrated in ≪ d dims / near-constant z
    poor_reconstruction       recon NRMSE ≫ noise floor
    latent_instability        rollout blow-up / spectral radius ≫ 1
    chaotic_divergence        one-step good, rollout diverges at rate consistent with positive λ
    poor_long_horizon         rollout error high without blow-up or chaos signature
    overfitting               train loss ≪ val loss
    underfitting              train loss high and still decreasing at the end
    noise_sensitivity         (needs paired noisy/clean runs) — flagged from tags only
    ood_failure               (needs OOD eval) — flagged from metrics keys if present
    training_failure          run failed / non-finite loss

Thresholds are explicit and configurable; the output records the evidence used.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from nssc.models.latent_model import LatentModel


@dataclass
class FailureThresholds:
    recon_nrmse_bad: float = 0.3
    rollout_nrmse_bad: float = 0.8
    one_step_good: float = 0.1
    rho_unstable: float = 1.5
    blowup_frac: float = 0.1
    overfit_ratio: float = 3.0
    underfit_train_loss: float = 0.5
    collapse_var_ratio: float = 0.02  # a latent dim with < 2% of total variance is "dead"
    collapse_frac_dead: float = 0.5


@dataclass
class FailureReport:
    categories: list[str] = field(default_factory=list)
    evidence: dict[str, Any] = field(default_factory=dict)
    verdict: str = "ok"

    def to_dict(self) -> dict[str, Any]:
        return {"categories": self.categories, "evidence": self.evidence, "verdict": self.verdict}


@torch.no_grad()
def latent_variance_profile(model: LatentModel, x: torch.Tensor) -> dict[str, Any]:
    z = model.encode(x).reshape(-1, model.latent_dim)
    var = z.var(dim=0)
    tot = float(var.sum()) + 1e-12
    ratios = (var / tot).cpu().numpy()
    return {"var_ratio": ratios.tolist(), "n_dead": int((ratios < 0.02).sum()),
            "effective_dim": float(np.exp(-(ratios * np.log(ratios + 1e-12)).sum()))}


def categorize(metrics: dict[str, Any], history: list[dict[str, Any]] | None = None,
               latent_profile: dict[str, Any] | None = None, thr: FailureThresholds | None = None,
               split: str = "val") -> FailureReport:
    thr = thr or FailureThresholds()
    m = metrics.get(split, metrics) if isinstance(metrics.get(split), dict) else metrics
    p = "" if split in metrics and isinstance(metrics.get(split), dict) else f"{split}/"

    def g(k: str, default=float("nan")) -> float:
        v = m.get(p + k, m.get(k, default))
        try:
            return float(v)
        except (TypeError, ValueError):
            return default

    rep = FailureReport()
    ev = rep.evidence
    if metrics.get("status") == "failed":
        rep.categories.append("training_failure")
    recon, one, roll = g("recon/nrmse"), g("teacher_forced/nrmse"), g("recursive/nrmse_mean")
    rho, blow, lam = g("stability/rho_max"), g("stability/frac_blowup"), g("stability/lyapunov_max")
    ev.update({"recon": recon, "one_step": one, "rollout": roll, "rho_max": rho, "frac_blowup": blow,
               "lyapunov": lam})
    if np.isfinite(recon) and recon > thr.recon_nrmse_bad:
        rep.categories.append("poor_reconstruction")
    if (np.isfinite(blow) and blow > thr.blowup_frac) or (np.isfinite(rho) and rho > thr.rho_unstable):
        rep.categories.append("latent_instability")
    if np.isfinite(roll) and roll > thr.rollout_nrmse_bad:
        if np.isfinite(one) and one < thr.one_step_good and np.isfinite(lam) and lam > 0.01:
            rep.categories.append("chaotic_divergence")
        elif "latent_instability" not in rep.categories:
            rep.categories.append("poor_long_horizon")
    if latent_profile:
        ev["latent"] = latent_profile
        d = len(latent_profile["var_ratio"])
        if d > 1 and latent_profile["n_dead"] / d >= thr.collapse_frac_dead:
            rep.categories.append("representation_collapse")
    if history:
        tr = [h.get("train/total") for h in history if h.get("train/total") is not None]
        va = [h.get("val/total") for h in history if h.get("val/total") is not None]
        if tr and va:
            ev["train_final"], ev["val_final"] = tr[-1], va[-1]
            if va[-1] > thr.overfit_ratio * max(tr[-1], 1e-8) and va[-1] > 0.05:
                rep.categories.append("overfitting")
        if tr and len(tr) >= 5:
            slope = (tr[-1] - tr[-5]) / 4
            if tr[-1] > thr.underfit_train_loss and slope < -0.01 * tr[-1]:
                rep.categories.append("underfitting")
        if tr and not np.isfinite(tr[-1]):
            rep.categories.append("training_failure")
    if any(k.startswith("ood/") for k in m) and g("ood/degradation_ratio", 1.0) > 2.0:
        rep.categories.append("ood_failure")
    rep.verdict = "ok" if not rep.categories else "failed:" + "+".join(rep.categories)
    return rep


def analyze_run(output_dir: str, split: str = "val") -> FailureReport:
    """Categorise a finished run from its on-disk artefacts (metrics.json, history.json, checkpoint)."""
    from pathlib import Path

    from nssc.utils.io import load_json

    d = Path(output_dir)
    metrics = load_json(d / "metrics.json") if (d / "metrics.json").exists() else {"status": "failed"}
    history = load_json(d / "history.json")["history"] if (d / "history.json").exists() else None
    prof = None
    if (d / "checkpoint").exists():
        try:
            from nssc.experiment import prepare_data
            from nssc.training import load_checkpoint

            model, meta = load_checkpoint(d / "checkpoint")
            splits, _, _ = prepare_data(meta["dataset"])
            prof = latent_variance_profile(model, torch.from_numpy(splits[split].x[:32]))
        except Exception as e:  # noqa: BLE001
            prof = {"error": str(e), "var_ratio": [], "n_dead": 0, "effective_dim": float("nan")}
    return categorize(metrics, history, prof if prof and prof.get("var_ratio") else None, split=split)
