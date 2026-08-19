"""Multi-objective scoring of candidates.

    J = λ1·L_recon + λ2·L_1step + λ3·L_rollout + λ4·C_complexity + λ5·C_instability

Every term is normalised *within the candidate pool being compared* so the
weights are dimensionless and comparable across datasets:

* error terms (recon / one-step / rollout NRMSE): ``log(x / x_min)`` — 0 for the
  best candidate, +log-ratio otherwise (robust to scale, penalises 2× error the
  same everywhere).
* complexity: ``log(params / params_min) / log(10)`` — one unit per decade of
  parameters.
* instability: ``instability_score`` (already ≥0; 0 = stable) plus a hard
  ``blowup_penalty`` if a run diverged.

Two selection criteria are supported to enable the H2 ablation:
``criterion: multi_objective`` (default) and ``criterion: val_mse``
(rank purely by validation one-step/recon MSE, ignoring rollout/stability/complexity).
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ScoreWeights:
    reconstruction: float = 1.0
    one_step: float = 1.0
    rollout: float = 2.0
    complexity: float = 0.1
    stability: float = 1.0
    blowup_penalty: float = 10.0
    rollout_horizon_key: str = "auto"  # metric key e.g. "recursive/nrmse@100" or "auto"
    error_floor: float = 0.01  # NRMSE floor inside log-ratios: log((x+f)/(best+f)); prevents an
    #                             exactly-zero best (e.g. PCA d=D) from penalising everyone infinitely
    criterion: str = "multi_objective"  # multi_objective | val_mse | rollout_only
    extra: dict[str, float] = field(default_factory=dict)

    @classmethod
    def from_config(cls, cfg: dict[str, Any] | None) -> ScoreWeights:
        cfg = dict(cfg or {})
        known = {k: v for k, v in cfg.items() if k in cls.__dataclass_fields__}
        return cls(**known)


def _params(m: dict[str, Any], prefix: str) -> float:
    """Model size for the complexity term: stored (parameters + buffers) when available.

    Falls back to trainable parameters for rows written before ``params/total_stored`` existed.
    """
    v = _get(m, f"{prefix}params/total_stored")
    return v if math.isfinite(v) else _get(m, f"{prefix}params/total")


def _get(m: dict[str, Any], key: str, default: float = float("nan")) -> float:
    v = m.get(key, default)
    try:
        v = float(v)
    except (TypeError, ValueError):
        return default
    return v if math.isfinite(v) else default


def pick_rollout_key(metrics_list: list[dict[str, Any]], prefix: str = "val/") -> str:
    """Longest horizon available in *all* candidates' metrics (fallback nrmse_mean)."""
    horizons = None
    for m in metrics_list:
        hs = {int(k.split("@")[1]) for k in m
              if k.startswith(f"{prefix}recursive/nrmse@") and k.split("@")[1].isdigit()}
        horizons = hs if horizons is None else horizons & hs
    if horizons:
        return f"{prefix}recursive/nrmse@{max(horizons)}"
    return f"{prefix}recursive/nrmse_mean"


def _stat(values: list[float]) -> float:
    vals = [v for v in values if math.isfinite(v)]
    return sum(vals) / len(vals) if vals else float("nan")


def aggregate_seeds(runs: list[dict[str, Any]]) -> dict[str, Any]:
    """Mean over seeds of scalar summary metrics; carries n_seeds and failure count."""
    keys: set[str] = set()
    completed = [r for r in runs if r.get("status") == "completed"]
    for r in completed:
        keys |= {k for k, v in r.get("summary", {}).items() if isinstance(v, (int, float))}
    agg: dict[str, Any] = {}
    for k in keys:
        vals = [float(r["summary"][k]) for r in completed if k in r["summary"]]
        agg[k] = _stat(vals)
        finite = [v for v in vals if math.isfinite(v)]
        # spread across seeds, so a headline margin can never be quoted without it (review R-46)
        agg[f"{k}__std"] = (statistics.stdev(finite) if len(finite) > 1 else 0.0)
        agg[f"{k}__min"] = min(finite) if finite else float("nan")
        agg[f"{k}__max"] = max(finite) if finite else float("nan")
    agg["n_seeds"] = len(completed)
    agg["n_failed"] = len(runs) - len(completed)
    # Stability across seeds is reported worst-case, not by majority: a single exploding seed is
    # the scientifically relevant fact (review finding R-20 — a candidate whose 1/3 seeds blew up
    # was labelled "stable"). ``n_unstable_seeds`` keeps the count visible in the report.
    order = ["stable", "locally_expanding", "collapses", "explodes", "failed"]
    verdicts = [r.get("summary", {}).get("val/stability/verdict", "failed") for r in completed]
    agg["val/stability/verdict"] = (max(verdicts, key=lambda v: order.index(v) if v in order else len(order))
                                    if verdicts else "failed")
    agg["val/stability/verdict_by_seed"] = verdicts
    agg["val/stability/n_unstable_seeds"] = sum(1 for v in verdicts if v != "stable")
    blow = [float(r.get("summary", {}).get("val/stability/frac_blowup", 0.0) or 0.0) for r in completed]
    agg["val/stability/frac_blowup_max"] = max(blow) if blow else float("nan")
    return agg


class MultiObjectiveScorer:
    def __init__(self, weights: ScoreWeights, split: str = "val") -> None:
        self.w = weights
        self.p = f"{split}/"

    def terms(self, agg: dict[str, Any], pool: list[dict[str, Any]], rollout_key: str
             ) -> dict[str, float]:
        p = self.p
        best = {
            "recon": min((_get(a, f"{p}recon/nrmse") for a in pool), default=float("nan")),
            "one_step": min((_get(a, f"{p}teacher_forced/nrmse") for a in pool), default=float("nan")),
            "rollout": min((_get(a, rollout_key) for a in pool), default=float("nan")),
            "params": min((_params(a, p) for a in pool), default=float("nan")),
        }

        f = self.w.error_floor

        def logratio(x: float, b: float, floor: float = f) -> float:
            if not (math.isfinite(x) and math.isfinite(b)) or b < 0:
                return float("nan")
            return math.log((max(x, 0.0) + floor) / (max(b, 0.0) + floor))

        t = {
            "recon": logratio(_get(agg, f"{p}recon/nrmse"), best["recon"]),
            "one_step": logratio(_get(agg, f"{p}teacher_forced/nrmse"), best["one_step"]),
            "rollout": logratio(_get(agg, rollout_key), best["rollout"]),
            "complexity": logratio(_params(agg, p), best["params"], floor=1.0) / math.log(10),
            "instability": _get(agg, f"{p}stability/instability_score", 0.0),
            "blowup": _get(agg, f"{p}stability/frac_blowup", 0.0),
        }
        return t

    def score(self, agg: dict[str, Any], pool: list[dict[str, Any]], rollout_key: str
              ) -> tuple[float, dict[str, float]]:
        w = self.w
        t = self.terms(agg, pool, rollout_key)
        if agg.get("n_seeds", 0) == 0:
            return float("inf"), t
        nan_pen = 5.0  # a missing/NaN term is treated as bad, not free

        def g(k: str) -> float:
            v = t[k]
            return v if math.isfinite(v) else nan_pen

        if w.criterion == "val_mse":
            J = 0.5 * g("recon") + 0.5 * g("one_step")
        elif w.criterion == "rollout_only":
            J = g("rollout")
        else:
            J = (w.reconstruction * g("recon") + w.one_step * g("one_step") + w.rollout * g("rollout")
                 + w.complexity * g("complexity") + w.stability * g("instability")
                 + w.blowup_penalty * g("blowup"))
        return float(J), t

    def rank(self, per_candidate: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
        """``per_candidate``: cand_id → list of run results (one per seed). Returns rows sorted by J."""
        aggs = {cid: aggregate_seeds(runs) for cid, runs in per_candidate.items()}
        pool = [a for a in aggs.values() if a.get("n_seeds", 0) > 0]
        rk = self.w.rollout_horizon_key
        if rk == "auto":
            rk = pick_rollout_key(pool, self.p) if pool else f"{self.p}recursive/nrmse_mean"
        rows = []
        for cid, agg in aggs.items():
            J, terms = self.score(agg, pool, rk)
            rows.append({"candidate_id": cid, "score": J, "terms": terms, "agg": agg,
                         "rollout_key": rk})
        rows.sort(key=lambda r: (not math.isfinite(r["score"]), r["score"]))
        for i, r in enumerate(rows):
            r["rank"] = i + 1
        return rows
