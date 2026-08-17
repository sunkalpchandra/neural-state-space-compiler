"""Out-of-distribution evaluation for a trained checkpoint.

Two protocols (Experiments G and H):

* ``param_shifts``: re-simulate the training system with shifted dynamical parameters
  (e.g. Lorenz ρ ∈ {20, 35} after training at ρ = 28) and evaluate the frozen model
  with the *training* normalisation statistics.
* ``ic_scales``: re-simulate with a widened initial-condition distribution and no
  transient (so states start off-attractor).

Reports in-distribution reference metrics, per-condition metrics and
``degradation_ratio`` = OOD rollout NRMSE / ID rollout NRMSE.
"""

from __future__ import annotations

import copy
from typing import Any

import numpy as np
import torch

from nssc.data.builder import build_dataset
from nssc.evaluation.evaluator import EvalConfig, evaluate_model
from nssc.experiment import prepare_data
from nssc.training.checkpoint import load_checkpoint


def _normalize(x: np.ndarray, stats: dict[str, Any]) -> np.ndarray:
    mean, std = np.asarray(stats["mean"], np.float32), np.asarray(stats["std"], np.float32)
    return ((x - mean) / std).astype(np.float32)


def evaluate_ood(checkpoint: str, param_shifts: dict[str, list[float]] | None = None,
                 ic_scales: list[float] | None = None, n_traj: int = 20, eval_cfg: EvalConfig | None = None,
                 device: torch.device | None = None, ref_key: str = "recursive/nrmse_mean",
                 seed: int = 1234) -> dict[str, Any]:
    model, meta = load_checkpoint(checkpoint)
    dcfg = dict(meta["dataset"])
    stats = meta["norm_stats"]
    ecfg = eval_cfg or EvalConfig(latency=False, stability=False)
    splits, _, raw = prepare_data(dcfg)
    ref = evaluate_model(model, torch.from_numpy(splits["test"].x), ecfg, sigma=np.ones(raw.obs_dim),
                         device=device)
    out: dict[str, Any] = {"reference": _slim(ref), "conditions": [], "ref_key": ref_key}
    conds: list[tuple[str, dict[str, Any]]] = []
    for pname, values in (param_shifts or {}).items():
        for v in values:
            c = copy.deepcopy(dcfg)
            c.setdefault("params", {})
            c["params"] = dict(c["params"], **{pname: float(v)})
            conds.append((f"param:{pname}={v}", c))
    for s in ic_scales or []:
        c = copy.deepcopy(dcfg)
        c["ic_scale"], c["transient"] = float(s), 0
        conds.append((f"ic_scale={s}", c))
    for name, c in conds:
        c["n_traj"], c["seed"] = int(n_traj), int(seed)
        c.pop("split", None)
        try:
            ds = build_dataset(c)
            x = torch.from_numpy(_normalize(ds.x, stats))
            m = evaluate_model(model, x, ecfg, sigma=np.ones(raw.obs_dim), device=device)
            r = _slim(m)
            r["degradation_ratio"] = float(m[ref_key] / max(ref[ref_key], 1e-12))
            r["condition"], r["status"] = name, "ok"
        except Exception as e:  # noqa: BLE001
            r = {"condition": name, "status": "failed", "error": str(e)}
        out["conditions"].append(r)
    ok = [c["degradation_ratio"] for c in out["conditions"] if c.get("status") == "ok"]
    out["ood/degradation_ratio_mean"] = float(np.mean(ok)) if ok else float("nan")
    out["ood/degradation_ratio_max"] = float(np.max(ok)) if ok else float("nan")
    return out


def _slim(m: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in m.items() if isinstance(v, (int, float, str)) and not k.startswith("latency")}
