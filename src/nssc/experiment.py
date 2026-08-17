"""Single-run experiment pipeline: dataset → model → train → evaluate → checkpoint → registry.

A *run config* has the shape::

    dataset: {...}            # nssc.data.builder schema, or {_file: configs/datasets/x.yaml}
    model:   {latent_dim, encoder, decoder, dynamics}
    training: {epochs, lr, rollout_horizon, loss: {...}, ...}   (TrainerConfig fields)
    windows: {context: 20, horizon: 30, stride: 5, batch_size: 64}
    eval:    {context: 20, horizons: [...], ...}                  (EvalConfig fields)
    seed: 0
    tags: [...]
    output_dir: results/raw/<name>     (checkpoint + metrics live here)
"""

from __future__ import annotations

import time
import traceback
from dataclasses import fields
from pathlib import Path
from typing import Any

import numpy as np
import torch

from nssc.data.builder import build_dataset
from nssc.data.dataset import TrajectoryDataset, make_loaders
from nssc.evaluation import EvalConfig, evaluate_model
from nssc.models.builder import build_latent_model, model_name
from nssc.training import Trainer, TrainerConfig, save_checkpoint
from nssc.utils.config import Config, load_yaml
from nssc.utils.env import default_device
from nssc.utils.experiment_registry import ExperimentRegistry
from nssc.utils.hashing import stable_hash
from nssc.utils.io import save_json
from nssc.utils.seeding import seed_everything


def _dc(cls, d: dict[str, Any]):
    names = {f.name for f in fields(cls)}
    return cls(**{k: v for k, v in d.items() if k in names})


def resolve_dataset_cfg(dcfg: dict[str, Any]) -> dict[str, Any]:
    if "_file" in dcfg:
        base = load_yaml(dcfg["_file"])
        base.update({k: v for k, v in dcfg.items() if k != "_file"})
        return base
    return dict(dcfg)


def prepare_data(dcfg: dict[str, Any], seed: int | None = None
                 ) -> tuple[dict[str, TrajectoryDataset], dict[str, np.ndarray], TrajectoryDataset]:
    """Build → trajectory-level split → normalise with *train* statistics."""
    dcfg = resolve_dataset_cfg(dcfg)
    ds = build_dataset(dcfg)
    splits = ds.split(seed=seed)
    train_n, stats = splits["train"].normalize()
    out = {"train": train_n}
    for k in ("val", "test"):
        if k in splits:
            out[k], _ = splits[k].normalize(stats)
    return out, stats, ds


def run_experiment(cfg: dict[str, Any] | Config, registry: ExperimentRegistry | None = None,
                   device: torch.device | None = None, log=print, save_ckpt: bool = True
                   ) -> dict[str, Any]:
    """Execute one run. Never raises on model failure: failures are recorded with status='failed'."""
    cfg = Config(dict(cfg))
    seed = int(cfg.get("seed", 0))
    seed_everything(seed)
    device = device or (torch.device(cfg.training.device) if cfg.get("training", {}).get("device")
                        else default_device())
    dcfg = resolve_dataset_cfg(dict(cfg["dataset"]))
    cfg["dataset"] = dcfg
    chash = stable_hash({k: v for k, v in cfg.to_dict().items() if k not in ("output_dir", "tags")})
    registry = registry or ExperimentRegistry()
    mname = model_name(cfg["model"])
    rec = registry.register(config=cfg.to_dict(), config_hash=chash,
                            dataset=dcfg.get("system") or dcfg.get("source") or dcfg.get("name") or "?",
                            model=mname, seed=seed, tags=list(cfg.get("tags", [])))
    out_dir = Path(cfg.get("output_dir", f"results/raw/{rec.experiment_id}"))
    out_dir.mkdir(parents=True, exist_ok=True)
    result: dict[str, Any] = {"experiment_id": rec.experiment_id, "config_hash": chash,
                              "model": mname, "seed": seed, "output_dir": str(out_dir)}
    try:
        t0 = time.perf_counter()
        splits, stats, raw = prepare_data(dcfg, seed=None)
        w = dict(cfg.get("windows", {}))
        loaders = make_loaders(splits, context=int(w.get("context", 20)), horizon=int(w.get("horizon", 30)),
                               batch_size=int(w.get("batch_size", 64)), stride=int(w.get("stride", 5)))
        model = build_latent_model(dict(cfg["model"]), obs_dim=raw.obs_dim)
        tcfg = _dc(TrainerConfig, dict(cfg.get("training", {})))
        trainer = Trainer(model, tcfg, device=device)
        fit = trainer.fit(loaders["train"], loaders.get("val"),
                          log=(lambda r: log(f"[{rec.experiment_id}] ep{r['epoch']} "
                                             f"train={r['train/total']:.4g} val={r.get('val/total', float('nan')):.4g}"))
                          if log else None)
        ecfg = _dc(EvalConfig, dict(cfg.get("eval", {})))
        sigma = np.ones(raw.obs_dim)  # data are normalised → σ = 1 per dim (train stats)
        metrics: dict[str, Any] = {}
        for split_name in ("val", "test"):
            if split_name in splits:
                x = torch.from_numpy(splits[split_name].x)
                m = evaluate_model(model, x, ecfg, sigma=sigma, device=device,
                                   dt=float(raw.metadata.get("dt", dcfg.get("dt", 1.0)) or 1.0))
                metrics[split_name] = m
        if getattr(model.dynamics, "is_stochastic", False) and "test" in splits:
            from nssc.uncertainty import evaluate_uncertainty

            unc = evaluate_uncertainty(model, torch.from_numpy(splits["test"].x).to(device),
                                       context=ecfg.context,
                                       horizon=min(100, splits["test"].x.shape[1] - ecfg.context))
            metrics["test"]["uncertainty"] = unc
            for k in ("nll", "coverage95", "ece", "sharpness", "std_error_corr"):
                metrics["test"][f"uncertainty/{k}"] = unc[k]
        metrics["train/best_val_loss"] = fit["best_val"]
        metrics["train/epochs_run"] = fit["epochs_run"]
        metrics["train/time_s"] = fit["train_time_s"]
        metrics["train/final"] = fit["history"][-1] if fit["history"] else {}
        ckpt = None
        if save_ckpt:
            ckpt = save_checkpoint(model, out_dir / "checkpoint",
                                   metadata={"experiment_id": rec.experiment_id, "seed": seed,
                                             "norm_stats": {k: v.tolist() for k, v in stats.items()},
                                             "dataset": dcfg, "metrics_summary": summarize(metrics)})
        save_json({"history": fit["history"]}, out_dir / "history.json")
        save_json(metrics, out_dir / "metrics.json")
        result.update({"metrics": metrics, "checkpoint": str(ckpt) if ckpt else None,
                       "wall_time_s": time.perf_counter() - t0, "status": "completed",
                       "summary": summarize(metrics)})
        registry.complete(rec, metrics=summarize(metrics), checkpoint=str(ckpt) if ckpt else None,
                          param_count=int(model.num_parameters()["total"]),
                          train_time_s=fit["train_time_s"])
    except Exception as e:  # noqa: BLE001
        tb = traceback.format_exc()
        save_json({"error": str(e), "traceback": tb}, out_dir / "error.json")
        registry.fail(rec, f"{type(e).__name__}: {e}")
        result.update({"status": "failed", "error": str(e), "traceback": tb})
        if log:
            log(f"[{rec.experiment_id}] FAILED: {e}")
    return result


SUMMARY_KEYS = ("recon/nrmse", "teacher_forced/nrmse", "recursive/nrmse@1", "recursive/nrmse@10",
                "recursive/nrmse@25", "recursive/nrmse@50", "recursive/nrmse@100", "recursive/nrmse@250",
                "recursive/nrmse@500", "recursive/nrmse_mean", "recursive/divergence_time",
                "params/total", "latent_dim", "latency/step_latency_ms_mean", "flops/dynamics_step",
                "stability/instability_score", "stability/rho_max", "stability/lyapunov_max",
                "stability/frac_blowup", "stability/verdict", "uncertainty/nll", "uncertainty/coverage95",
                "uncertainty/ece", "uncertainty/std_error_corr")


def summarize(metrics: dict[str, Any]) -> dict[str, Any]:
    """Flat, registry-friendly subset: ``<split>/<key>`` for scalar keys."""
    out: dict[str, Any] = {}
    for split in ("val", "test"):
        m = metrics.get(split, {})
        for k in SUMMARY_KEYS:
            if k in m:
                out[f"{split}/{k}"] = m[k]
    for k in ("train/best_val_loss", "train/epochs_run", "train/time_s"):
        if k in metrics:
            out[k] = metrics[k]
    return out
