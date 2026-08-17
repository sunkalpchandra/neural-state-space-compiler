"""Baseline run pipeline mirroring :func:`nssc.experiment.run_experiment`.

Run config::

    dataset: {...}                       # as in nssc.experiment
    model:   {baseline: gru, kwargs: {...}, mode: recursive|direct, direct_horizon: 30,
              size: small|medium|large}   # size → kwargs preset from configs/models/baselines/<key>.yaml
    training: {...}                      # BaselineTrainerConfig fields (context defaults to windows.context)
    windows: {context: 20, horizon: 30, stride: 5, batch_size: 64}
    eval:    {context: 20, horizons: [...], max_horizon, batch_size, divergence_threshold, latency}
    seed: 0
    tags: [...]
    output_dir: results/raw/<name>

Registered with model name ``baseline:<key>`` (``baseline:<key>/direct`` in direct mode).
Checkpoint: ``<output_dir>/checkpoint/{model.pt, config.json, metadata.json}``.
"""

from __future__ import annotations

import time
import traceback
from dataclasses import fields
from pathlib import Path
from typing import Any

import numpy as np
import torch

from nssc.baselines import build_baseline
from nssc.baselines.base import SequenceForecaster
from nssc.baselines.evaluate import evaluate_forecaster
from nssc.baselines.trainer import BaselineTrainer, BaselineTrainerConfig
from nssc.data.dataset import make_loaders
from nssc.experiment import prepare_data, resolve_dataset_cfg, summarize
from nssc.metrics.prediction import DEFAULT_HORIZONS
from nssc.utils.config import Config, load_yaml
from nssc.utils.env import default_device
from nssc.utils.experiment_registry import ExperimentRegistry
from nssc.utils.hashing import stable_hash
from nssc.utils.io import load_json, save_json
from nssc.utils.seeding import seed_everything

PRESET_DIR = Path(__file__).resolve().parents[3] / "configs" / "models" / "baselines"


def _dc(cls, d: dict[str, Any]):
    names = {f.name for f in fields(cls)}
    return cls(**{k: v for k, v in d.items() if k in names})


def load_preset(key: str, size: str = "medium") -> dict[str, Any]:
    """kwargs for ``key`` at ``size`` from ``configs/models/baselines/<key>.yaml`` (``sizes:`` map)."""
    cfg = load_yaml(PRESET_DIR / f"{'persistence' if key == 'mean' else key}.yaml")
    sizes = cfg.get("sizes") or {}
    if size not in sizes:
        raise KeyError(f"{key}: unknown size {size!r}; available {sorted(sizes)}")
    return dict(sizes[size] or {})


def resolve_model_cfg(mcfg: dict[str, Any], windows: dict[str, Any]) -> dict[str, Any]:
    """Normalise ``model`` config: fill kwargs from ``size``, default direct_horizon to windows.horizon."""
    m = dict(mcfg)
    key = m["baseline"]
    kw = dict(m.get("kwargs") or {})
    if "size" in m:
        kw = {**load_preset(key, m["size"]), **kw}
    m["kwargs"] = kw
    m["mode"] = m.get("mode", "recursive")
    if m["mode"] == "direct" and not m.get("direct_horizon"):
        m["direct_horizon"] = int(windows.get("horizon", 30))
    return m


def baseline_model_name(mcfg: dict[str, Any]) -> str:
    name = f"baseline:{mcfg.get('baseline', '?')}"
    return name + "/direct" if mcfg.get("mode") == "direct" else name


def save_forecaster_checkpoint(model: SequenceForecaster, mcfg: dict[str, Any], path: str | Path,
                               metadata: dict[str, Any] | None = None) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path / "model.pt")
    save_json({"baseline": mcfg["baseline"], "kwargs": mcfg.get("kwargs", {}), "mode": model.mode,
               "direct_horizon": model.direct_horizon, "obs_dim": model.obs_dim,
               "resolved": model.config()}, path / "config.json")
    save_json(metadata or {}, path / "metadata.json")
    return path


def load_forecaster_checkpoint(path: str | Path, map_location: str | torch.device = "cpu"
                               ) -> tuple[SequenceForecaster, dict[str, Any]]:
    path = Path(path)
    cfg = load_json(path / "config.json")
    model = build_baseline(cfg["baseline"], obs_dim=int(cfg["obs_dim"]), mode=cfg["mode"],
                           direct_horizon=cfg.get("direct_horizon"), **cfg.get("kwargs", {}))
    model.load_state_dict(torch.load(path / "model.pt", map_location=map_location, weights_only=True))
    model.eval()
    meta = load_json(path / "metadata.json") if (path / "metadata.json").exists() else {}
    return model, meta


def run_baseline_experiment(cfg: dict[str, Any] | Config, registry: ExperimentRegistry | None = None,
                            device: torch.device | None = None, log=print, save_ckpt: bool = True
                            ) -> dict[str, Any]:
    """Execute one baseline run. Failures are recorded (status='failed'), never raised."""
    cfg = Config(dict(cfg))
    seed = int(cfg.get("seed", 0))
    seed_everything(seed)
    device = device or (torch.device(cfg.training.device) if cfg.get("training", {}).get("device")
                        else default_device())
    dcfg = resolve_dataset_cfg(dict(cfg["dataset"]))
    cfg["dataset"] = dcfg
    w = dict(cfg.get("windows", {}))
    resolve_err: Exception | None = None
    try:
        mcfg = resolve_model_cfg(dict(cfg["model"]), w)
    except Exception as ex:  # noqa: BLE001  (recorded as a failed run below)
        resolve_err, mcfg = ex, dict(cfg["model"])
    cfg["model"] = mcfg
    chash = stable_hash({k: v for k, v in cfg.to_dict().items() if k not in ("output_dir", "tags")})
    registry = registry or ExperimentRegistry()
    mname = baseline_model_name(mcfg)
    rec = registry.register(config=cfg.to_dict(), config_hash=chash, dataset=dcfg.get("system", "?"),
                            model=mname, seed=seed, tags=list(cfg.get("tags", [])))
    out_dir = Path(cfg.get("output_dir", f"results/raw/{rec.experiment_id}"))
    out_dir.mkdir(parents=True, exist_ok=True)
    result: dict[str, Any] = {"experiment_id": rec.experiment_id, "config_hash": chash,
                              "model": mname, "seed": seed, "output_dir": str(out_dir)}
    try:
        if resolve_err is not None:
            raise resolve_err
        t0 = time.perf_counter()
        splits, stats, raw = prepare_data(dcfg, seed=None)
        context, horizon = int(w.get("context", 20)), int(w.get("horizon", 30))
        loaders = make_loaders(splits, context=context, horizon=horizon,
                               batch_size=int(w.get("batch_size", 64)), stride=int(w.get("stride", 5)))
        model = build_baseline(mcfg["baseline"], obs_dim=raw.obs_dim, mode=mcfg["mode"],
                               direct_horizon=mcfg.get("direct_horizon"), **mcfg["kwargs"])
        tdict = {"context": context, **dict(cfg.get("training", {}))}
        tcfg = _dc(BaselineTrainerConfig, tdict)
        trainer = BaselineTrainer(model, tcfg, device=device)
        fit = trainer.fit(loaders["train"], loaders.get("val"),
                          log=(lambda r: log(f"[{rec.experiment_id}] ep{r['epoch']} "
                                             f"train={r['train/total']:.4g} val={r.get('val/total', float('nan')):.4g}"))
                          if log else None)
        e = dict(cfg.get("eval", {}))
        sigma = np.ones(raw.obs_dim)  # normalised data → σ = 1 per dim (train stats)
        metrics: dict[str, Any] = {}
        for split_name in ("val", "test"):
            if split_name in splits:
                x = torch.from_numpy(splits[split_name].x)
                metrics[split_name] = evaluate_forecaster(
                    model, x, context=int(e.get("context", context)),
                    horizons=tuple(e.get("horizons", DEFAULT_HORIZONS)), sigma=sigma, device=device,
                    max_horizon=e.get("max_horizon"), batch_size=int(e.get("batch_size", 64)),
                    divergence_threshold=float(e.get("divergence_threshold", 1.0)),
                    latency=bool(e.get("latency", True)))
        metrics["train/best_val_loss"] = fit["best_val"]
        metrics["train/epochs_run"] = fit["epochs_run"]
        metrics["train/time_s"] = fit["train_time_s"]
        metrics["train/final"] = fit["history"][-1] if fit["history"] else {}
        ckpt = None
        if save_ckpt:
            ckpt = save_forecaster_checkpoint(
                model, mcfg, out_dir / "checkpoint",
                metadata={"experiment_id": rec.experiment_id, "seed": seed,
                          "norm_stats": {k: v.tolist() for k, v in stats.items()},
                          "dataset": dcfg, "metrics_summary": summarize(metrics)})
        save_json({"history": fit["history"]}, out_dir / "history.json")
        save_json(metrics, out_dir / "metrics.json")
        result.update({"metrics": metrics, "checkpoint": str(ckpt) if ckpt else None,
                       "wall_time_s": time.perf_counter() - t0, "status": "completed",
                       "summary": summarize(metrics)})
        registry.complete(rec, metrics=summarize(metrics), checkpoint=str(ckpt) if ckpt else None,
                          param_count=int(model.num_parameters()), train_time_s=fit["train_time_s"])
    except Exception as ex:  # noqa: BLE001
        tb = traceback.format_exc()
        save_json({"error": str(ex), "traceback": tb}, out_dir / "error.json")
        registry.fail(rec, f"{type(ex).__name__}: {ex}")
        result.update({"status": "failed", "error": str(ex), "traceback": tb})
        if log:
            log(f"[{rec.experiment_id}] FAILED: {ex}")
    return result
