"""Benchmark suite runner.

A suite YAML::

    name: synthetic_core
    datasets:                       # each entry is a run-config dataset block (or {_file: ...})
      lorenz63: {_file: configs/datasets/lorenz63.yaml}
    windows: {...}  training: {...}  eval: {...}     # shared base
    seeds: [0, 1, 2, 3, 4]
    models:                          # latent state-space models (nssc.experiment)
      mlp_resmlp_d8: {latent_dim: 8, encoder: mlp, dynamics: residual_mlp}
    baselines:                       # sequence forecasters (nssc.baselines)
      gru_medium: {baseline: gru, size: medium}
    per_dataset: {lorenz63: {models: {...}, training: {...}}}   # optional overrides
    output_dir: results/raw/benchmarks/synthetic_core

Runs already present in the registry with the same config hash + seed and
status ``completed`` are skipped, so a suite is resumable and idempotent.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from nssc.experiment import resolve_dataset_cfg, run_experiment
from nssc.utils.config import deep_merge, load_config
from nssc.utils.experiment_registry import ExperimentRegistry
from nssc.utils.hashing import stable_hash
from nssc.utils.io import save_json


def _hash_for(cfg: dict[str, Any]) -> str:
    return stable_hash({k: v for k, v in cfg.items() if k not in ("output_dir", "tags")})


def expand_suite(suite: dict[str, Any]) -> list[dict[str, Any]]:
    """Materialise the list of run configs (with ``_kind`` = latent|baseline)."""
    runs: list[dict[str, Any]] = []
    base = {k: suite[k] for k in ("windows", "training", "eval") if k in suite}
    seeds = [int(s) for s in suite.get("seeds", [0])]
    out_root = Path(suite.get("output_dir", f"results/raw/benchmarks/{suite.get('name', 'suite')}"))
    for dname, dcfg in suite["datasets"].items():
        pd = (suite.get("per_dataset", {}) or {}).get(dname, {}) or {}
        dbase = deep_merge(base, {k: pd[k] for k in ("windows", "training", "eval") if k in pd})
        dcfg_r = resolve_dataset_cfg(dict(dcfg))
        models = deep_merge(suite.get("models", {}) or {}, pd.get("models", {}) or {})
        baselines = deep_merge(suite.get("baselines", {}) or {}, pd.get("baselines", {}) or {})
        for mname, mcfg in models.items():
            mcfg = dict(mcfg)
            m_train = mcfg.pop("_training", {}) or {}  # per-model training overrides (e.g. loss weights)
            for seed in seeds:
                cfg = deep_merge(dbase, {"dataset": dcfg_r, "model": mcfg, "seed": seed,
                                         "training": m_train})
                cfg["tags"] = [f"suite:{suite.get('name', 'suite')}", f"ds:{dname}", f"m:{mname}"]
                cfg["output_dir"] = str(out_root / dname / mname / f"seed{seed}")
                cfg["_kind"], cfg["_name"], cfg["_dataset"] = "latent", mname, dname
                runs.append(cfg)
        for bname, bcfg in baselines.items():
            for seed in seeds:
                cfg = deep_merge(dbase, {"dataset": dcfg_r, "model": dict(bcfg), "seed": seed})
                cfg["tags"] = [f"suite:{suite.get('name', 'suite')}", f"ds:{dname}", f"m:{bname}", "baseline"]
                cfg["output_dir"] = str(out_root / dname / bname / f"seed{seed}")
                cfg["_kind"], cfg["_name"], cfg["_dataset"] = "baseline", bname, dname
                runs.append(cfg)
    return runs


def run_suite(path: str | Path, overrides: list[str] | None = None, device: str | None = None,
              log=print, registry: ExperimentRegistry | None = None, dry_run: bool = False,
              only: str | None = None) -> list[dict[str, Any]]:
    suite = load_config(path, overrides or []).to_dict()
    registry = registry or ExperimentRegistry()
    dev = torch.device(device) if device else None
    runs = expand_suite(suite)
    if only:
        runs = [r for r in runs if only in r["_name"] or only in r["_dataset"]]
    results = []
    log(f"suite '{suite.get('name')}': {len(runs)} runs")
    for i, cfg in enumerate(runs):
        kind, name, ds = cfg.pop("_kind"), cfg.pop("_name"), cfg.pop("_dataset")
        h = _hash_for(cfg)
        done = [r for r in registry.find_by_hash(h, seed=cfg["seed"]) if r["status"] == "completed"]
        tag = f"[{i + 1}/{len(runs)}] {ds}/{name} seed={cfg['seed']}"
        if done:
            log(f"{tag} skip (done: {done[0]['experiment_id']})")
            results.append({"experiment_id": done[0]["experiment_id"], "status": "completed",
                            "summary": done[0]["metrics"], "dataset": ds, "name": name,
                            "seed": cfg["seed"], "kind": kind, "cached": True})
            continue
        if dry_run:
            log(f"{tag} (dry run)")
            continue
        if kind == "baseline":
            from nssc.baselines.run import run_baseline_experiment

            res = run_baseline_experiment(cfg, registry=registry, device=dev, log=None)
        else:
            res = run_experiment(cfg, registry=registry, device=dev, log=None)
        s = res.get("summary", {})
        log(f"{tag} {res['status']} {res.get('experiment_id')} "
            f"test_roll={s.get('test/recursive/nrmse_mean', float('nan')):.4f} "
            f"params={s.get('test/params/total')}")
        res.update({"dataset": ds, "name": name, "seed": cfg["seed"], "kind": kind})
        results.append(res)
    out_root = Path(suite.get("output_dir", f"results/raw/benchmarks/{suite.get('name', 'suite')}"))
    save_json([{k: v for k, v in r.items() if k not in ("metrics", "traceback")} for r in results],
              out_root / "suite_results.json")
    return results
