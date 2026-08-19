"""Recompute metrics for finished runs from their checkpoints, with the current evaluation code.

Evaluation is deterministic given a checkpoint, so when a *metric* is added or corrected there is
no need to retrain: re-evaluate. Training semantics are a different matter — those are versioned by
``nssc.experiment.PROTOCOL_VERSION`` and require a re-run.

Each re-evaluated run gets a fresh appended registry row (the ledger is append-only) whose metrics
carry ``eval/reevaluated_at`` and ``eval/eval_protocol``; ``metrics.json`` next to the checkpoint is
rewritten. Runs whose checkpoint is missing (weights are gitignored) are reported as skipped.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from nssc.evaluation.evaluator import EvalConfig, evaluate_model
from nssc.utils.experiment_registry import ExperimentRecord, ExperimentRegistry
from nssc.utils.io import save_json

EVAL_PROTOCOL = 2
"""Bump when the *metric set or its definition* changes (not when training changes)."""


def _eval_cfg(cfg: dict[str, Any]) -> EvalConfig:
    from nssc.experiment import _dc

    ecfg, _ = _dc(EvalConfig, dict(cfg.get("eval", {})))
    return ecfg


def reevaluate_record(rec: dict[str, Any], registry: ExperimentRegistry,
                      device: torch.device | None = None, split: str = "test",
                      write: bool = True) -> dict[str, Any]:
    """Re-evaluate one completed run. Returns ``{status, experiment_id, metrics|reason}``."""
    from nssc.experiment import prepare_data, summarize

    ckpt = rec.get("checkpoint")
    if not ckpt or not (Path(ckpt) / "model.pt").exists():
        return {"status": "skipped", "experiment_id": rec["experiment_id"], "reason": "no checkpoint"}
    cfg = rec["config"]
    is_baseline = str(rec.get("model", "")).startswith("baseline:")
    splits, _, raw = prepare_data(dict(cfg["dataset"]))
    ecfg = _eval_cfg(cfg)
    sigma = np.ones(raw.obs_dim)
    metrics: dict[str, Any] = {}
    if is_baseline:
        from nssc.baselines.evaluate import evaluate_forecaster
        from nssc.baselines.run import load_forecaster_checkpoint

        model, _ = load_forecaster_checkpoint(ckpt)
        for name in ("val", "test"):
            if name in splits:
                metrics[name] = evaluate_forecaster(
                    model, torch.from_numpy(splits[name].x), context=ecfg.context,
                    horizons=tuple(ecfg.horizons), sigma=sigma, device=device)
    else:
        from nssc.training import load_checkpoint

        model, _ = load_checkpoint(ckpt)
        for name in ("val", "test"):
            if name in splits:
                metrics[name] = evaluate_model(model, torch.from_numpy(splits[name].x), ecfg,
                                               sigma=sigma, device=device,
                                               dt=float(cfg["dataset"].get("dt") or 1.0))
    summary = summarize(metrics)
    summary["eval/reevaluated_at"] = time.time()
    summary["eval/eval_protocol"] = EVAL_PROTOCOL
    if write:
        save_json(metrics, Path(ckpt).parent / "metrics.json")
        row = ExperimentRecord(**{k: v for k, v in rec.items() if k in ExperimentRecord.__dataclass_fields__})
        registry.update(row, metrics=summary, notes=(row.notes + "\nre-evaluated").strip())
    return {"status": "ok", "experiment_id": rec["experiment_id"], "metrics": summary}


def reevaluate_suite(suite: str | None = None, tag: str | None = None,
                     registry_path: str = "results/registry.jsonl", device: torch.device | None = None,
                     limit: int | None = None, log=print) -> dict[str, int]:
    reg = ExperimentRegistry(registry_path)
    recs = [r for r in reg.records() if r["status"] == "completed"]
    if suite:
        recs = [r for r in recs if f"suite:{suite}" in r.get("tags", [])]
    if tag:
        recs = [r for r in recs if tag in r.get("tags", [])]
    if limit:
        recs = recs[:limit]
    counts = {"ok": 0, "skipped": 0, "failed": 0}
    for i, r in enumerate(recs, 1):
        try:
            out = reevaluate_record(r, reg, device=device)
        except Exception as e:  # noqa: BLE001 — a bad row must not stop the sweep
            out = {"status": "failed", "experiment_id": r["experiment_id"], "reason": str(e)}
        counts[out["status"]] = counts.get(out["status"], 0) + 1
        if log:
            log(f"[{i}/{len(recs)}] {r['experiment_id']} {r['model']} → {out['status']}"
                + (f" ({out.get('reason')})" if out["status"] != "ok" else ""))
    return counts
