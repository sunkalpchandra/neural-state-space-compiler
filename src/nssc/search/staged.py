"""Staged, resumable search over candidates.

    coarse screening → discard clearly inferior → fine evaluation → long-horizon
    validation (more seeds) → complexity/stability analysis → final compilation

Each stage is a dict in the compiler config::

    stages:
      - {name: screen, epochs: 20,  seeds: [0],       keep_top: 8, keep_frac: 0.5}
      - {name: fine,   epochs: 100, seeds: [0],       keep_top: 3}
      - {name: final,  epochs: 200, seeds: [0, 1, 2]}

Every candidate run goes through :func:`nssc.experiment.run_experiment`, so it is
registered in the experiment registry and checkpointed like any other run.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import torch

from nssc.compiler.scorer import MultiObjectiveScorer, ScoreWeights
from nssc.experiment import run_config_hash, run_experiment
from nssc.search.space import CandidateSpec
from nssc.search.state import SearchState
from nssc.utils.config import deep_merge
from nssc.utils.experiment_registry import ExperimentRegistry


class StagedSearch:
    def __init__(self, base_run_cfg: dict[str, Any], stages: list[dict[str, Any]],
                 weights: ScoreWeights, output_dir: str | Path,
                 registry: ExperimentRegistry | None = None, device: torch.device | None = None,
                 log=print, reuse_registry: bool = True) -> None:
        self.base = base_run_cfg
        self.stages = stages
        self.scorer = MultiObjectiveScorer(weights)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.state = SearchState(self.output_dir / "search_state.json")
        self.registry = registry or ExperimentRegistry()
        self.device = device
        self.reuse_registry = reuse_registry
        self.log = log or (lambda *_: None)

    # ------------------------------------------------------------------ util
    def run_cfg_for(self, cand: CandidateSpec, stage: dict[str, Any], seed: int) -> dict[str, Any]:
        cfg = deep_merge(self.base, {"model": cand.model_config(), "seed": seed})
        train_over = {k: v for k, v in stage.items() if k in ("epochs", "lr", "rollout_horizon",
                                                                "early_stopping_patience",
                                                                "max_batches_per_epoch", "loss")}
        cfg = deep_merge(cfg, {"training": train_over})
        cfg = deep_merge(cfg, {"training": dict(cand.training_overrides)})
        if "eval" in stage:
            cfg = deep_merge(cfg, {"eval": stage["eval"]})
        cfg["tags"] = sorted(set(cfg.get("tags", [])) | {f"stage:{stage['name']}", "compiler",
                                                          f"cand:{cand.id}"} | set(cand.tags))
        cfg["output_dir"] = str(self.output_dir / "runs" / stage["name"] / cand.id / f"seed{seed}")
        return cfg

    def _run_one(self, cand: CandidateSpec, stage: dict[str, Any], seed: int) -> dict[str, Any]:
        st = stage["name"]
        cached = self.state.get(st, cand.id, seed)
        if cached and cached.get("status") in ("completed", "failed"):
            return cached
        cfg = self.run_cfg_for(cand, stage, seed)
        res = None
        if self.reuse_registry:
            h = run_config_hash(cfg)  # must match exactly what run_experiment registers (F-004/F-008)
            prior = [r for r in self.registry.find_by_hash(h, seed=seed)
                     if r["status"] == "completed" and r.get("checkpoint")
                     and Path(r["checkpoint"]).exists()]
            if prior:
                r0 = prior[-1]
                res = {"experiment_id": r0["experiment_id"], "config_hash": h, "model": r0["model"],
                       "seed": seed, "output_dir": str(Path(r0["checkpoint"]).parent),
                       "checkpoint": r0["checkpoint"], "status": "completed",
                       "summary": r0["metrics"], "reused": True}
        if res is None:
            res = run_experiment(cfg, registry=self.registry, device=self.device, log=None,
                                 save_ckpt=stage.get("save_ckpt", True))
        slim = {k: v for k, v in res.items() if k != "metrics"}
        slim["candidate"] = cand.to_dict()
        slim["stage"] = st
        self.state.put(st, cand.id, seed, slim)
        return slim

    # ----------------------------------------------------------------- stages
    def run_stage(self, stage: dict[str, Any], candidates: list[CandidateSpec]
                  ) -> tuple[list[dict[str, Any]], list[CandidateSpec]]:
        st = stage["name"]
        seeds = [int(s) for s in stage.get("seeds", [0])]
        per_cand: dict[str, list[dict[str, Any]]] = {}
        total = len(candidates) * len(seeds)
        done = 0
        for cand in candidates:
            for seed in seeds:
                res = self._run_one(cand, stage, seed)
                per_cand.setdefault(cand.id, []).append(res)
                done += 1
                s = res.get("summary", {})
                self.log(f"[{st} {done}/{total}] {cand.name} seed={seed} {res['status']}"
                         + (f" val_roll={s.get('val/recursive/nrmse_mean', float('nan')):.3f}"
                            f" params={s.get('val/params/total')}" if res["status"] == "completed" else ""))
        rows = self.scorer.rank(per_cand)
        by_id = {c.id: c for c in candidates}
        survivors = self._prune(rows, stage)
        info = {"n_candidates": len(candidates), "seeds": seeds,
                "ranking": [{"candidate_id": r["candidate_id"], "name": by_id[r["candidate_id"]].name,
                             "rank": r["rank"], "score": r["score"], "terms": r["terms"],
                             "agg": r["agg"]} for r in rows],
                "survivors": [c.id for c in survivors]}
        self.state.set_stage(st, info)
        return rows, [by_id[c.id] for c in survivors]

    def _prune(self, rows: list[dict[str, Any]], stage: dict[str, Any]) -> list[CandidateSpec]:
        ok = [r for r in rows if math.isfinite(r["score"])]
        keep = len(ok)
        if "keep_frac" in stage:
            keep = min(keep, max(1, math.ceil(len(ok) * float(stage["keep_frac"]))))
        if "keep_top" in stage:
            keep = min(keep, int(stage["keep_top"]))
        # never discard within `tolerance` of the best score (avoid noise-driven pruning)
        tol = float(stage.get("score_tolerance", 0.0))
        if tol > 0 and ok:
            best = ok[0]["score"]
            keep = max(keep, sum(1 for r in ok if r["score"] <= best + tol))
        kept = ok[:keep]
        return [CandidateSpec.from_dict(self._cand_dict(r["candidate_id"])) for r in kept]

    def _cand_dict(self, cid: str) -> dict[str, Any]:
        for k, v in self.state.namespaced_runs().items():
            if k.split("|")[2] == cid and "candidate" in v:
                return v["candidate"]
        raise KeyError(cid)

    def run(self, candidates: list[CandidateSpec]) -> dict[str, Any]:
        survivors = candidates
        history = []
        for stage in self.stages:
            self.log(f"── stage '{stage['name']}': {len(survivors)} candidates × "
                     f"{len(stage.get('seeds', [0]))} seeds")
            rows, survivors = self.run_stage(stage, survivors)
            history.append({"stage": stage["name"], "rows": rows,
                            "survivors": [c.id for c in survivors]})
            if not survivors:
                self.log("no surviving candidates!")
                break
        return {"history": history, "final": survivors, "state_path": str(self.state.path)}
