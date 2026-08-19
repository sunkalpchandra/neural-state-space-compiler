"""StateSpaceCompiler: profile → candidates → staged search → select → report.

Config schema (see configs/compiler/default.yaml)::

    dataset: {...}                     # run-config dataset block
    windows: {...}   training: {...}   eval: {...}      # base run config (shared by candidates)
    candidates: {latent_dims: auto|[...], encoders: [...], dynamics: [...], hidden_dims: [...]}
    stages: [{name, epochs, seeds, keep_top, keep_frac, score_tolerance, eval: {...}}, ...]
    objective: {reconstruction, one_step, rollout, complexity, stability, criterion, ...}
    output_dir: results/compile/<name>
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from nssc.compiler.profiler import profile_dataset
from nssc.compiler.report import CompileReport, build_reasons
from nssc.compiler.scorer import ScoreWeights
from nssc.data.builder import build_dataset
from nssc.experiment import resolve_dataset_cfg
from nssc.models.latent_model import LatentModel
from nssc.search.space import CandidateSpec, generate_candidates
from nssc.search.staged import StagedSearch
from nssc.training.checkpoint import load_checkpoint
from nssc.utils.config import Config, save_yaml
from nssc.utils.experiment_registry import ExperimentRegistry
from nssc.utils.io import save_json


@dataclass
class CompiledModel:
    model: LatentModel | None
    spec: CandidateSpec
    report: CompileReport
    checkpoint: str | None
    output_dir: str

    def rollout(self, x_context: torch.Tensor, horizon: int):
        assert self.model is not None
        return self.model.rollout(x_context, horizon)


class StateSpaceCompiler:
    def __init__(self, cfg: dict[str, Any] | Config, device: torch.device | None = None,
                 registry: ExperimentRegistry | None = None, log=print) -> None:
        self.cfg = Config(dict(cfg))
        self.device = device
        self.registry = registry or ExperimentRegistry()
        self.log = log or (lambda *_: None)
        self.output_dir = Path(self.cfg.get("output_dir", "results/compile/default"))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.profile: dict[str, Any] | None = None
        self.candidates: list[CandidateSpec] = []
        self.weights = ScoreWeights.from_config(self.cfg.get("objective", {}))

    # ---------------------------------------------------------------- stages
    def fit(self) -> dict[str, Any]:
        """Stage 1: profile the dataset (cached in output_dir/profile.json)."""
        dcfg = resolve_dataset_cfg(dict(self.cfg["dataset"]))
        self.cfg["dataset"] = dcfg
        ppath = self.output_dir / "profile.json"
        if ppath.exists():
            from nssc.utils.io import load_json

            self.profile = load_json(ppath)
        else:
            # Profile the TRAIN split only: the profile drives candidate latent dimensions, so
            # computing it on all trajectories would leak held-out data into model selection
            # (review finding R-51). Splitting is trajectory-level and identical to prepare_data.
            ds = build_dataset(dcfg)
            train = ds.split()["train"]
            prof = profile_dataset(train)
            self.profile = prof.to_dict()
            self.profile["computed_on"] = {"split": "train", "n_traj": train.n_traj,
                                           "of_total": ds.n_traj}
            save_json(self.profile, ppath)
        self.obs_dim = int(self.profile["obs_dim"])
        self.log(f"profile: D={self.obs_dim} suggested latent dims={self.profile.get('suggested_latent_dims')}"
                 f" hints={self.profile.get('recommendations')}")
        return self.profile

    def propose(self) -> list[CandidateSpec]:
        """Stage 2–4: enumerate candidate (latent dim × encoder × dynamics)."""
        if self.profile is None:
            self.fit()
        ccfg = dict(self.cfg.get("candidates", {}))
        self.candidates = generate_candidates(ccfg, self.profile, self.obs_dim)
        save_json([c.to_dict() for c in self.candidates], self.output_dir / "candidates.json")
        self.log(f"{len(self.candidates)} candidates")
        return self.candidates

    def base_run_cfg(self) -> dict[str, Any]:
        keep = ("dataset", "windows", "training", "eval")
        base = {k: dict(self.cfg[k]) for k in keep if k in self.cfg}
        base["tags"] = list(self.cfg.get("tags", []))
        return base

    def search(self) -> dict[str, Any]:
        """Stage 5–6: staged search + multi-objective scoring."""
        if not self.candidates:
            self.propose()
        stages = list(self.cfg.get("stages") or [{"name": "single", "seeds": [0]}])
        self.searcher = StagedSearch(self.base_run_cfg(), stages, self.weights, self.output_dir,
                                     registry=self.registry, device=self.device, log=self.log,
                                     reuse_registry=bool(self.cfg.get("reuse_registry", True)))
        return self.searcher.run(self.candidates)

    def compile(self, search_result: dict[str, Any] | None = None) -> CompiledModel:
        """Select the winner of the final stage, load its best-seed checkpoint, write report."""
        t0 = time.time()
        sr = search_result or self.search()
        final = sr["history"][-1]
        rows = final["rows"]
        by_id = {c.id: c for c in self.candidates}
        for r in rows:
            r["name"] = by_id[r["candidate_id"]].name if r["candidate_id"] in by_id else r["candidate_id"]
        completed = [r for r in rows if r["agg"].get("n_seeds", 0) > 0]
        if not completed:
            raise RuntimeError("compiler: no candidate completed successfully")
        top = completed[0]
        spec = by_id[top["candidate_id"]]
        # best seed run checkpoint for the winner (lowest val rollout among its seeds)
        runs = self.searcher.state.stage_results(final["stage"]).get(spec.id, [])
        rk = top["rollout_key"]
        ok = [r for r in runs if r.get("status") == "completed" and r.get("checkpoint")
              and (Path(r["checkpoint"]) / "model.pt").exists()]
        best_run = min(ok, key=lambda r: r["summary"].get(rk, float("inf"))) if ok else None
        model = None
        if best_run:
            model, _ = load_checkpoint(best_run["checkpoint"])
        runs_ns = self.searcher.state.namespaced_runs()
        n_runs = len(runs_ns)
        n_failed = sum(1 for r in runs_ns.values() if r.get("status") == "failed")
        report = CompileReport(
            selected=spec.to_dict(), selected_metrics=top["agg"], ranking=rows,
            stage_summaries=[{"stage": h["stage"], "n_candidates": len(h["rows"]),
                              "n_survivors": len(h["survivors"])} for h in sr["history"]],
            profile=self.profile or {}, weights=self.weights.__dict__,
            reasons=build_reasons(rows, spec.to_dict(), rk), n_runs=n_runs, n_failed=n_failed,
            wall_time_s=time.time() - self.searcher.state.data.get("created", t0),
            dataset=dict(self.cfg["dataset"]), checkpoint=best_run["checkpoint"] if best_run else None,
            rollout_key=rk)
        report.save(self.output_dir / "compile_report.json")
        save_yaml(self.cfg.to_dict(), self.output_dir / "compiler_config.yaml")
        if best_run:
            save_yaml({"model": spec.model_config(), "checkpoint": best_run["checkpoint"],
                       "experiment_id": best_run["experiment_id"]}, self.output_dir / "compiled_model.yaml")
        self.log(f"selected {spec.name} ({int(top['agg'].get('val/params/total', 0))} params)")
        return CompiledModel(model, spec, report, report.checkpoint, str(self.output_dir))

    def run(self, resume: bool = True) -> CompiledModel:
        if not resume:
            sp = self.output_dir / "search_state.json"
            if sp.exists():
                sp.unlink()
        self.fit()
        self.propose()
        return self.compile(self.search())

    def evaluate(self, compiled: CompiledModel, split: str = "test") -> dict[str, Any]:
        """Evaluate the compiled model on a held-out split with the compiler's eval config."""
        import numpy as np

        from nssc.evaluation import EvalConfig, evaluate_model
        from nssc.experiment import _dc, prepare_data

        splits, _, raw = prepare_data(dict(self.cfg["dataset"]))
        ecfg, _ignored = _dc(EvalConfig, dict(self.cfg.get("eval", {})))
        assert compiled.model is not None
        return evaluate_model(compiled.model, torch.from_numpy(splits[split].x), ecfg,
                              sigma=np.ones(raw.obs_dim), device=self.device)
