"""Compiler end-to-end on the tiny config, including crash-resume behaviour."""

import json
from pathlib import Path

import pytest
import torch

from nssc.compiler import CompileReport, StateSpaceCompiler
from nssc.search.state import SearchState
from nssc.utils.config import load_config
from nssc.utils.experiment_registry import ExperimentRegistry

ROOT = Path(__file__).resolve().parents[2]


def _cfg(tmp_path):
    cfg = load_config(ROOT / "configs/compiler/tiny.yaml")
    cfg["output_dir"] = str(tmp_path / "compile")
    return cfg


@pytest.mark.slow
def test_tiny_compile_selects_linear_low_dim(tmp_path):
    reg = ExperimentRegistry(tmp_path / "reg.jsonl")
    comp = StateSpaceCompiler(_cfg(tmp_path), device=torch.device("cpu"), registry=reg, log=None)
    cm = comp.run()
    rep = cm.report
    # Gate E sanity: harmonic oscillator lifted linearly → linear dynamics, d ≤ 4
    assert cm.spec.dynamics == "linear"
    assert cm.spec.latent_dim <= 4
    assert cm.model is not None and cm.checkpoint
    assert rep.reasons and rep.n_runs > 0
    md = rep.to_markdown()
    assert "Selected dynamics" in md and "Final ranking" in md
    out = Path(cm.output_dir)
    for f in ("compile_report.json", "compile_report.md", "search_state.json", "profile.json",
              "candidates.json", "compiled_model.yaml"):
        assert (out / f).exists(), f
    loaded = CompileReport.load(out / "compile_report.json")
    assert loaded.selected["latent_dim"] == cm.spec.latent_dim
    # every run in the search state exists in the registry
    st = json.loads((out / "search_state.json").read_text())
    ids = {r["experiment_id"] for r in st["runs"].values()}
    assert ids <= {r["experiment_id"] for r in reg.records()}
    # test-split evaluation via compiler API
    ev = comp.evaluate(cm, split="test")
    assert ev["recursive/nrmse@10"] < 0.5


@pytest.mark.slow
def test_compile_resume_skips_completed_runs(tmp_path):
    reg = ExperimentRegistry(tmp_path / "reg.jsonl")
    cfg = _cfg(tmp_path)
    comp = StateSpaceCompiler(cfg, device=torch.device("cpu"), registry=reg, log=None)
    comp.fit()
    comp.propose()
    # run only the first stage, then "crash"
    from nssc.search.staged import StagedSearch

    s = StagedSearch(comp.base_run_cfg(), comp.cfg["stages"], comp.weights, comp.output_dir,
                     registry=reg, device=torch.device("cpu"), log=None)
    s.run_stage(comp.cfg["stages"][0], comp.candidates)
    n_before = SearchState(comp.output_dir / "search_state.json").n_completed
    n_reg_before = len(reg.records())
    assert n_before > 0
    # resume: stage-1 runs must not be re-executed (registry gains only later-stage runs)
    comp2 = StateSpaceCompiler(cfg, device=torch.device("cpu"), registry=reg, log=None)
    cm = comp2.run(resume=True)
    n_reg_after = len(reg.records())
    n_final_runs = len([k for k in comp2.searcher.state.data["runs"] if k.startswith("final|")])
    assert n_reg_after - n_reg_before == n_final_runs
    assert cm.report.n_runs == n_before + n_final_runs
