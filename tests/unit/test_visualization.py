"""Smoke tests for every figure function (synthetic arrays) + end-to-end figure generation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from nssc.utils.config import load_config
from nssc.utils.experiment_registry import ExperimentRegistry
from nssc.visualization import (
    align_latents,
    figures_for_compile,
    figures_for_experiment,
    figures_for_suite,
    model_color,
    plot_compiler_decision,
    plot_eigenvalue_spectrum,
    plot_family_comparison,
    plot_horizon_curves,
    plot_latent_dim_sweep,
    plot_latent_trajectories,
    plot_latent_vs_true,
    plot_norm_growth,
    plot_one_step_vs_long_horizon,
    plot_pareto,
    plot_phase_portrait,
    plot_rollout_comparison,
    plot_spectral_radius_hist,
    plot_stage_funnel,
    plot_training_curves,
    plot_vector_field,
    save,
    use_style,
    visualize_experiment,
)

ROOT = Path(__file__).resolve().parents[2]
MIN_BYTES = 1024


def _check(paths, n_expected: int = 2):
    assert len(paths) == n_expected
    for p in paths:
        assert p.exists() and p.stat().st_size > MIN_BYTES, p


def _write(fig, tmp_path, name):
    paths = save(fig, tmp_path / name)
    _check(paths)
    assert {p.suffix for p in paths} == {".png", ".pdf"}
    return paths


@pytest.fixture
def z_np():
    rng = np.random.default_rng(0)
    t = np.linspace(0, 6 * np.pi, 120)
    z = np.stack([np.sin(t), np.cos(t), 0.3 * t / t.max()], -1)  # (T,3)
    return z + 0.02 * rng.standard_normal(z.shape)


# ------------------------------------------------------------------------------- style
def test_style_and_colors(tmp_path):
    with use_style():
        import matplotlib

        assert matplotlib.get_backend().lower() == "agg"
        assert matplotlib.rcParams["savefig.dpi"] == 300
    assert model_color("mlp+residual_mlp@d4") == model_color("pca+residual_mlp@d2")
    assert model_color("mlp+linear@d4") != model_color("mlp+residual_mlp@d4")
    assert model_color("baseline:gru") == model_color("baseline:gru/direct")
    assert model_color("something_unknown") == model_color("something_unknown")
    fig = plot_latent_trajectories(np.zeros((10, 1)))
    a = save(fig, tmp_path / "x.png", formats=("png",))
    assert len(a) == 1 and a[0].suffix == ".png"


# ------------------------------------------------------------------------------ latent
def test_latent_trajectories_numpy_torch_batched(tmp_path, z_np):
    _write(plot_latent_trajectories(z_np, title="z"), tmp_path, "latent_trajectories")
    zb = torch.tensor(np.stack([z_np, z_np * 0.5]))  # (B,T,d)
    zb[0, 3, 1] = float("nan")
    fig = plot_latent_trajectories(zb)
    assert "non-finite" in fig.axes[0].get_title()
    _write(fig, tmp_path, "latent_trajectories_batched")


def test_phase_portrait_2d_3d(tmp_path, z_np):
    _write(plot_phase_portrait(z_np, dims=(0, 1)), tmp_path, "phase_portrait_2d")
    _write(plot_phase_portrait(torch.tensor(z_np), dims=(0, 1, 2), color_by_time=False),
           tmp_path, "phase_portrait_3d")
    with pytest.raises(ValueError):
        plot_phase_portrait(z_np, dims=(0, 5))


def test_latent_vs_true_alignment(tmp_path, z_np):
    W = np.array([[1.0, 0.5], [-0.3, 2.0], [0.1, 0.1]])
    z_true = z_np @ W + np.array([0.2, -1.0])
    al = align_latents(z_np, z_true)
    assert al["r2"] > 0.999 and al["z_aligned"].shape == z_true.shape
    fig = plot_latent_vs_true(z_np, torch.tensor(z_true), title="align")
    assert "R^2" in fig.axes[0].get_title()
    _write(fig, tmp_path, "latent_vs_true")
    zt_bad = z_true.copy()
    zt_bad[0, 0] = np.inf
    _write(plot_latent_vs_true(z_np, zt_bad), tmp_path, "latent_vs_true_bad")


# ----------------------------------------------------------------------------- rollout
def test_rollout_comparison(tmp_path):
    T, D, ctx = 60, 6, 20
    x = np.sin(np.linspace(0, 8, T))[:, None] * np.arange(1, D + 1)
    pred = x[ctx:] + 0.1
    std = np.full_like(pred, 0.2)
    _write(plot_rollout_comparison(x, pred, ctx, dims=4, x_std=std), tmp_path, "rollout_comparison")
    pred_t = torch.tensor(pred)
    pred_t[5, 0] = float("inf")
    fig = plot_rollout_comparison(torch.tensor(x), pred_t, ctx, dims=[0, 2])
    assert len(fig.axes) == 2 and "non-finite" in fig.axes[0].get_title()
    _write(fig, tmp_path, "rollout_comparison_dims")


def test_horizon_curves_and_one_step(tmp_path):
    h = np.arange(1, 51)
    curves = {"mlp+residual_mlp@d4": 0.05 * h ** 0.8, "pca+linear@d2": (0.1 * h ** 0.5, 0.01 * h ** 0.5),
              "baseline:gru": torch.tensor(0.08 * h ** 0.7), "nan_model": np.full(50, np.nan)}
    _write(plot_horizon_curves(curves, mark_horizons=[10, 25]), tmp_path, "horizon_curve")
    _write(plot_horizon_curves(curves, horizons=h, logy=False, logx=True), tmp_path, "horizon_curve_lin")
    T, D, ctx = 40, 3, 10
    x = np.random.default_rng(1).standard_normal((T, D))
    _write(plot_one_step_vs_long_horizon(x, x[1:] + 0.05, x[ctx:] + 0.2, ctx), tmp_path,
           "one_step_vs_recursive")


# ------------------------------------------------------------------------------ pareto
def test_pareto(tmp_path):
    pts = [
        {"name": "pca+linear@d2", "params": 50, "error": 0.5, "family": "linear"},
        {"name": "mlp+residual_mlp@d4", "params": 3000, "error": 0.1, "family": "residual_mlp",
         "is_selected": True, "error_std": 0.02},
        {"name": "mlp+residual_mlp@d8", "params": 9000, "error": 0.12, "family": "residual_mlp"},
        {"name": "baseline:gru", "params": 40000, "error": 0.09, "family": "baseline:gru"},
        {"name": "broken", "params": None, "error": float("nan")},
    ]
    fig = plot_pareto(pts)
    assert "omitted" in fig.axes[0].get_title()
    _write(fig, tmp_path, "pareto")


# --------------------------------------------------------------------------- stability
def test_stability_plots(tmp_path):
    rng = np.random.default_rng(0)
    ev = 0.9 * np.exp(1j * rng.uniform(0, 2 * np.pi, (64, 4))) * rng.uniform(0.5, 1.2, (64, 4))
    ev[0, 0] = complex(np.nan, 0)
    fig = plot_eigenvalue_spectrum(ev)
    assert "non-finite" in fig.axes[0].get_title()
    _write(fig, tmp_path, "eigenvalue_spectrum")
    _write(plot_eigenvalue_spectrum(torch.tensor(np.nan_to_num(ev))), tmp_path, "eigenvalue_spectrum_t")
    _write(plot_spectral_radius_hist(np.abs(ev).max(1)), tmp_path, "spectral_radius_hist")
    norms = np.exp(0.01 * np.arange(100))[None] * rng.uniform(0.5, 1.5, (8, 1))
    norms[2, -1] = np.inf
    _write(plot_norm_growth(torch.tensor(norms), ref_norm=1.0), tmp_path, "norm_growth")
    _write(plot_norm_growth(norms[0]), tmp_path, "norm_growth_single")


def test_vector_field_callable_and_dynamics(tmp_path):
    A = np.array([[0.99, -0.1], [0.1, 0.99]])
    z = np.random.default_rng(0).standard_normal((200, 2))
    traj = np.stack([np.linalg.matrix_power(A, k) @ np.array([1.0, 0.0]) for k in range(50)])
    fig = plot_vector_field(lambda x: x @ A.T, z, trajectory=traj)
    _write(fig, tmp_path, "vector_field_np")
    from nssc.dynamics.linear import LinearDynamics

    dyn = LinearDynamics(3)
    z3 = torch.randn(4, 30, 3)
    _write(plot_vector_field(dyn, z3, dims=(0, 2), grid=12, stream=False), tmp_path, "vector_field_dyn")
    with pytest.raises(ValueError):
        plot_vector_field(dyn, np.zeros((5, 1)))


# ---------------------------------------------------------------------------- compiler
def _fake_report():
    def row(rank, name, score, terms, params, err, ld):
        return {"rank": rank, "candidate_id": name + "-abc", "name": name, "score": score,
                "terms": terms, "agg": {"n_seeds": 1, "val/params/total": params,
                                        "val/recursive/nrmse@25": err, "val/latent_dim": ld}}

    ranking = [
        row(1, "pca+linear@d2", 0.3, {"recon": 0.1, "one_step": 0.0, "rollout": 0.0, "complexity": 0.0,
                                     "instability": 0.1, "blowup": 0.0}, 60, 0.05, 2),
        row(2, "mlp+residual_mlp@d4", 1.2, {"recon": 0.0, "one_step": 0.2, "rollout": 0.4,
                                           "complexity": 1.5, "instability": 0.0, "blowup": 0.0},
            5000, 0.08, 4),
        row(3, "mlp+linear@d4", 7.0, {"recon": 0.5, "one_step": 0.5, "rollout": float("nan"),
                                     "complexity": 1.0, "instability": 0.3, "blowup": 0.5}, 800,
            float("nan"), 4),
    ]
    return {"selected": {"latent_dim": 2, "encoder": "pca", "dynamics": "linear", "decoder": "pca"},
            "selected_metrics": ranking[0]["agg"], "ranking": ranking,
            "stage_summaries": [{"stage": "screen", "n_candidates": 12, "n_survivors": 3},
                                {"stage": "final", "n_candidates": 3, "n_survivors": 3}],
            "profile": {}, "weights": {"reconstruction": 1.0, "one_step": 1.0, "rollout": 2.0,
                                       "complexity": 0.1, "stability": 1.0, "blowup_penalty": 10.0,
                                       "criterion": "multi_objective"},
            "n_runs": 15, "n_failed": 1, "rollout_key": "val/recursive/nrmse@25"}


def test_compiler_plots(tmp_path):
    rep = _fake_report()
    _write(plot_compiler_decision(rep), tmp_path, "compiler_decision")
    rep2 = dict(rep, weights=dict(rep["weights"], criterion="val_mse"))
    _write(plot_compiler_decision(rep2), tmp_path, "compiler_decision_valmse")
    _write(plot_stage_funnel(rep), tmp_path, "stage_funnel")
    _write(plot_compiler_decision({"ranking": [], "selected": {}}), tmp_path, "compiler_decision_empty")
    rows = [{"latent_dim": d, "value": 0.1 * d ** -0.5 * (1 if f == "linear" else 1.5), "std": 0.01,
             "family": f} for d in (2, 4, 8, 16) for f in ("linear", "residual_mlp")]
    rows.append({"latent_dim": 4, "value": float("nan"), "family": "linear"})
    fig = plot_latent_dim_sweep(rows, intrinsic_dim=3)
    assert "omitted" in fig.axes[0].get_title()
    _write(fig, tmp_path, "latent_dim_sweep")
    _write(plot_family_comparison([{"family": "linear", "value": 0.1, "std": 0.02},
                                   {"family": "residual_mlp", "value": 0.2, "std": 0.05},
                                   {"family": "koopman", "value": float("inf")}]),
           tmp_path, "family_comparison")
    _write(plot_training_curves([{"epoch": i, "train/total": 1.0 / (i + 1), "val/total": 1.2 / (i + 1)}
                                 for i in range(5)]), tmp_path, "training_curves")


# ------------------------------------------------------------------------ end-to-end
def _cfg(tmp_path, **over):
    cfg = load_config(ROOT / "configs/experiments/smoke.yaml")
    cfg["dataset"]["_file"] = str(ROOT / cfg["dataset"]["_file"])
    cfg["output_dir"] = str(tmp_path / "run")
    for k, v in over.items():
        cfg.set_path(k, v)
    return cfg


def test_figures_for_experiment_real_run(tmp_path):
    from nssc.experiment import run_experiment

    reg = ExperimentRegistry(tmp_path / "registry.jsonl")
    res = run_experiment(_cfg(tmp_path), registry=reg, device=torch.device("cpu"), log=None)
    assert res["status"] == "completed", res.get("traceback")
    out = tmp_path / "figs"
    paths = figures_for_experiment(res["experiment_id"], out, registry_path=reg.path)
    pngs = [p for p in paths if p.suffix == ".png"]
    assert len(pngs) >= 6, [p.name for p in pngs]
    names = {p.stem for p in pngs}
    for n in ("latent_trajectories", "phase_portrait", "rollout_comparison", "horizon_curve",
              "eigenvalue_spectrum", "norm_growth", "vector_field", "training_curves"):
        assert n in names, n
    for p in pngs:
        assert p.stat().st_size > MIN_BYTES
    # via run directory + CLI hook
    paths2 = visualize_experiment(experiment=res["output_dir"], output=str(tmp_path / "cli"))
    assert any(p.name == "rollout_comparison.png" for p in paths2)
    with pytest.raises(ValueError):
        visualize_experiment()
    with pytest.raises(FileNotFoundError):
        figures_for_experiment("EXP-9999", out, registry_path=reg.path)


def test_figures_for_suite_from_registry(tmp_path):
    """Two 'suite' runs (tagged like nssc.search.runner) → per-dataset horizon curve + pareto + table."""
    from nssc.experiment import run_experiment

    reg = ExperimentRegistry(tmp_path / "registry.jsonl")
    for seed in (0, 1):
        cfg = _cfg(tmp_path / f"s{seed}", seed=seed, tags=["suite:unit", "ds:harmonic", "m:mlp_res"])
        cfg["training"]["epochs"] = 1
        r = run_experiment(cfg, registry=reg, device=torch.device("cpu"), log=None)
        assert r["status"] == "completed", r.get("traceback")
    out = tmp_path / "suite"
    paths = figures_for_suite(reg.path, "unit", out, metric="test/recursive/nrmse@10")
    names = {p.name for p in paths}
    assert "horizon_curve_harmonic.png" in names and "pareto_harmonic.png" in names
    assert (out / "table.md").exists() and "mlp_res" in (out / "table.md").read_text()
    with pytest.raises(FileNotFoundError):
        figures_for_suite(reg.path, "does_not_exist", out)


@pytest.mark.slow
def test_figures_for_compile_tiny(tmp_path):
    from nssc.compiler import StateSpaceCompiler

    cfg = load_config(ROOT / "configs/compiler/tiny.yaml")
    cfg["output_dir"] = str(tmp_path / "compile")
    reg = ExperimentRegistry(tmp_path / "reg.jsonl")
    cm = StateSpaceCompiler(cfg, device=torch.device("cpu"), registry=reg, log=None).run()
    out = tmp_path / "figs"
    paths = figures_for_compile(cm.output_dir, out)
    names = {p.name for p in paths}
    for n in ("compiler_decision.png", "stage_funnel.png", "pareto.png", "latent_dim_sweep.png",
              "family_comparison.png"):
        assert n in names, n
    assert any(p.parent.name == "selected" and p.name == "rollout_comparison.png" for p in paths)
    for p in paths:
        assert p.stat().st_size > MIN_BYTES
    paths2 = visualize_experiment(compile_dir=cm.output_dir, output=str(tmp_path / "cli"))
    assert any(p.name == "compiler_decision.png" for p in paths2)
