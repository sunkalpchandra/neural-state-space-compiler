"""High-level figure generators used by the CLI: per experiment, per compile run, per suite.

All outputs are written with stable file names (``<out_dir>/<name>.{png,pdf}``) so docs can
reference them. Individual figures that fail (e.g. a 1-D latent has no phase portrait) are
skipped with a warning; missing inputs (no checkpoint, no report) raise ``FileNotFoundError``.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from nssc.utils.io import load_json
from nssc.visualization.compiler_plots import (
    plot_compiler_decision,
    plot_family_comparison,
    plot_latent_dim_sweep,
    plot_stage_funnel,
    selected_name,
)
from nssc.visualization.latent import (
    plot_latent_trajectories,
    plot_latent_vs_true,
    plot_phase_portrait,
)
from nssc.visualization.pareto import plot_pareto
from nssc.visualization.rollout import (
    plot_horizon_curves,
    plot_one_step_vs_long_horizon,
    plot_rollout_comparison,
)
from nssc.visualization.stability import (
    plot_eigenvalue_spectrum,
    plot_norm_growth,
    plot_spectral_radius_hist,
    plot_vector_field,
)
from nssc.visualization.style import COLORS, DOUBLE_COL, family_of, save

# isort: split
import matplotlib.pyplot as plt


class _Emitter:
    """Collects saved paths; runs each figure builder in isolation."""

    def __init__(self, out_dir: Path, formats: tuple[str, ...]) -> None:
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.formats = formats
        self.paths: list[Path] = []
        self.errors: dict[str, str] = {}

    def __call__(self, name: str, builder: Callable[[], Any]) -> None:
        try:
            fig = builder()
        except Exception as e:  # noqa: BLE001 - one bad figure must not kill the batch
            self.errors[name] = f"{type(e).__name__}: {e}"
            warnings.warn(f"figure '{name}' skipped: {type(e).__name__}: {e}", stacklevel=2)
            plt.close("all")
            return
        if fig is None:
            return
        self.paths.extend(save(fig, self.out_dir / name, formats=self.formats))


# ---------------------------------------------------------------------------- experiment
def _resolve_run(experiment: str | Path, registry_path: str | Path | None
                 ) -> tuple[Path, dict[str, Any] | None]:
    """→ (output_dir containing checkpoint/, registry record or None)."""
    p = Path(str(experiment))
    if p.is_dir() and (p / "checkpoint").is_dir():
        return p, None
    if p.is_dir() and (p / "model.pt").exists():  # a checkpoint dir itself
        return p.parent, None
    from nssc.utils.experiment_registry import ExperimentRegistry

    reg = ExperimentRegistry(registry_path) if registry_path else ExperimentRegistry()
    rec = reg.get(str(experiment))
    if rec is None:
        raise FileNotFoundError(f"unknown experiment or run directory: {experiment}")
    out = rec.get("config", {}).get("output_dir")
    if out and (Path(out) / "checkpoint").is_dir():
        return Path(out), rec
    if rec.get("checkpoint") and Path(rec["checkpoint"]).is_dir():
        return Path(rec["checkpoint"]).parent, rec
    raise FileNotFoundError(f"no checkpoint found for {experiment} (status={rec.get('status')})")


def plot_training_curves(history: list[dict[str, Any]], title: str = "Training curves") -> plt.Figure:
    """Loss vs epoch from ``history.json`` records (``train/total``, ``val/total`` if present)."""
    ep = np.array([h.get("epoch", i) for i, h in enumerate(history)], float)
    fig, ax = plt.subplots(figsize=(DOUBLE_COL * 0.55, DOUBLE_COL * 0.38))
    bad = False
    for key, c, lab in (("train/total", COLORS["blue"], "train"), ("val/total", COLORS["vermillion"], "val")):
        v = np.array([h.get(key, np.nan) for h in history], float)
        if np.isfinite(v).any():
            bad |= bool((~np.isfinite(v)).any())
            ax.plot(ep, np.nan_to_num(v, nan=np.nan, posinf=np.nan), marker=".", color=c, label=lab)
    finite = [h.get("train/total") for h in history if isinstance(h.get("train/total"), (int, float))
              and math.isfinite(h.get("train/total")) and h.get("train/total") > 0]
    if finite:
        ax.set_yscale("log")
    ax.set_xlabel("epoch")
    ax.set_ylabel("total loss")
    ax.set_title(f"{title} (n_epochs={len(history)})" + (" [non-finite omitted]" if bad else ""))
    ax.legend(loc="upper right")
    return fig


def figures_for_experiment(experiment: str | Path, out_dir: str | Path,
                           registry_path: str | Path | None = None, split: str = "test",
                           traj_index: int = 0, formats: tuple[str, ...] = ("png", "pdf"),
                           n_traj: int = 8, device: str = "cpu") -> list[Path]:
    """Generate the standard per-run figure set for a registered experiment id or run dir.

    Produces (when applicable): latent_trajectories, phase_portrait, rollout_comparison,
    one_step_vs_recursive, horizon_curve, eigenvalue_spectrum, spectral_radius_hist, norm_growth,
    vector_field (d ≥ 2), latent_vs_true (if ground-truth latents exist), training_curves.
    """
    import torch

    from nssc.experiment import prepare_data
    from nssc.stability.spectral import jacobian_spectrum
    from nssc.training.checkpoint import load_checkpoint

    run_dir, rec = _resolve_run(experiment, registry_path)
    emit = _Emitter(Path(out_dir), formats)
    model, meta = load_checkpoint(run_dir / "checkpoint", map_location=device)
    model.eval()
    cfg = (rec or {}).get("config", {}) if rec else {}
    dcfg = meta.get("dataset") or cfg.get("dataset")
    if not dcfg:
        raise FileNotFoundError(f"no dataset config in checkpoint metadata or registry for {experiment}")
    label = str(meta.get("experiment_id") or (rec or {}).get("experiment_id") or run_dir.name)
    splits, _, _raw = prepare_data(dcfg, seed=None)
    ds = splits.get(split) or splits.get("val") or splits["train"]
    x_np = ds.x[: max(n_traj, traj_index + 1)]
    x = torch.from_numpy(x_np).float()
    T = x.shape[1]
    ecfg = dict(cfg.get("eval", {})) if cfg else {}
    context = int(ecfg.get("context", meta.get("context", min(20, max(2, T // 4)))))
    context = min(context, T - 2)
    d = model.latent_dim
    with torch.no_grad():
        z = model.encode(x)  # (B,T,d)
        z0 = z[:, traj_index]
        H = T - context
        x_roll, z_roll = model.rollout(x[:, :context], H)
        x_tf, _, _ = model.predict_teacher_forced(x)
        z_free = model.dynamics.rollout(z[:, 0], max(H, 2 * T))
        norms = z_free.norm(dim=-1)
        ref_norm = float(z.norm(dim=-1).mean())
    tag = f"{label} ({split} split)"
    emit("latent_trajectories", lambda: plot_latent_trajectories(z, title=f"Latent trajectories — {tag}"))
    if d >= 2:
        dims = (0, 1, 2) if d >= 3 else (0, 1)
        emit("phase_portrait", lambda: plot_phase_portrait(z[traj_index], dims=dims,
                                                           title=f"Latent phase portrait — {label}"))
    # uncertainty envelope (optional; deterministic models use perturbation ensembles)
    x_std = None
    try:
        from nssc.uncertainty.rollout import probabilistic_rollout

        pr = probabilistic_rollout(model, x[traj_index : traj_index + 1, :context], H, n_samples=16)
        x_std = pr["std"][0]
        env_label = pr["method"]
    except Exception:  # noqa: BLE001
        env_label = ""
    emit("rollout_comparison", lambda: plot_rollout_comparison(
        x[traj_index], x_roll[traj_index], context, dims=4, x_std=x_std,
        title=f"Recursive rollout — {tag}" + (f" [±2σ: {env_label}]" if env_label else "")))
    emit("one_step_vs_recursive", lambda: plot_one_step_vs_long_horizon(
        x[traj_index], x_tf[traj_index], x_roll[traj_index], context,
        title=f"One-step vs recursive — {tag}"))
    # horizon curve from metrics.json
    mpath = run_dir / "metrics.json"
    if mpath.exists():
        m = load_json(mpath)

        def _hc():
            curves = {}
            for sp in ("val", "test"):
                c = (m.get(sp, {}) or {}).get("curves", {}).get("recursive_nrmse")
                if c:
                    curves[f"{label} [{sp}]"] = c
            if not curves:
                raise ValueError("metrics.json has no curves.recursive_nrmse")
            return plot_horizon_curves(curves, title=f"Rollout NRMSE vs horizon — {label}")

        emit("horizon_curve", _hc)
    # stability
    with torch.enable_grad():
        spec = jacobian_spectrum(model.dynamics, z.reshape(-1, d), max_points=256)
    emit("eigenvalue_spectrum", lambda: plot_eigenvalue_spectrum(
        spec["eigvals"], title=f"Latent Jacobian eigenvalues — {label}"))
    emit("spectral_radius_hist", lambda: plot_spectral_radius_hist(
        spec["spectral_radius"], title=f"Local spectral radius — {label}"))
    emit("norm_growth", lambda: plot_norm_growth(norms, ref_norm=ref_norm,
                                                 title=f"Free-rollout latent norm — {label}"))
    if d >= 2:
        emit("vector_field", lambda: plot_vector_field(model.dynamics, z, dims=(0, 1),
                                                       trajectory=z[traj_index],
                                                       title=f"Latent map — {label}"))
    if ds.z_true is not None:
        zt = ds.z_true[: z.shape[0]]
        emit("latent_vs_true", lambda: plot_latent_vs_true(
            z, zt, title=f"Latent vs ground truth — {label}"))
    hpath = run_dir / "history.json"
    if hpath.exists():
        hist = load_json(hpath).get("history", [])
        if hist:
            emit("training_curves", lambda: plot_training_curves(hist, title=f"Training — {label}"))
    del z0
    return emit.paths


# ------------------------------------------------------------------------------- compile
def _fam(name: str) -> str:
    return family_of(name)


def _pareto_points_from_ranking(ranking: list[dict[str, Any]], rollout_key: str, sel: str
                                ) -> list[dict[str, Any]]:
    pts = []
    for r in ranking:
        a = r.get("agg", {}) or {}
        name = str(r.get("name") or r.get("candidate_id"))
        pts.append({"name": name, "params": a.get("val/params/total"), "error": a.get(rollout_key),
                    "family": _fam(name), "is_selected": name == sel})
    return pts


def figures_for_compile(compile_dir: str | Path, out_dir: str | Path,
                        formats: tuple[str, ...] = ("png", "pdf"),
                        include_selected: bool = True) -> list[Path]:
    """Compiler figure set: compiler_decision, stage_funnel, pareto (final ranking), latent_dim_sweep
    and family_comparison (screen stage = all candidates), plus the experiment set for the
    selected checkpoint under ``out_dir/selected/``."""
    compile_dir = Path(compile_dir)
    rpath = compile_dir / "compile_report.json"
    if not rpath.exists():
        raise FileNotFoundError(f"no compile_report.json in {compile_dir}")
    report = load_json(rpath)
    emit = _Emitter(Path(out_dir), formats)
    sel = selected_name(report)
    rk = report.get("rollout_key") or "val/recursive/nrmse_mean"
    horizon = rk.split("@")[-1] if "@" in rk else "mean"
    ylabel = f"NRMSE@{horizon} (recursive, val)" if horizon != "mean" else "mean NRMSE (recursive, val)"
    emit("compiler_decision", lambda: plot_compiler_decision(report))
    emit("stage_funnel", lambda: plot_stage_funnel(report))
    emit("pareto", lambda: plot_pareto(_pareto_points_from_ranking(report.get("ranking", []), rk, sel),
                                       title="Final ranking: parameters vs rollout error", ylabel=ylabel))
    # all-candidate view from the first (screen) stage of the search state
    spath = compile_dir / "search_state.json"
    stage_rank: list[dict[str, Any]] = []
    stage_name = ""
    if spath.exists():
        st = load_json(spath)
        stages = st.get("stages", {}) or {}
        order = [s.get("stage") for s in report.get("stage_summaries", [])] or list(stages)
        for sname in order:
            if sname in stages and stages[sname].get("ranking"):
                stage_name = sname
                stage_rank = stages[sname]["ranking"]
                break
    if stage_rank:
        rows_dim, rows_fam = [], []
        by_fam: dict[str, list[float]] = {}
        for r in stage_rank:
            a = r.get("agg", {}) or {}
            if a.get("n_seeds", 0) == 0:
                continue
            name = str(r.get("name") or r.get("candidate_id"))
            fam = _fam(name)
            v = a.get(rk)
            ld = a.get("val/latent_dim")
            if ld is None and "@d" in name:
                ld = name.split("@d")[-1]
            rows_dim.append({"latent_dim": ld, "value": v, "std": 0.0, "family": fam})
            if isinstance(v, (int, float)) and math.isfinite(v):
                by_fam.setdefault(fam, []).append(float(v))
        for fam, vals in by_fam.items():
            rows_fam.append({"family": fam, "value": float(np.mean(vals)),
                             "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0})
        n_seeds = (stage_rank[0].get("agg", {}) or {}).get("n_seeds", 1)
        prof = report.get("profile", {}) or {}
        intrinsic = prof.get("intrinsic_dim_estimate") or prof.get("estimated_intrinsic_dim")
        emit("latent_dim_sweep", lambda: plot_latent_dim_sweep(
            rows_dim, ylabel=ylabel, intrinsic_dim=int(intrinsic) if intrinsic else None,
            title=f"Error vs latent dim ({stage_name} stage, {len(rows_dim)} candidates, "
                  f"{n_seeds} seed(s))"))
        emit("family_comparison", lambda: plot_family_comparison(
            rows_fam, ylabel=ylabel, logy=True, title=f"Dynamics family comparison ({stage_name} stage; "
                                           "mean ± std over candidates)"))
    paths = list(emit.paths)
    if include_selected and report.get("checkpoint"):
        ck = Path(report["checkpoint"])
        if ck.is_dir():
            try:
                paths += figures_for_experiment(ck.parent, Path(out_dir) / "selected", formats=formats)
            except Exception as e:  # noqa: BLE001
                warnings.warn(f"selected-model figures skipped: {type(e).__name__}: {e}", stacklevel=2)
    return paths


# --------------------------------------------------------------------------------- suite
def _run_dir_of(rec: dict[str, Any]) -> Path | None:
    out = (rec.get("config", {}) or {}).get("output_dir")
    if out and Path(out).exists():
        return Path(out)
    ck = rec.get("checkpoint")
    if ck and Path(ck).exists():
        return Path(ck).parent
    return None


def _curve_of(rec: dict[str, Any], split: str = "test", key: str = "recursive_nrmse") -> list[float] | None:
    rd = _run_dir_of(rec)
    if rd is None or not (rd / "metrics.json").exists():
        return None
    m = load_json(rd / "metrics.json")
    c = (m.get(split, {}) or {}).get("curves", {}).get(key)
    return list(c) if c else None


def figures_for_suite(registry_path: str | Path, suite_name: str | None, out_dir: str | Path,
                      metric: str = "test/recursive/nrmse@50", formats: tuple[str, ...] = ("png", "pdf"),
                      curve_split: str = "test") -> list[Path]:
    """Per-dataset horizon curves (mean ± std over seeds) and Pareto plots for a benchmark suite,
    plus a markdown summary table at ``out_dir/table.md``."""
    from nssc.evaluation.aggregate import format_markdown, group_runs, mean_std, summary_table
    from nssc.utils.experiment_registry import ExperimentRegistry

    records = ExperimentRegistry(registry_path).records()
    groups = group_runs(records, suite=suite_name)
    if not groups:
        raise FileNotFoundError(f"no completed runs for suite={suite_name!r} in {registry_path}")
    out_dir = Path(out_dir)
    emit = _Emitter(out_dir, formats)
    datasets = sorted({ds for ds, _ in groups})
    for ds in datasets:
        models = {m: recs for (d, m), recs in groups.items() if d == ds}
        curves: dict[str, Any] = {}
        pts = []
        for m, recs in sorted(models.items()):
            cs = [c for c in (_curve_of(r, curve_split) for r in recs) if c]
            if cs:
                L = min(len(c) for c in cs)
                arr = np.array([c[:L] for c in cs], float)
                curves[m] = (arr.mean(0), arr.std(0, ddof=1) if len(cs) > 1 else np.zeros(L))
            mu, sd, n = mean_std([r.get("metrics", {}).get(metric, float("nan")) for r in recs])
            params = [r.get("param_count") for r in recs if r.get("param_count") is not None]
            pts.append({"name": m, "params": float(np.mean(params)) if params else float("nan"),
                        "error": mu, "error_std": sd, "family": _fam(m), "n_seeds": n})
        n_seeds = max((len(r) for r in models.values()), default=0)
        if curves:
            emit(f"horizon_curve_{ds}", lambda c=curves, d=ds, n=n_seeds: plot_horizon_curves(
                c, title=f"{d}: NRMSE vs horizon ({curve_split}, mean ± std, n={n} seeds)",
                mode_label="recursive"))
        emit(f"pareto_{ds}", lambda p=pts, d=ds: plot_pareto(
            p, title=f"{d}: parameters vs {metric}", ylabel=metric))
    rows = summary_table(groups, [metric])
    (out_dir / "table.md").write_text(
        f"# Suite {suite_name or '(all)'} — {metric}\n\n" + format_markdown(rows, [metric]) + "\n")
    return emit.paths + [out_dir / "table.md"]
