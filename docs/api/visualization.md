# `nssc.visualization`

### `nssc.visualization`

Script-generated figures (matplotlib, Agg backend pinned in ``style``).

### `nssc.visualization.cli_hooks`

Entry point used by ``nssc visualize``.

#### `visualize_experiment(experiment: 'str | None' = None, compile_dir: 'str | None' = None, output: 'str' = 'results/figures', registry_path: 'str | None' = None) -> 'list[Path]'`

Dispatch: ``--compile-dir`` → compiler figure set (into ``output/<compile name>/``);
``--experiment`` (id or run dir) → per-run figure set (into ``output/<experiment>/``).

### `nssc.visualization.compiler_plots`

Compiler figures: score decomposition of the final ranking, stage funnel, sweeps.

#### `plot_compiler_decision(report: 'dict[str, Any]', max_candidates: 'int' = 20, title: 'str | None' = None) -> 'plt.Figure'`

Horizontal stacked bars: score J of every final-ranking candidate decomposed into
weighted terms; the selected candidate is highlighted.

#### `plot_family_comparison(rows: 'list[dict[str, Any]]', title: 'str' = 'Model family comparison', ylabel: 'str' = 'NRMSE (recursive, val)', logy: 'bool' = False) -> 'plt.Figure'`

``rows``: [{family, value, std?}] → bar chart with error bars (one bar per family, in
ascending order of value).

#### `plot_latent_dim_sweep(rows: 'list[dict[str, Any]]', title: 'str' = 'Error vs latent dimension', ylabel: 'str' = 'NRMSE (recursive, val)', logy: 'bool' = True, intrinsic_dim: 'int | None' = None) -> 'plt.Figure'`

``rows``: [{latent_dim, value, std?, family?}] → one line per family (mean ± std per dim).

#### `plot_stage_funnel(report: 'dict[str, Any]', title: 'str' = 'Search funnel') -> 'plt.Figure'`

Candidates entering / surviving each search stage.

#### `score_contributions(row: 'dict[str, Any]', weights: 'dict[str, Any]', criterion: 'str' = 'multi_objective') -> 'dict[str, float]'`

weight × term for each scorer term (NaN terms → weight × 5 penalty, as in the scorer).

#### `selected_name(report: 'dict[str, Any]') -> 'str'`

### `nssc.visualization.figures`

High-level figure generators used by the CLI: per experiment, per compile run, per suite.

All outputs are written with stable file names (``<out_dir>/<name>.{png,pdf}``) so docs can
reference them. Individual figures that fail (e.g. a 1-D latent has no phase portrait) are
skipped with a warning; missing inputs (no checkpoint, no report) raise ``FileNotFoundError``.

#### `figures_for_compile(compile_dir: 'str | Path', out_dir: 'str | Path', formats: 'tuple[str, ...]' = ('png', 'pdf'), include_selected: 'bool' = True) -> 'list[Path]'`

Compiler figure set: compiler_decision, stage_funnel, pareto (final ranking), latent_dim_sweep
and family_comparison (screen stage = all candidates), plus the experiment set for the
selected checkpoint under ``out_dir/selected/``.

#### `figures_for_experiment(experiment: 'str | Path', out_dir: 'str | Path', registry_path: 'str | Path | None' = None, split: 'str' = 'test', traj_index: 'int' = 0, formats: 'tuple[str, ...]' = ('png', 'pdf'), n_traj: 'int' = 8, device: 'str' = 'cpu') -> 'list[Path]'`

Generate the standard per-run figure set for a registered experiment id or run dir.

Produces (when applicable): latent_trajectories, phase_portrait, rollout_comparison,
one_step_vs_recursive, horizon_curve, eigenvalue_spectrum, spectral_radius_hist, norm_growth,
vector_field (d ≥ 2), latent_vs_true (if ground-truth latents exist), training_curves.

#### `figures_for_suite(registry_path: 'str | Path', suite_name: 'str | None', out_dir: 'str | Path', metric: 'str' = 'test/recursive/nrmse@50', formats: 'tuple[str, ...]' = ('png', 'pdf'), curve_split: 'str' = 'test') -> 'list[Path]'`

Per-dataset horizon curves (mean ± std over seeds) and Pareto plots for a benchmark suite,
plus a markdown summary table at ``out_dir/table.md``.

#### `plot_training_curves(history: 'list[dict[str, Any]]', title: 'str' = 'Training curves') -> 'plt.Figure'`

Loss vs epoch from ``history.json`` records (``train/total``, ``val/total`` if present).

### `nssc.visualization.latent`

Latent-space figures: stacked trajectories, phase portraits, alignment to ground truth.

#### `align_latents(z: 'Any', z_true: 'Any') -> 'dict[str, Any]'`

Least-squares affine map ``z_true ≈ W z + b`` on flattened (·, d) arrays.

Returns ``{"z_aligned": (…, d_true), "W", "b", "r2": float, "r2_per_dim": array}``.

#### `plot_latent_trajectories(z: 'Any', title: 'str' = 'Latent trajectories', t: 'Any | None' = None, max_traj: 'int' = 8, dim_labels: 'list[str] | None' = None) -> 'plt.Figure'`

Stacked time series of z_1..z_d. ``z``: (T,d) or (B,T,d); each trajectory a faint line.

#### `plot_latent_vs_true(z: 'Any', z_true: 'Any', title: 'str' = 'Latent vs ground-truth state', max_traj: 'int' = 3) -> 'plt.Figure'`

Overlay of linearly aligned latents (``z_true ≈ W z + b``) on true latents; R² in title.

Caption warning: alignment R² is a *linear recoverability* measure, not evidence that
latent coordinates are physical variables.

#### `plot_phase_portrait(z: 'Any', dims: 'tuple[int, ...]' = (0, 1), color_by_time: 'bool' = True, title: 'str' = 'Latent phase portrait', max_traj: 'int' = 6) -> 'plt.Figure'`

2D or 3D phase portrait of latent trajectories along ``dims`` (2 or 3 indices).

Colored by time with viridis (ordinal → sequential colormap) if ``color_by_time``.

### `nssc.visualization.pareto`

Complexity–accuracy Pareto figure.

#### `plot_pareto(points: 'list[dict[str, Any]]', title: 'str' = 'Complexity vs error', xlabel: 'str' = 'parameters', ylabel: 'str' = 'NRMSE (recursive)', logy: 'bool' = True, annotate_front: 'bool' = True, max_annotations: 'int' = 12) -> 'plt.Figure'`

``points``: dicts with ``name``, ``params``, ``error``, optional ``family``, ``is_selected``,
``error_std``. Log-x parameters; Pareto front (min params, min error) drawn as a step line;
selected candidate starred; front points annotated.

### `nssc.visualization.rollout`

Rollout figures: true vs predicted observations, error vs horizon, one-step vs recursive.

#### `plot_horizon_curves(curves: 'dict[str, Any]', horizons: 'Any | None' = None, logy: 'bool' = True, logx: 'bool' = False, title: 'str' = 'Rollout error vs horizon', ylabel: 'str' = 'NRMSE', mode_label: 'str' = 'recursive', mark_horizons: 'Any | None' = None) -> 'plt.Figure'`

NRMSE vs horizon for several models. ``curves``: {name: curve} or {name: (mean, std)}.

Values are plotted per step ``1..H`` unless ``horizons`` (len H) is given.

#### `plot_one_step_vs_long_horizon(x_true: 'Any', x_tf_pred: 'Any', x_recursive_pred: 'Any', context: 'int', dim: 'int' = 0, title: 'str' = 'One-step vs recursive prediction') -> 'plt.Figure'`

Two panels: (left) teacher-forced one-step x̂_{t+1} vs truth; (right) recursive rollout
from ``context``. ``x_tf_pred``: (T-1,D) aligned to targets x_{2:T}; ``x_recursive_pred``: (H,D).

#### `plot_rollout_comparison(x_true: 'Any', x_pred: 'Any', context: 'int', dims: 'int | list[int]' = 4, x_std: 'Any | None' = None, title: 'str' = 'Recursive rollout', mode_label: 'str' = 'recursive') -> 'plt.Figure'`

True (T,D) vs predicted (H,D) for the H steps after ``context``; ±2σ envelope optional.

``dims``: number of leading dims or explicit list of dim indices (max 4 recommended).

### `nssc.visualization.stability`

Stability figures: Jacobian eigenvalue spectra, spectral radius, norm growth, vector fields.

#### `plot_eigenvalue_spectrum(eigvals: 'Any', title: 'str' = 'Latent Jacobian eigenvalues', max_points: 'int' = 20000) -> 'plt.Figure'`

Scatter of complex eigenvalues (M,d) in the complex plane with the unit circle;
points colored by local density (Gaussian-KDE-free 2-D histogram lookup).

#### `plot_norm_growth(norms: 'Any', ref_norm: 'float | None' = None, title: 'str' = 'Latent norm growth', logy: 'bool' = True, max_lines: 'int' = 32) -> 'plt.Figure'`

||ẑ_t|| along free rollouts, ``norms``: (B,H). Median and 10–90% band plus individual lines.
``ref_norm``: mean norm of encoded data (dashed reference).

#### `plot_spectral_radius_hist(rho: 'Any', title: 'str' = 'Local spectral radius', bins: 'int' = 30) -> 'plt.Figure'`

Histogram of per-point spectral radii ρ(J) with the ρ=1 stability boundary marked.

#### `plot_vector_field(dynamics: 'Any', z_samples: 'Any', dims: 'tuple[int, int]' = (0, 1), grid: 'int' = 25, trajectory: 'Any | None' = None, title: 'str' = 'Latent vector field', stream: 'bool' = True, is_displacement: 'bool' = False) -> 'plt.Figure'`

Displacement field F(z) − z of the discrete latent map on the plane spanned by ``dims``.

``dynamics``: a ``Dynamics`` module (``step``) or callable ``f(z)->F(z)`` (numpy or torch);
if ``is_displacement`` the callable already returns F(z) − z (e.g. an ODE vector field).
``z_samples``: (N,d) or (B,T,d) latents defining plot ranges; other dims held at their mean.
``trajectory``: optional (T,d) latent trajectory to overlay.

### `nssc.visualization.style`

Global matplotlib style for all nssc figures (headless, colorblind-safe, print-ready).

Import this module *before* ``matplotlib.pyplot`` anywhere in the visualization
package: it pins the Agg backend so figure generation works without a display.

#### `family_of(name: 'str') -> 'str'`

Model family key for ``name``: ``baseline:<k>`` → baseline key; ``enc+dyn@dN`` → dyn.

#### `model_color(name: 'str') -> 'str'`

Deterministic color for a model/candidate name (fixed per family; hashed fallback).

#### `save(fig: 'plt.Figure', path: 'str | Path', formats: 'Iterable[str]' = ('png', 'pdf'), close: 'bool' = True, dpi: 'int' = 300) -> 'list[Path]'`

Save ``fig`` as ``<path stem>.<fmt>`` for every format. Returns the written paths.

A suffix on ``path`` is ignored (stem is used) so ``save(fig, "a.png")`` and
``save(fig, "a")`` write the same files. PDFs get fixed metadata for byte-stable output.

#### `use_style() -> '_StyleContext'`

Apply the project style. Works both as a plain call and as a context manager.
