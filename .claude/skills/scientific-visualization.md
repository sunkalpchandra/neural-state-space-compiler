# Skill: scientific-visualization

## Purpose
Rules for figures in `nssc`. All figures are produced by scripts in
`src/nssc/visualization/` (invoked via `scripts/make_figures.py`) from registry /
processed data. No hand-edited figures, no notebook screenshots in README.

## Relevant theory
- A figure encodes one comparison. Ask: what is on the x-axis, what varies by color,
  what does the reader conclude in five seconds?
- Uncertainty is always visible: mean line + shaded ±std or 95% bootstrap CI band across
  seeds 0–4; n written in the caption.
- Log axes for error vs horizon (errors span decades) and for parameter counts.
- Colorblind-safe categorical palette (matplotlib `tab10` is acceptable; prefer
  `tableau-colorblind10`); model identity → fixed color across *all* figures
  (`nssc.visualization.style.MODEL_COLORS`).
- Chaotic systems: pointwise error saturates; show VPT and attractor-statistics panels
  rather than pretending horizon-500 MSE is informative.

## Project-specific conventions
- Matplotlib only in the package (plotly is dashboard-only, optional dep).
- Style set once in `nssc.visualization.style.apply()`: font size 9, figure widths
  3.4 in (single) / 7.0 in (double), `savefig(dpi=200, bbox_inches="tight")`, both
  `.png` and `.pdf` to `results/figures/<figXX_name>.{png,pdf}`.
- Every figure function signature: `plot_<name>(df: pd.DataFrame, out: Path, **kw)`,
  input is a tidy DataFrame from `nssc.utils.registry.load_results()` (columns:
  experiment_id, system, model, candidate_id, seed, mode, horizon, metric, value,
  n_params, latency_ms, ...).
- Filenames are stable ids used by docs and README:

  | id | file | content | source cell |
  |----|------|---------|-------------|
  | F1 | `fig01_dataset_overview` | 2–3 sample trajectories per system + observation-map example (obs dim, noise) | A |
  | F2 | `fig02_latent_dim_sweep` | validation recursive NRMSE@25 and recon vs latent dim d, per system, with intrinsic n marked | B |
  | F3 | `fig03_rollout_error_vs_horizon` | NRMSE vs horizon (log-log), all baselines + compiled model, one panel per system, mode = recursive | D |
  | F4 | `fig04_pareto_complexity` | long-horizon error (H=100 or VPT) vs parameter count / latency; Pareto front highlighted; compiled candidates vs baselines | H |
  | F5 | `fig05_stability_spectrum` | eigenvalues of latent Jacobians (unit circle) / spectral radius distribution, and norm growth over 2000-step free rollouts | G |
  | F6 | `fig06_selection_ablation` | val-MSE-selected vs multi-objective-selected: rollout error at H ∈ {50,100,250,500} and fraction of diverged rollouts | E |
  | F7 | `fig07_multiscale_ablation` | single-scale vs slow/fast latent on FHN, Kuramoto two-cluster, Lorenz-96: error vs horizon | F |
  | F8 | `fig08_ood_generalization` | error vs parameter distance from training range (Van der Pol μ, Lorenz ρ, L96 F), ID band shaded | I |
  | F9 | `fig09_latent_phase_portraits` | 2-D/3-D latent trajectories of the compiled model vs ground-truth state (after linear alignment, R² in title) — caption must warn against physical interpretation | C |
  | F10 | `fig10_compiler_search_trace` | candidates surviving each search stage, score J vs stage, wall-clock per stage, chosen model marked | L |

  Additional figures get ids F11+ and a row in this table before merge.
- Captions are generated into `results/figures/captions.md` by the script and include:
  EXP ids, seeds, mode, n, CI type.

## Implementation requirements
- `scripts/make_figures.py --fig all|F3 --registry results/registry.jsonl` regenerates
  deterministically; running twice yields byte-identical PDFs (fixed metadata,
  `metadata={"CreationDate": None}`).
- Figures fail loudly if data are missing (`raise MissingResults(exp_ids)`) rather than
  plotting partial data silently.
- Every figure function has a unit test with a synthetic DataFrame (smoke: file created,
  no exception, axes labels present).
- Legends outside plot area or in an empty panel for multi-panel figures; never overlap
  data.
- Axis labels include units: "NRMSE", "horizon (steps)", "parameters", "latency (ms /
  step, CPU)".

## Common failure modes
- Averaging over seeds before computing log → use per-seed values then aggregate.
- Comparing `teacher_forced` and `recursive` curves on the same axes without labeling.
- Plotting only the best seed's rollout as "typical".
- Rainbow colormaps for ordinal quantities (use `viridis`).
- Phase-portrait figure implying `z_1 = x` of Lorenz; always show alignment R² and use
  hedged caption.
- Hardcoding metric values or paths in the plotting script.

## Validation checklist
- [ ] Figure produced by `scripts/make_figures.py` from registry data; no manual edits.
- [ ] Uncertainty band + n in caption; mode label in title/legend.
- [ ] Fixed model colors; colorblind-safe.
- [ ] Log axes where errors/params span decades.
- [ ] Filename matches the F-table; caption in `captions.md` cites EXP ids.
- [ ] Deterministic regeneration verified.
