# `nssc.evaluation`

### `nssc.evaluation`

Evaluation protocols: reconstruction, teacher-forced one-step, recursive rollout, cost, stability.

### `nssc.evaluation.aggregate`

Aggregate registry records into tables: mean ± std over seeds, CIs, paired tests, Pareto sets.

#### `bootstrap_ci(values: 'list[float]', n_boot: 'int' = 2000, alpha: 'float' = 0.05, seed: 'int' = 0) -> 'tuple[float, float]'`

#### `format_markdown(rows: 'list[dict[str, Any]]', metrics: 'list[str]', labels: 'dict[str, str] | None' = None) -> 'str'`

#### `group_runs(records: 'list[dict[str, Any]]', suite: 'str | None' = None) -> 'dict[tuple[str, str], list[dict[str, Any]]]'`

(dataset, model_name) → list of completed records (one per seed; latest per seed wins).

#### `load_groups(registry_path: 'str' = 'results/registry.jsonl', suite: 'str | None' = None)`

#### `mean_std(values: 'list[float]') -> 'tuple[float, float, int]'`

#### `paired_test(a: 'list[float]', b: 'list[float]') -> 'dict[str, float]'`

Paired comparison across seeds (same seeds!): Wilcoxon signed-rank + paired t.
With n=5 the minimum two-sided Wilcoxon p is 0.0625 — report, don't over-interpret.

#### `pareto_front(points: 'list[tuple[float, float]]') -> 'list[bool]'`

Minimise both coordinates. Returns mask of Pareto-efficient points.

#### `summary_table(groups: 'dict[tuple[str, str], list[dict[str, Any]]]', metrics: 'list[str]') -> 'list[dict[str, Any]]'`

### `nssc.evaluation.evaluator`

Standard evaluation of a LatentModel on held-out trajectories.

Evaluation modes are always labelled explicitly:

* ``recon``            x̂ = D(E(x))
* ``teacher_forced``   x̂_{t+1} = D(F(E(x)_t))   (one-step, ground-truth history)
* ``recursive``        encode ``context`` steps, roll F forward H steps, decode

#### class `EvalConfig(context: 'int' = 20, horizons: 'tuple[int, ...]' = (1, 5, 10, 25, 50, 100, 250, 500), max_horizon: 'int | None' = None, batch_size: 'int' = 64, stability: 'bool' = True, stability_horizon: 'int' = 200, latency: 'bool' = True, divergence_threshold: 'float' = 1.0, extra: 'dict[str, Any]' = <factory>) -> None`

EvalConfig(context: 'int' = 20, horizons: 'tuple[int, ...]' = (1, 5, 10, 25, 50, 100, 250, 500), max_horizon: 'int | None' = None, batch_size: 'int' = 64, stability: 'bool' = True, stability_horizon: 'int' = 200, latency: 'bool' = True, divergence_threshold: 'float' = 1.0, extra: 'dict[str, Any]' = <factory>)


#### `evaluate_model(model: 'LatentModel', x: 'Tensor', cfg: 'EvalConfig | None' = None, sigma: 'np.ndarray | None' = None, device: 'torch.device | None' = None, dt: 'float' = 1.0) -> 'dict[str, Any]'`

``x``: (N, T, D) held-out, normalised like training data. Returns flat metrics dict
plus ``curves`` (per-step NRMSE) and ``stability`` sub-dicts.

### `nssc.evaluation.failure_analysis`

Automated failure categorisation for trained latent models.

Categories (a run may receive several):

    representation_collapse   latent variance concentrated in ≪ d dims / near-constant z
    poor_reconstruction       recon NRMSE ≫ noise floor
    latent_instability        rollout blow-up / spectral radius ≫ 1
    chaotic_divergence        one-step good, rollout diverges at rate consistent with positive λ
    poor_long_horizon         rollout error high without blow-up or chaos signature
    overfitting               train loss ≪ val loss
    underfitting              train loss high and still decreasing at the end
    noise_sensitivity         (needs paired noisy/clean runs) — flagged from tags only
    ood_failure               (needs OOD eval) — flagged from metrics keys if present
    training_failure          run failed / non-finite loss

Thresholds are explicit and configurable; the output records the evidence used.

#### class `FailureReport(categories: 'list[str]' = <factory>, evidence: 'dict[str, Any]' = <factory>, verdict: 'str' = 'ok') -> None`

FailureReport(categories: 'list[str]' = <factory>, evidence: 'dict[str, Any]' = <factory>, verdict: 'str' = 'ok')

- `to_dict(self) -> 'dict[str, Any]'`

#### class `FailureThresholds(recon_nrmse_bad: 'float' = 0.3, rollout_nrmse_bad: 'float' = 0.8, one_step_good: 'float' = 0.1, rho_unstable: 'float' = 1.5, blowup_frac: 'float' = 0.1, overfit_ratio: 'float' = 3.0, underfit_train_loss: 'float' = 0.5, collapse_var_ratio: 'float' = 0.02, collapse_frac_dead: 'float' = 0.5) -> None`

FailureThresholds(recon_nrmse_bad: 'float' = 0.3, rollout_nrmse_bad: 'float' = 0.8, one_step_good: 'float' = 0.1, rho_unstable: 'float' = 1.5, blowup_frac: 'float' = 0.1, overfit_ratio: 'float' = 3.0, underfit_train_loss: 'float' = 0.5, collapse_var_ratio: 'float' = 0.02, collapse_frac_dead: 'float' = 0.5)


#### `analyze_run(output_dir: 'str', split: 'str' = 'val') -> 'FailureReport'`

Categorise a finished run from its on-disk artefacts (metrics.json, history.json, checkpoint).

#### `categorize(metrics: 'dict[str, Any]', history: 'list[dict[str, Any]] | None' = None, latent_profile: 'dict[str, Any] | None' = None, thr: 'FailureThresholds | None' = None, split: 'str' = 'val') -> 'FailureReport'`

#### `latent_variance_profile(model: 'LatentModel', x: 'torch.Tensor') -> 'dict[str, Any]'`

### `nssc.evaluation.ood`

Out-of-distribution evaluation for a trained checkpoint.

Two protocols (Experiments G and H):

* ``param_shifts``: re-simulate the training system with shifted dynamical parameters
  (e.g. Lorenz ρ ∈ {20, 35} after training at ρ = 28) and evaluate the frozen model
  with the *training* normalisation statistics.
* ``ic_scales``: re-simulate with a widened initial-condition distribution and no
  transient (so states start off-attractor).

Reports in-distribution reference metrics, per-condition metrics and
``degradation_ratio`` = OOD rollout NRMSE / ID rollout NRMSE.

#### `evaluate_ood(checkpoint: 'str', param_shifts: 'dict[str, list[float]] | None' = None, ic_scales: 'list[float] | None' = None, n_traj: 'int' = 20, eval_cfg: 'EvalConfig | None' = None, device: 'torch.device | None' = None, ref_key: 'str' = 'recursive/nrmse_mean', seed: 'int' = 1234) -> 'dict[str, Any]'`

### `nssc.evaluation.tables`

Generate benchmark tables (markdown + json) from the experiment registry.

#### `suite_tables(suite: 'str', registry_path: 'str' = 'results/registry.jsonl', out_dir: 'str | Path' = 'results/tables', metrics: 'list[str] | None' = None, reference_model: 'str | None' = None) -> 'dict[str, Any]'`
