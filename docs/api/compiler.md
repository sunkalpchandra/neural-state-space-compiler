# `nssc.compiler`

### `nssc.compiler`

Compiler stage: dataset profiling, candidate generation, scoring, reports.

### `nssc.compiler.compiler`

StateSpaceCompiler: profile → candidates → staged search → select → report.

Config schema (see configs/compiler/default.yaml)::

    dataset: {...}                     # run-config dataset block
    windows: {...}   training: {...}   eval: {...}      # base run config (shared by candidates)
    candidates: {latent_dims: auto|[...], encoders: [...], dynamics: [...], hidden_dims: [...]}
    stages: [{name, epochs, seeds, keep_top, keep_frac, score_tolerance, eval: {...}}, ...]
    objective: {reconstruction, one_step, rollout, complexity, stability, criterion, ...}
    output_dir: results/compile/<name>

#### class `CompiledModel(model: 'LatentModel | None', spec: 'CandidateSpec', report: 'CompileReport', checkpoint: 'str | None', output_dir: 'str') -> None`

CompiledModel(model: 'LatentModel | None', spec: 'CandidateSpec', report: 'CompileReport', checkpoint: 'str | None', output_dir: 'str')

- `rollout(self, x_context: 'torch.Tensor', horizon: 'int')`

#### class `StateSpaceCompiler(cfg: 'dict[str, Any] | Config', device: 'torch.device | None' = None, registry: 'ExperimentRegistry | None' = None, log=<built-in function print>) -> 'None'`

- `base_run_cfg(self) -> 'dict[str, Any]'`
- `compile(self, search_result: 'dict[str, Any] | None' = None) -> 'CompiledModel'` — Select the winner of the final stage, load its best-seed checkpoint, write report.
- `evaluate(self, compiled: 'CompiledModel', split: 'str' = 'test') -> 'dict[str, Any]'` — Evaluate the compiled model on a held-out split with the compiler's eval config.
- `fit(self) -> 'dict[str, Any]'` — Stage 1: profile the dataset (cached in output_dir/profile.json).
- `propose(self) -> 'list[CandidateSpec]'` — Stage 2–4: enumerate candidate (latent dim × encoder × dynamics).
- `run(self, resume: 'bool' = True) -> 'CompiledModel'`
- `search(self) -> 'dict[str, Any]'` — Stage 5–6: staged search + multi-objective scoring.

### `nssc.compiler.profiler`

Dataset profiler: cheap numpy statistics that steer the compiler's search.

:func:`profile_dataset` inspects a :class:`~nssc.data.dataset.TrajectoryDataset`
(raw, unnormalised) and returns a :class:`DatasetProfile` with shape / scale
summaries, intrinsic-dimension estimates (PCA, Levina–Bickel MLE, correlation
dimension), temporal structure (autocorrelation, spectrum, smoothness), noise
and non-stationarity estimates, a linear-predictability score, a crude
largest-Lyapunov proxy and a ``recommendations`` dict of boolean/list hints.

All arrays are ``(N, T, D)`` (trajectories, time, observation dim). NaNs in ``x``
(missing values) are tolerated: statistics are NaN-aware and dynamical estimates
use a linearly time-interpolated copy. Every scalar is finite or explicitly NaN.

#### class `DatasetProfile(n_traj: 'int', n_steps: 'int', obs_dim: 'int', dt: 'float', total_samples: 'int', has_missing: 'bool', missing_rate: 'float', sampling_rate_hz: 'float', mean_min: 'float', mean_max: 'float', std_min: 'float', std_median: 'float', std_max: 'float', dynamic_range: 'float', pca_dims_for_variance: 'dict[str, int]', explained_variance_curve: 'list[float]', mle_dim_k10: 'float', mle_dim_k20: 'float', correlation_dim: 'float', suggested_latent_dims: 'list[int]', autocorr: 'list[float]', autocorr_time: 'float', smoothness: 'float', dominant_period_steps: 'float', spectral_flatness: 'float', noise_std_estimate: 'float', signal_std: 'float', noise_ratio_estimate: 'float', nonstationarity_mean: 'float', nonstationarity_std: 'float', nonstationary_dim_fraction: 'float', linear_predictability_r2: 'float', linear_r2_at_10_steps: 'float', lyapunov_proxy: 'float', lyapunov_proxy_per_time: 'float', recommendations: 'dict[str, Any]' = <factory>) -> None`

Result of :func:`profile_dataset`. See module docstring for semantics.

- `to_dict(self) -> 'dict[str, Any]'`
- `to_markdown(self) -> 'str'`

#### `profile_dataset(ds: 'TrajectoryDataset', max_samples: 'int' = 20000, seed: 'int' = 0) -> 'DatasetProfile'`

Profile ``ds`` (raw data, numpy only). See module docstring.

``max_samples`` bounds the number of ``(t, D)`` points used by PCA / linear
fits; MLE (≤5000), correlation dimension (≤2000) and Lyapunov (≤1000) use
tighter internal caps.

### `nssc.compiler.report`

Human- and machine-readable compile report.

The ``reasons`` list is generated from actual comparisons within the final
candidate pool (never templated numbers).

#### class `CompileReport(selected: 'dict[str, Any]', selected_metrics: 'dict[str, Any]', ranking: 'list[dict[str, Any]]', stage_summaries: 'list[dict[str, Any]]', profile: 'dict[str, Any]', weights: 'dict[str, Any]', reasons: 'list[str]' = <factory>, n_runs: 'int' = 0, n_failed: 'int' = 0, wall_time_s: 'float' = 0.0, dataset: 'dict[str, Any]' = <factory>, checkpoint: 'str | None' = None, rollout_key: 'str' = '') -> None`

CompileReport(selected: 'dict[str, Any]', selected_metrics: 'dict[str, Any]', ranking: 'list[dict[str, Any]]', stage_summaries: 'list[dict[str, Any]]', profile: 'dict[str, Any]', weights: 'dict[str, Any]', reasons: 'list[str]' = <factory>, n_runs: 'int' = 0, n_failed: 'int' = 0, wall_time_s: 'float' = 0.0, dataset: 'dict[str, Any]' = <factory>, checkpoint: 'str | None' = None, rollout_key: 'str' = '')

- `save(self, path: 'str | Path') -> 'None'`
- `to_dict(self) -> 'dict[str, Any]'`
- `to_markdown(self) -> 'str'`

#### `build_reasons(rows: 'list[dict[str, Any]]', selected: 'dict[str, Any]', rollout_key: 'str') -> 'list[str]'`

Derive plain-language justifications from the final ranking.

#### `report_for(experiment: 'str | None' = None, compile_dir: 'str | None' = None) -> 'str'`

### `nssc.compiler.scorer`

Multi-objective scoring of candidates.

    J = λ1·L_recon + λ2·L_1step + λ3·L_rollout + λ4·C_complexity + λ5·C_instability

Every term is normalised *within the candidate pool being compared* so the
weights are dimensionless and comparable across datasets:

* error terms (recon / one-step / rollout NRMSE): ``log(x / x_min)`` — 0 for the
  best candidate, +log-ratio otherwise (robust to scale, penalises 2× error the
  same everywhere).
* complexity: ``log(params / params_min) / log(10)`` — one unit per decade of
  parameters.
* instability: ``instability_score`` (already ≥0; 0 = stable) plus a hard
  ``blowup_penalty`` if a run diverged.

Two selection criteria are supported to enable the H2 ablation:
``criterion: multi_objective`` (default) and ``criterion: val_mse``
(rank purely by validation one-step/recon MSE, ignoring rollout/stability/complexity).

#### class `MultiObjectiveScorer(weights: 'ScoreWeights', split: 'str' = 'val') -> 'None'`

- `rank(self, per_candidate: 'dict[str, list[dict[str, Any]]]') -> 'list[dict[str, Any]]'` — ``per_candidate``: cand_id → list of run results (one per seed). Returns rows sorted by J.
- `score(self, agg: 'dict[str, Any]', pool: 'list[dict[str, Any]]', rollout_key: 'str') -> 'tuple[float, dict[str, float]]'`
- `terms(self, agg: 'dict[str, Any]', pool: 'list[dict[str, Any]]', rollout_key: 'str') -> 'dict[str, float]'`

#### class `ScoreWeights(reconstruction: 'float' = 1.0, one_step: 'float' = 1.0, rollout: 'float' = 2.0, complexity: 'float' = 0.1, stability: 'float' = 1.0, blowup_penalty: 'float' = 10.0, rollout_horizon_key: 'str' = 'auto', error_floor: 'float' = 0.01, criterion: 'str' = 'multi_objective', extra: 'dict[str, float]' = <factory>) -> None`

ScoreWeights(reconstruction: 'float' = 1.0, one_step: 'float' = 1.0, rollout: 'float' = 2.0, complexity: 'float' = 0.1, stability: 'float' = 1.0, blowup_penalty: 'float' = 10.0, rollout_horizon_key: 'str' = 'auto', error_floor: 'float' = 0.01, criterion: 'str' = 'multi_objective', extra: 'dict[str, float]' = <factory>)


#### `aggregate_seeds(runs: 'list[dict[str, Any]]') -> 'dict[str, Any]'`

Mean over seeds of scalar summary metrics; carries n_seeds and failure count.

#### `pick_rollout_key(metrics_list: 'list[dict[str, Any]]', prefix: 'str' = 'val/') -> 'str'`

Longest horizon available in *all* candidates' metrics (fallback nrmse_mean).
