# Skill: experiment-design

## Purpose
How an `nssc` experiment is specified so that it is comparable, reproducible, and answers
a hypothesis. Covers splits, seeds, horizons, evaluation modes, budgets, and the config
schema in `configs/experiments/`.

## Relevant theory
- **Trajectory-level splits**: temporal windows from one trajectory are strongly
  correlated; a random timestep split leaks the future into training and inflates every
  metric. Splits are over whole trajectories (or subjects for EEG, recording sessions for
  motion capture).
- **In-distribution vs. OOD**: for parameterized systems (Van der Pol μ, Lorenz-63 ρ,
  Lorenz-96 F, Kuramoto K, Gray–Scott (F,k)) an additional split partitions the
  *parameter range*: `param_range_train` (e.g. μ ∈ [0.5, 2.0]) and `param_range_test`
  (μ ∈ [2.5, 4.0]). ID test = unseen trajectories, seen range; OOD test = unseen range.
- **Evaluation modes** (always labeled):
  - `teacher_forced`: at each t, `z_t = E(x_≤t)` from ground-truth inputs, predict
    `x̂_{t+1} = D(F(z_t))`. Measures one-step fit only.
  - `recursive`: encode a context window `x_{1:c}` → `z_c`, then roll `F` for H steps
    in latent space, decode each. This is the primary mode for H1/H3/H4.
  - `direct`: model predicts `x̂_{t+h}` for a fixed h from `z_t` in one shot (multi-head
    or h-conditioned). Baselines like TCN/Transformer often do this; keep it separate.
- **Horizons**: 1, 5, 10, 25, 50, 100, 250, 500 steps (in *samples*, dt in config). For
  chaotic systems horizons beyond ~2 Lyapunov times use attractor statistics and
  valid-prediction-time (VPT) alongside NRMSE.
- **Seeds**: 0–4 for reported results; seed controls model init, data order, dropout.
  The dataset seed is separate and fixed per dataset config (so all models see the same
  trajectories).
- **Budget parity**: same max epochs, early-stopping patience (on *validation*
  recursive loss at H=25 by default), optimizer family, and data for every model in a
  comparison, unless the comparison is about budget.

## Project-specific conventions
- Experiment config (`configs/experiments/<name>.yaml`) → `ExperimentConfig` dataclass:
  ```yaml
  experiment: {name: h1_longhorizon_lorenz63, hypothesis: H1, matrix_cell: D}
  dataset: {config: configs/datasets/lorenz63.yaml}
  models: [configs/models/gru.yaml, configs/models/lstm.yaml, ...]   # or compiler
  compiler: {config: configs/compiler/default.yaml}                  # optional
  seeds: [0, 1, 2, 3, 4]
  training: {max_epochs: 200, patience: 20, batch_size: 64, lr: 1e-3,
             rollout_horizon_train: 25, curriculum: true}
  evaluation:
    modes: [teacher_forced, recursive]
    horizons: [1, 5, 10, 25, 50, 100, 250, 500]
    context_length: 50
    n_eval_trajectories: 20
    attractor_stats: true            # for chaotic systems
  selection: {split: val, criterion: multi_objective | val_mse}
  ```
- Split spec lives in the dataset config: `split: {train: 80, val: 10, test: 10}`
  trajectory counts, `seed_offsets: {train: 0, val: 10000, test: 20000}` for
  initial-condition streams; OOD: `param_range_train`, `param_range_test`,
  `n_test_ood`.
- EEG: `split: {by: subject, train_subjects: [...], val_subjects: [...],
  test_subjects: [...]}`. Motion capture: by session/trial.
- Registry row per (experiment, model/candidate, seed). `experiment_id` is per driver
  invocation; runs within share it and differ by `candidate_id` and `seed`.
- Naming: `EXP-0007/gru_h64/seed3`.
- Ablations change exactly one config field vs. a named base config and record
  `base_config_hash` in the registry `params`.

## Implementation requirements
- `nssc.data.splits.trajectory_split(n_traj, spec, seed) -> dict[str, np.ndarray]` is
  the *only* split function; unit test asserts disjointness and coverage.
- `nssc.evaluation.protocols` implements the three modes as separate functions returning
  `{"mode": ..., "horizon": h, "metric": value}` rows; the mode string is written into
  every metrics record — no metric without a mode label.
- Early stopping monitors `val/<criterion>` and never touches test.
- `--dry-run` runs 2 epochs on 2 trajectories, all horizons, and writes the same
  artifacts (used by integration tests).
- Every driver checks `assert set(train_ids) & set(test_ids) == set()` at runtime.
- Horizon list truncated automatically when T − context < 500 and the truncation is
  logged in the registry `note`.

## Common failure modes
- Windows sampled from all trajectories then split → leakage. Use trajectory ids.
- Reporting `teacher_forced` MSE at "horizon 100" (it is always one-step; horizon is
  meaningless there) — label bug that inflates results.
- Different context lengths across models (GRU warm-up 50 vs Transformer 200) — fix
  `context_length` in evaluation config.
- Using validation trajectories from the OOD range for selection (contaminates OOD
  claim): validation is in-range only.
- Early stopping on one-step loss selects models with poor rollouts (this is *the*
  H2 phenomenon; do not let it happen silently in baselines — patience metric is in
  config and reported).
- Seeds 0–2 "because time": mark the run `preliminary` in the registry, exclude from
  tables.

## Validation checklist
- [ ] Config states hypothesis + matrix cell.
- [ ] Split is trajectory-level / subject-level; OOD ranges in dataset config.
- [ ] Modes labeled; horizons from config; context length fixed across models.
- [ ] Seeds 0–4; dataset seed fixed and shared.
- [ ] Budget parity across compared models documented in config.
- [ ] Selection criterion named (`val_mse` or `multi_objective`) and validation-only.
- [ ] Registry rows created for every (candidate, seed), including failures.


## Lessons learned (2026-08-17)
- Pool-normalised scores need an explicit **error floor**: an exactly-zero best term
  (PCA d=D reconstruction) otherwise assigns an unbounded penalty to every other candidate
  (research/failures.md F-001). Check `objective.error_floor` before trusting a ranking.
- Explicit `latent_dims` lists are honoured verbatim (overcomplete latents allowed);
  only `auto` is clipped to `obs_dim`.
- Long background runs: `caffeinate -i`, `OMP_NUM_THREADS=2`, ≤3 concurrent jobs on the
  8-core host; small models are faster on CPU than MPS.
