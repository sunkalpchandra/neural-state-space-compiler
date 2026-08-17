# The nssc compiler

`StateSpaceCompiler.compile(dataset)` turns a dataset into a `CompiledModel` (a
`LatentModel` with chosen `d`, encoder, dynamics, decoder) and a `CompileReport` that
justifies the choice. Everything below is driven by `configs/compiler/*.yaml`
(`CompilerConfig`); the compiler contains no family-specific code (see
`docs/architecture.md` §4).

## Stages

| # | stage | input → output | implementation |
|---|-------|----------------|----------------|
| 0 | `fit` (profile) | dataset → `DatasetProfile` (`profile.json`) | `nssc.compiler.profiler.profile_dataset`: shape/dt, PCA variance curve, Levina–Bickel MLE + correlation dimension, autocorrelation time, smoothness, noise estimate, stationarity, linear one-step/10-step R², Rosenstein Lyapunov proxy → `suggested_latent_dims` and hints (`likely_linear`, `likely_chaotic`, `noisy`, `long_memory`, `nonstationary`). No training. |
| 1 | `propose` | profile + `candidates` config → `list[CandidateSpec]` (`candidates.json`) | `nssc.search.space.generate_candidates`: Cartesian product latent_dims × encoders × dynamics (× hidden_dims). `latent_dims: auto` uses the profile's suggestions. PCA is only paired with linear/affine dynamics; multi-scale encoder/dynamics must agree on `slow_dim`. `candidate.id = enc+dyn@d-<hash>`. |
| 2..k | `search` — one entry per `stages:` item | survivors → survivors | `nssc.search.staged.StagedSearch`. Each stage trains every surviving candidate for `epochs` on `seeds` via `nssc.experiment.run_experiment` (so every run is an `EXP-####` with a checkpoint), evaluates on **validation**, ranks with the multi-objective score below, then prunes with `keep_top` / `keep_frac` while never discarding candidates within `score_tolerance` of the best. Default stages: `screen` (12–20 epochs, 1 seed, capped batches) → `fine` (50–60 epochs, 1 seed) → `final` (120–200 epochs, seeds 0–2). Stability metrics are part of every evaluation (`nssc.stability.analyze_stability`). Identical run configs already completed in the registry are reused (`reuse_registry`). |
| k+1 | `compile` | final ranking → `CompiledModel` (`compiled_model.yaml`, `compile_report.{json,md}`) | winner = rank 1 of the last stage; its best-seed checkpoint (lowest validation rollout NRMSE) is loaded. The report's `reasons` are computed from the ranking (ratios vs best linear candidate, runner-up, smallest model; stability verdict). |
| k+2 | `evaluate` (optional) | compiled model → test metrics | `StateSpaceCompiler.evaluate(compiled, split="test")` — the only place the test split is touched. |

Every stage writes its outputs to `SearchState` and persists it (`search_state.json`
next to the checkpoints) before the next stage starts.

## Multi-objective score

For a candidate `c` with validation metrics:

    J(c) = λ1 · L_recon(c) + λ2 · L_1step(c) + λ3 · L_rollout(c)
         + λ4 · C_complexity(c) + λ5 · C_instability(c) + λ6 · blowup_frac(c)

Implementation: `nssc.compiler.scorer.MultiObjectiveScorer` (this section mirrors the code).

- Every error term is normalised **within the pool of candidates being ranked** as a
  log-ratio to the best candidate: `L_x(c) = log(NRMSE_x(c) / min_c' NRMSE_x(c'))`, so
  the best candidate contributes 0 and a candidate with 2× the error contributes
  log 2 ≈ 0.69 regardless of dataset scale.
  - `L_recon`: validation reconstruction NRMSE.
  - `L_1step`: validation teacher-forced one-step NRMSE.
  - `L_rollout`: validation recursive NRMSE at the longest horizon available for *all*
    candidates in the pool (`objective.rollout_horizon_key: auto`), or an explicit key.
- `C_complexity = log10(n_params(c) / min_c' n_params(c'))` — one unit per decade.
- `C_instability = instability_score` from `nssc.stability.analyze_stability`
  (2·blow-up fraction + collapse fraction + 0.5·max(0, ρ_max − 1) + 0.25·log norm ratio),
  plus a separate `blowup_penalty × frac_blowup` term.
- A missing/NaN term is replaced by a fixed penalty (5.0), never treated as free.
- Seeds are aggregated by the mean of each summary metric before scoring; a candidate
  with zero completed seeds gets J = ∞.

Default weights `(λ1..λ6) = (1.0, 1.0, 2.0, 0.1, 1.0, 10.0)`; see
`configs/compiler/default.yaml`. Alternative criteria for ablations
(`objective.criterion`): `val_mse` (J = ½L_recon + ½L_1step; H2 ablation) and
`rollout_only`. Weights are configuration; changing them is a new experiment.

## Staged search: resumability

- `SearchState` (dataclass, JSON on disk): profile, candidate list, per-stage results
  `{candidate_id: {seed: metrics}}`, per-stage survivor lists with elimination reason,
  stage status (`pending|running|done`), stage wall-clock, git commit, config hash.
- `nssc compile --config X --resume` loads the state, verifies `config_hash` and
  `dataset_hash` match (refuses otherwise), skips `done` stages, and re-runs `running`
  ones from their last completed (candidate, seed) using existing checkpoints
  (`model.pt` present + metadata `epoch == max_epochs or early_stopped`).
- Determinism: candidate order is sorted by `candidate_id`; seeds fixed; so an
  interrupted-then-resumed search yields the same final choice as an uninterrupted one
  (integration test kills after stage 2).
- Registry: every trained (candidate, seed) is a registry row with `experiment_id` of
  the compile run and `candidate_id`; the compile run itself has a summary row with
  `model = "nssc_compiled"`.

## Report format

`compile_report.md` (human) and `compile_report.json` (machine), both under
`results/processed/<EXP-id>/`:

1. **Header** — experiment id, dataset name + hash, config hash, git commit, hardware,
   total wall-clock, date.
2. **Dataset profile** — D, T, n_traj, dt, PCA energy at d ∈ grid, `est_intrinsic_dim`
   (both estimators), autocorrelation time, noise estimate, chaotic flag + λ_ref.
3. **Candidate space** — encoders, dynamics, decoders, latent dims enumerated; number of
   candidates; compatibility exclusions.
4. **Search trace** — table per stage: candidate, seed(s), key validation metrics,
   `kept|eliminated`, reason (`below top-k`, `diverged`, `nrmse@100 > 1.0`, `nan_loss`),
   wall-clock. Nothing is omitted; eliminated candidates remain listed.
5. **Score breakdown** — for finalists: raw and normalized `L_recon, L_1step, L_rollout,
   C_complexity, C_instability`, weights, `J`, Pareto membership.
6. **Chosen model** — candidate id, `d`, families, `n_params`, latency, per-seed
   validation J and the selected seed, **test** metrics (all modes, all horizons, with
   mode labels), stability metrics, checkpoint path.
7. **Justification** — templated plain language, e.g. "Chose `mlp_ae_residual_d4`:
   lowest J (0.21) among 6 finalists; recursive NRMSE@100 on validation 0.18 vs 0.31 for
   the best linear candidate; spectral radius 0.97 (stable); 8.1k parameters (2nd
   smallest). Latent dim 4 vs PCA-95% estimate 3 and two-NN estimate 3.4." Sentences are
   built from numbers in the JSON; no free-text claims. Includes a **caveats** list:
   near-ties within `slack`, candidates eliminated at coarse stage that were within slack,
   diverged seeds, MPS used (if so).
8. **Reproduce** — the exact command line and config hash to re-run.

## Families available to the compiler (registry names)

Encoders: `pca`, `linear_ae`, `mlp_ae`, `tcn`, `gru`, `ssm`, `multiscale`.
Dynamics: `linear`, `affine`, `mlp`, `residual`, `koopman`, `neural_ode`, `ssm`,
`gaussian`. Decoders: `linear`, `mlp`, `pca_inverse`. Each family's docstring states its
transition equation and complexity; new families are added per
`docs/architecture.md` §8 without touching this package.

## CLI

    nssc compile --config configs/experiments/compile_lorenz63.yaml [--seed 0] [--device cpu] [--resume] [--dry-run]
    nssc profile --dataset configs/datasets/lorenz63.yaml
    nssc report  --exp EXP-0012            # re-render report from search_state.json
    nssc registry list | show EXP-0012
    nssc smoke                             # tiny end-to-end run (used by `make smoke`)
