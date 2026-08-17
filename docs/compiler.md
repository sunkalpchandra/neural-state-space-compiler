# The nssc compiler

`StateSpaceCompiler.compile(dataset)` turns a dataset into a `CompiledModel` (a
`LatentModel` with chosen `d`, encoder, dynamics, decoder) and a `CompileReport` that
justifies the choice. Everything below is driven by `configs/compiler/*.yaml`
(`CompilerConfig`); the compiler contains no family-specific code (see
`docs/architecture.md` §4).

## Stages

| # | stage | input → output | notes |
|---|-------|----------------|-------|
| 0 | `profile` | dataset → `DatasetProfile` | D, T, n_traj, dt; PCA energy curve → `est_intrinsic_dim` (95 % / 99 % energy elbow, plus a nonlinear estimate — two-NN or correlation dimension on a subsample); autocorrelation time; spectral slope; stationarity flag; noise estimate (high-frequency residual). Cheap, deterministic, no training. |
| 1 | `candidates` | profile + `candidate_space` → `list[Candidate]` | latent dims: config list or derived grid `{⌈n̂/2⌉, n̂, 2n̂, 4n̂}` clipped to `[d_min, d_max]`; encoders × dynamics × decoders from the registry (config may restrict); each candidate has a stable `candidate_id` = `f"{enc}_{dyn}_d{d}"`. Incompatible pairs (e.g. multi-scale encoder with non-multiscale dynamics) are dropped by a declared `compatible_with` set on the component, not by the compiler. |
| 2 | `search.coarse` | candidates → survivors | short training budget (`stages.coarse.budget`: e.g. 15 epochs, 1 seed, subset of train trajectories), monitor validation `onestep` + `recursive nrmse@10`; keep top `stages.coarse.keep` (fraction or count) *plus* any candidate within `stages.coarse.slack` of the best (avoid discarding near-ties). |
| 3 | `search.fine` | survivors → survivors | full training budget, seeds `stages.fine.seeds` (default 1–2), validation metrics at horizons up to 50; keep top-k. |
| 4 | `search.long_horizon` | survivors → survivors | evaluate recursive rollouts on validation at all configured horizons (up to 500) + attractor stats for chaotic data; drop candidates with `diverged_frac > threshold` or `nrmse@H_long > threshold`. |
| 5 | `stability` | survivors → stability metrics | `StabilityAnalyzer`: Jacobians at 100 sampled validation latents, spectral radius (mean/max), Benettin λ₁ estimate over 1000-step latent rollout, norm growth over 2000 free steps, NaN/inf checks. Computed in fp32 on CPU. |
| 6 | `score` | metrics + complexity + stability → `ScoreBreakdown` per candidate | multi-objective J below; also records the Pareto set. |
| 7 | `final` | best candidate → retrain on seeds `stages.final.seeds` (default 0–4), select seed by validation J, evaluate **once** on test, save `CompiledModel` | test metrics appear only here. |
| 8 | `report` | `SearchState` → `compile_report.md/json` | see format below. |

Every stage writes its outputs to `SearchState` and persists it (`search_state.json`
next to the checkpoints) before the next stage starts.

## Multi-objective score

For a candidate `c` with validation metrics:

    J(c) = λ1 · L_recon(c) + λ2 · L_1step(c) + λ3 · L_rollout(c)
         + λ4 · C_complexity(c) + λ5 · C_instability(c)

- `L_recon`  = validation reconstruction NRMSE (`D(E(x))` vs `x`).
- `L_1step`  = validation teacher-forced NRMSE@1.
- `L_rollout` = weighted mean of validation recursive NRMSE over horizons
  `score.horizons` (default {10, 25, 50, 100}), weights `score.horizon_weights`
  (default uniform in log-horizon), with diverged rollouts assigned NRMSE = `score.nrmse_cap`
  (default 2.0). For chaotic datasets (flag from profile / config) `L_rollout` may add
  `score.attractor_weight · attractor_stat_err`.
- `C_complexity` = `α · log10(n_params / n_params_ref) + (1−α) · d / D` (defaults α = 0.5,
  `n_params_ref` = smallest candidate's params), i.e. penalizes parameter count and
  latent dimension; optionally latency (`score.use_latency`).
- `C_instability` = `max(0, ρ_mean − 1) + max(0, λ̂₁ − λ_ref)⁺ + diverged_frac +
  1[norm growth > growth_cap]` where `λ_ref` is the dataset's known/estimated leading
  Lyapunov exponent if any (0 for dissipative non-chaotic systems), so a chaotic system
  is not penalized for being chaotic but a latent model exploding faster than the data
  is.

Normalization: each term is min–max normalized across the surviving candidate set on
validation only (`score.normalization: minmax | none | zscore`) so λ's are comparable;
raw and normalized values are both stored. Default weights
`λ = (1.0, 1.0, 2.0, 0.5, 1.0)`; `selection.criterion: val_mse` sets
`λ = (0, 1, 0, 0, 0)` for the H2 ablation. Weights are config; changing them is a new
`config_hash`.

The scorer also computes the Pareto set over `(L_rollout, C_complexity)` and flags
whether the argmin-J candidate is on it (report field `chosen_is_pareto`).

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
