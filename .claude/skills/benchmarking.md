# Skill: benchmarking

## Purpose
The baseline suite, metric definitions, and Pareto analysis used to answer H1/H5 in
`experiments/benchmarks/`. Benchmarks are fixed contracts: **never modify a benchmark
definition because a model performs poorly** (CLAUDE.md).

## Relevant theory
- The research question compares a *compiled* low-dimensional latent SSM against
  *large sequence models fitted directly*. So the baseline suite must span: trivial
  (persistence), linear (PCA + linear dynamics), small nonlinear latent (AE + MLP), and
  the strong direct sequence models (GRU, LSTM, TCN, Transformer, SSM/S4-style).
- Persistence sets the floor: any model worse than `x̂_{t+h} = x_t` at horizon h has
  learned nothing about dynamics at that horizon.
- Complexity vs accuracy is a multi-objective problem; report the Pareto front, not a
  single "winner".

## Project-specific conventions

### Baselines (all registered under `nssc.representations` / `nssc.dynamics`, configs in
`configs/models/`)
| id | encoder / model | dynamics | eval mode(s) | notes |
|----|-----------------|----------|--------------|-------|
| `persistence` | — | `x̂_{t+h}=x_t` | recursive (trivially) | zero params |
| `pca_linear` | PCA to d | least-squares linear `A` (DMD-style) | teacher_forced, recursive | closed form; d swept |
| `linae_linear` | linear AE (d) | linear `A` | tf, recursive | trained jointly |
| `mlpae_mlp` | MLP AE (d) | MLP residual `z + f(z)` | tf, recursive | the "small latent" reference |
| `gru` | GRU (hidden h) on x | GRU state = latent | tf, recursive | h ∈ {64,128,256} |
| `lstm` | LSTM | — | tf, recursive | same h grid |
| `tcn` | temporal conv net, receptive field ≥ context | direct multi-horizon or autoregressive | direct, recursive | dilations 1..2^k |
| `transformer` | causal encoder-only, d_model ∈ {64,128}, 2–4 layers | autoregressive | direct, recursive | positional enc, context = eval context |
| `ssm` | diagonal linear SSM (S4D/Mamba-lite style) layers | — | tf, recursive | the "large SSM fitted directly" reference |
| `nssc_compiled` | compiler output | chosen family | tf, recursive | the object under test |

Budget parity: same `max_epochs`, `patience`, optimizer (AdamW, lr from model config,
cosine schedule), same train/val/test trajectories, same `context_length`. Model sizes
are swept on a small grid and *all* grid points are reported (Pareto), with the
val-selected size marked.

### Metrics (`nssc.metrics`, names are the metric column values)
- `recon_mse`: `mean‖D(E(x)) − x‖²` (per-dim mean, on normalized data), teacher-forced.
- `kstep_mse@h`, `nrmse@h`: `‖x̂_{t+h} − x_{t+h}‖ / σ_x` per horizon
  h ∈ {1,5,10,25,50,100,250,500}, mode-labeled. NRMSE denominator: per-dim std of the
  *train* split, averaged over dims.
- `vpt`: valid prediction time = first h at which `nrmse@h > 0.5` (per trajectory;
  median/IQR reported), in steps and in Lyapunov times where λ₁ known.
- `attractor_stat_err`: for chaotic systems, Wasserstein-1 between per-dim marginals of
  1000-step free rollout and ground truth; plus relative error of mean and std.
- `diverged_frac`: fraction of rollouts with NaN/inf or norm > 10× data max.
- `spectral_radius`: max |eig| of latent Jacobian at 100 sampled latent points (mean, max).
- `lyapunov_est`: leading Lyapunov exponent estimated from the latent rollout.
- `n_params`, `latent_dim`, `flops_per_step` (estimated), `latency_ms_step` (CPU,
  batch 1, median of 100), `train_time_s`, `peak_mem_mb`.
- `calibration` (probabilistic models only): PIT histogram uniformity / CRPS.

### Pareto analysis (`nssc.metrics.pareto`)
- Objectives: minimize (`nrmse@100` recursive, `n_params`) and separately
  (`nrmse@100`, `latency_ms_step`). Non-dominated set computed per system per seed, then
  the *frequency* with which each model is on the front across seeds is reported (a
  model on the front in 5/5 seeds is robustly efficient). Hypervolume relative to
  reference point (nrmse = 1.0, params = max) as a scalar summary.

## Implementation requirements
- One config per baseline in `configs/models/`; drivers in `experiments/benchmarks/`
  loop over models × seeds and register each run.
- Baseline definitions are frozen once used in a registered experiment; changes require
  a new model id (`gru_v2`) and a `research/decisions.md` entry.
- Persistence and PCA+linear are always included (they cost nothing and calibrate the
  reader).
- Metric functions are pure `torch`/`numpy` with unit tests against hand-computed values;
  `nrmse` of persistence on a constant series must be 0; of noise must be ≈√2.
- Latency measured with `torch.set_num_threads(1)`, warm-up 10, then median of 100
  single-step calls; device and torch version recorded.
- Tables: `results/tables/benchmark_<system>.md` from `scripts/make_tables.py`.

## Common failure modes
- Baseline under-tuned relative to the proposed model ("straw man"): use the same
  patience/epochs and a size grid; report the grid.
- Comparing direct-mode Transformer against recursive GRU at horizon 100 without saying
  so.
- Different normalization for baselines and compiled model.
- Reporting `recon_mse` for GRU (it has no decoder of x from a bottleneck — mark N/A).
- Changing the horizon list or NRMSE denominator after seeing results.
- Latency measured on MPS for one model and CPU for another.

## Validation checklist
- [ ] All 10 baseline rows present (or explicitly N/A with reason) for each system.
- [ ] Budget parity documented; size grid reported.
- [ ] Every metric row has `mode` and `horizon` (or `horizon = null` for static metrics).
- [ ] Persistence floor plotted/tabulated.
- [ ] Pareto front computed per seed; frequencies + hypervolume reported.
- [ ] Latency/params measured identically across models.
- [ ] Benchmark definition unchanged since first registered use (or new id).
