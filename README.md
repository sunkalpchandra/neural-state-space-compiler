# Neural State-Space Compiler

**Neural State-Space Compiler (`nssc`) automatically discovers compact latent dynamical
systems from high-dimensional temporal observations.**

Given multivariate observations `x_{1:T}, x_t ∈ R^D`, the compiler profiles the data,
searches over candidate latent dimensions, encoders and latent-dynamics families,
scores every candidate on reconstruction, one-step prediction, long-horizon rollout,
complexity and stability, and emits a compiled model

    z_t = E_φ(x_≤t),   z_{t+1} = F_θ(z_t),   x̂_t = D_ψ(z_t),   d ≪ D

together with a report explaining *why* that model was selected.

> **Status:** research code under active development. Every number in this README, in
> `results/`, and in the figures is produced by scripts from registered experiment runs
> (`results/registry.jsonl`). Sections marked *pending* have not been run yet.

## Research question

> Can we automatically compile high-dimensional temporal observations into a
> low-dimensional, structured, predictive state-space representation that preserves the
> underlying dynamics better than simply fitting a large sequence model?

Hypotheses H1–H7 and their falsification criteria: [research/hypotheses.md](research/hypotheses.md).

## Architecture

```
raw temporal dataset
   ↓  DatasetProfiler          intrinsic dim (PCA / MLE / corr-dim), autocorrelation, noise, stationarity, chaos hint
   ↓  CandidateGenerator       latent dims × encoders × dynamics families (registry-driven)
   ↓  StagedSearch             screen → fine → final; resumable; every run registered + checkpointed
   ↓  Evaluator + Stability    recon / teacher-forced / recursive rollout; Jacobian spectra, Lyapunov, norm growth
   ↓  MultiObjectiveScorer     J = λ1·recon + λ2·1-step + λ3·rollout + λ4·complexity + λ5·instability
   ↓  CompiledModel + report   selected model, ranking, data-derived reasons
   ↓  Explorer dashboard       latent trajectories, phase portraits, vector field, counterfactual rollouts
```

Components (all registered, so new families plug in without touching the compiler):

| encoders | dynamics | baselines |
|---|---|---|
| PCA, linear AE, MLP AE, temporal conv (TCN), GRU/LSTM, diagonal SSM, **multi-scale slow/fast** | linear, affine, MLP, residual MLP, Koopman (lifted linear), neural ODE (RK4), SSM, **multi-scale**, Gaussian (probabilistic) | persistence, mean, GRU, LSTM, TCN, Transformer, SSM forecasters |

Details: [docs/architecture.md](docs/architecture.md), [docs/compiler.md](docs/compiler.md).

## Installation

```bash
git clone https://github.com/sunkalpchandra/neural-state-space-compiler
cd neural-state-space-compiler
pip install -e ".[dev]"            # + ".[dashboard]" for the explorer, ".[eeg]" for real data
pytest -q -m "not slow"           # ~1 min
```

## Quick start

```bash
nssc smoke                                              # tiny end-to-end run (seconds)
nssc data list                                          # synthetic systems
nssc profile  --config configs/datasets/lorenz63.yaml   # dataset profile
nssc train    --config configs/experiments/lorenz63_mlp_resmlp.yaml
nssc compile  --config configs/compiler/tiny.yaml       # tiny compiler run (< 1 min)
nssc compile  --config configs/compiler/lorenz63.yaml   # real compiler run (hours, resumable)
nssc benchmark --suite synthetic_core                   # baselines vs latent models, 5 seeds
nssc tables   --suite synthetic_core --reference mlpae_resmlp_d3
nssc visualize --compile-dir results/compile/lorenz63
nssc report   --compile-dir results/compile/lorenz63
nssc dashboard                                          # http://127.0.0.1:8050
```

Python API:

```python
from nssc.compiler import StateSpaceCompiler
from nssc.utils.config import load_config

compiler = StateSpaceCompiler(load_config("configs/compiler/lorenz63.yaml"))
compiled = compiler.run()              # profile → candidates → staged search → select
print(compiled.report.to_markdown())   # selected d / encoder / dynamics + reasons
x_hat, z_hat = compiled.rollout(x_context, horizon=250)
```

## Compiler output

*pending — filled from `results/compile/*/compile_report.md` once the Lorenz-63 / Van der Pol /
high-dimensional Lorenz compiles finish.*

## Benchmarks

*pending — `results/tables/synthetic_core.md` (5 seeds, mean ± std, trajectory-level splits).*

## Interactive demo

`nssc dashboard` serves a dark, instrument-style explorer: raw signal, learned latent
state, latent phase portrait, true-vs-predicted rollout with adjustable horizon,
latent vector field, local Jacobian eigenvalues, the compiler's decision table, and
counterfactual rollouts from an edited initial state.

## Reproducibility

Every run records seed, dataset version (config hash), full config, git commit,
parameter count, training time, hardware, metrics and checkpoint path in
`results/registry.jsonl` (`EXP-####`, append-only; failed runs are kept). Splits are
always trajectory-level (or subject-level for EEG). `scripts/reproduce.sh
{smoke,compile,ablations,benchmark,figures}` re-runs each block. See
[docs/reproducibility.md](docs/reproducibility.md).

## Project structure

```
configs/{datasets,models,experiments,compiler}   YAML (every experimental setting is explicit)
src/nssc/data            synthetic systems (11), observation maps, real loaders, trajectory splits
src/nssc/representations encoders/decoders        src/nssc/dynamics   latent transition families
src/nssc/compiler        profiler, scorer, report, StateSpaceCompiler
src/nssc/search          candidate space, resumable staged search, suite runner
src/nssc/stability       spectra, Lyapunov, norm growth     src/nssc/uncertainty  probabilistic rollouts
src/nssc/metrics         central metrics                    src/nssc/evaluation   protocols, OOD, failure analysis, tables
src/nssc/baselines       sequence-model forecasters         src/nssc/visualization figures (script-generated)
dashboard/               FastAPI + Plotly explorer          tests/{unit,integration,regression}
research/                hypotheses, experiment log, failures, decisions, open questions
```

## Experiments

The experiment matrix (A–L) and review gates are in [docs/experiments.md](docs/experiments.md);
the running log is [research/experiment_log.md](research/experiment_log.md).

## Citation

```bibtex
@software{chandra2026nssc,
  author = {Chandra, Sunkalp},
  title  = {Neural State-Space Compiler},
  year   = {2026},
  url    = {https://github.com/sunkalpchandra/neural-state-space-compiler}
}
```

## License

MIT — see [LICENSE](LICENSE).
