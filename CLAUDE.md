# Neural State-Space Compiler (nssc) — CLAUDE.md

## PROJECT

`nssc` automatically compiles high-dimensional multivariate time series
`x_{1:T}, x_t ∈ R^D` into a compact latent state-space model:

    z_t = E_φ(x_≤t)            (encoder / state inference)      z_t ∈ R^d, d ≪ D
    z_{t+1} = F_θ(z_t, u_t)    (latent dynamics)
    x̂_t = D_ψ(z_t)             (decoder)

The *compiler* profiles a dataset, searches over candidate latent dimensions,
encoders and dynamics families, scores them under a configurable multi-objective
(reconstruction, one-step, rollout, complexity, stability), and emits a compiled
model plus a human-readable justification. This is a research codebase, not a demo.

## RESEARCH QUESTION

> Can we automatically compile high-dimensional temporal observations into a
> low-dimensional, structured, predictive state-space representation that
> preserves the underlying dynamics better than simply fitting a large sequence
> model?

Sub-hypotheses live in `research/hypotheses.md`. Every experiment must state
which hypothesis it tests.

## ARCHITECTURE (see docs/architecture.md)

    raw dataset → DatasetProfiler → CandidateGenerator → StagedSearch
      → (train encoder+dynamics+decoder per candidate) → Evaluator
      → StabilityAnalyzer → MultiObjectiveScorer → CompiledModel + CompileReport

Package layout under `src/nssc/`:

| package           | responsibility |
|-------------------|----------------|
| `data/`           | synthetic system generators, observation maps, real loaders, trajectory-level splits |
| `representations/`| encoders/decoders: PCA, linear AE, MLP AE, TCN, GRU, SSM, multi-scale slow/fast |
| `dynamics/`       | latent transitions: linear, affine, MLP, residual, Koopman, neural ODE, SSM, Gaussian |
| `compiler/`       | profiler, candidate registry, scorer, compile report, `StateSpaceCompiler` |
| `search/`         | staged/resumable search over candidates |
| `stability/`      | Jacobians, spectral radius, Lyapunov estimates, norm growth |
| `uncertainty/`    | probabilistic rollouts, calibration |
| `metrics/`        | central metrics: recon, k-step, NRMSE, complexity, latency, calibration |
| `training/`       | trainer, losses, schedules, checkpointing |
| `evaluation/`     | rollout protocols (teacher-forced / recursive / direct), failure analysis |
| `visualization/`  | figure generation (scripts only, never hand-edited) |
| `utils/`          | config, seeding, registry, hashing, git info, hardware info |
| `cli/`            | `nssc` typer app |

Every model component registers itself in a registry (`nssc.utils.registry`) so
the compiler can enumerate candidates without hard-coded imports. New dynamics or
encoders MUST be addable without editing the compiler.

## DIRECTORY STRUCTURE

    configs/{datasets,models,experiments,compiler}/   YAML only
    src/nssc/                                          package
    experiments/{synthetic,real_world,ablations,benchmarks}/  experiment drivers
    scripts/                                           thin entrypoints
    tests/{unit,integration,regression}/
    results/{raw,processed,figures,tables}/            generated; raw/ checkpoints are gitignored
    research/                                          hypotheses, log, failures, decisions, open questions
    docs/                                              architecture, compiler, experiments, reproducibility
    dashboard/                                         interactive explorer
    agents/  .claude/agents/  .claude/skills/          agent roles + skills

## EXPERIMENT PROTOCOL

- Splits are ALWAYS trajectory-level (never random timesteps). Parameterized
  systems additionally use train/unseen parameter ranges. Subject-level splits for EEG.
- Every run is registered in the experiment registry (`results/registry.jsonl`)
  with: experiment_id, git_commit, config_hash, dataset, model, seed, params,
  train_time, hardware, metrics, checkpoint, status.
- Final experiments use seeds 0–4 and report mean ± std (+ CI where appropriate).
- Rollout horizons: 1, 5, 10, 25, 50, 100, 250, 500 where feasible.
- Evaluation modes are always labeled: `teacher_forced`, `recursive`, `direct`.
- Never optimize/select on the test split. Selection uses validation only.

## CODING STANDARDS

- Python ≥3.10, PyTorch. Type hints on public APIs. Docstrings state shapes.
- Tensors are `(batch, time, dim)` unless a docstring says otherwise.
- No monolithic scripts. Config-driven (YAML → dataclasses in `nssc.utils.config`).
- `ruff` clean. Line length 100.
- Do not import heavy optional deps (mne, fastapi, plotly) at package import time.

## TESTING REQUIREMENTS

- Every subsystem gets unit tests in `tests/unit/`.
- `tests/integration/` runs dataset → train → compile → evaluate → report on a tiny system.
- `tests/regression/` stores known metric values with tolerances.
- Numerical tests: finite outputs, no NaNs, gradient flow, determinism under seed, save/load round-trip.
- `pytest -q -m "not slow"` must pass before every push.

## REPRODUCIBILITY REQUIREMENTS

Every experiment records seed, dataset version, config (+hash), git commit,
model config, parameter count, training time, hardware, metrics, checkpoint path.
Any change to split / preprocessing / architecture / loss / optimizer / seed /
horizon must appear explicitly in configuration — never as a silent code default change.

## SCIENTIFIC INTEGRITY RULES

- Never fabricate results. Every number in README/figures/tables comes from a
  registered run.
- No "outperforms existing methods" language unless a benchmark in this repo shows it.
- **Never modify benchmark definitions simply because a model performs poorly.**
- **Never delete failed experiments. Failed experiments are scientific evidence
  and should remain traceable** (registry status `failed`, entry in `research/failures.md`).
- Do not claim latent dimensions correspond to physical/biological variables
  unless explicitly demonstrated.
- Ask "what scientific claim is this testing?" before "does it run?".

## GIT WORKFLOW

- Commit after every meaningful, self-contained change (one file / one concern
  per commit is preferred). Prefixes: `feat:`, `fix:`, `test:`, `exp:`, `docs:`,
  `refactor:`, `chore:`, `build:`, `ci:`.
- Before push: tests pass, ruff passes, imports work, no credentials, no large artifacts.
- The paper (`paper/`) is out of scope for this project by owner decision.

## AGENT WORKFLOW

Researcher proposes hypothesis → architect checks design → engineer implements →
tests → benchmark → reviewer inspects → researcher decides next. Subagents get
objective, files, constraints, expected output, verification criteria. Their
output is always inspected before commit.
