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

## CURRENT STATE

Before adding a claim to any document, read:

- `research/review_2026-08-18.md` — adversarial multi-agent review of `README.md`, `research/*.md`,
  `results/tables/*` and `docs/*` against the actual code, configs and `results/registry.jsonl`;
  43 findings survived two independent refutation attempts each. It is evidence, not a to-do list:
  do not delete it.
- `research/experiment_log.md` — chronological record of what has actually been run.

Several documents were drafted before the code and describe APIs that were never implemented.
When a document and the code disagree, the code wins: verify against `src/nssc/` first, and label
genuinely absent behaviour "not implemented" rather than describing it as if it exists.

## RESEARCH QUESTION

> Can we automatically compile high-dimensional temporal observations into a
> low-dimensional, structured, predictive state-space representation that
> preserves the underlying dynamics better than simply fitting a large sequence
> model?

Sub-hypotheses live in `research/hypotheses.md`. Every experiment must state
which hypothesis it tests.

## ARCHITECTURE (see docs/architecture.md)

    raw dataset → profile_dataset → generate_candidates → StagedSearch
      → per candidate × seed: run_experiment
          (build_latent_model → Trainer.fit → evaluate_model, which calls analyze_stability
           unless eval.stability is off)
      → MultiObjectiveScorer.rank + prune, per stage
      → StateSpaceCompiler.compile → CompiledModel + CompileReport

Layout under `src/nssc/` (`experiment.py` is a single module; the rest are subpackages):

| path              | responsibility |
|-------------------|----------------|
| `data/`           | synthetic system generators, integrators, observation maps, real loaders (EEG BCI, motion), trajectory / parameter-range / subject splits |
| `representations/`| encoders `pca, linear, mlp, tcn, gru, lstm, ssm, multiscale`; decoders `pca, linear, mlp` |
| `dynamics/`       | latent transitions `linear, affine, mlp, residual_mlp, koopman, neural_ode, ssm, gaussian, multiscale` |
| `models/`         | assembles registered encoder + dynamics + decoder into a `LatentModel` (`build_latent_model`) |
| `baselines/`      | observation-space sequence forecasters `gru, lstm, tcn, ssm, transformer, persistence, mean`, their own trainer, and `recursive`/`direct` evaluation; `LatentModelForecaster` puts a compiled model through the same harness |
| `compiler/`       | dataset profiler, multi-objective scorer, compile report, `StateSpaceCompiler` |
| `search/`         | candidate space (`generate_candidates` — latent dim × encoder × dynamics), staged search, resumable search state, benchmark-suite runner |
| `stability/`      | Jacobians, spectral radius, Lyapunov estimates, norm growth |
| `uncertainty/`    | probabilistic rollouts and their calibration, for stochastic dynamics |
| `metrics/`        | central metrics: recon, k-step, NRMSE, complexity, latency, calibration |
| `training/`       | trainer, losses, schedules, checkpointing |
| `experiment.py`   | single-run pipeline `run_experiment`: dataset → model → train → evaluate → checkpoint → registry |
| `evaluation/`     | `evaluate_model` rollout protocols (`recon` / `teacher_forced` / `recursive`), failure analysis, OOD, seed aggregation, suite tables, Pareto fronts |
| `visualization/`  | figure generation (scripts only, never hand-edited) |
| `utils/`          | config, seeding, component registry, experiment registry, hashing, git info, hardware info |
| `cli/`            | `nssc` typer app: `profile, train, evaluate, registry, compile, benchmark, visualize, report, tables, pareto, failures, smoke, dashboard, data` |

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
