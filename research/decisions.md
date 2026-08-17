# Decisions

Architecture, protocol, and scope decisions with rationale. Append-only; reversals get
a new entry referencing the old id.

## Template

```
### D-xxx — YYYY-MM-DD — <title>
- Decision:
- Alternatives considered:
- Rationale:
- Consequences / what must change:
- Revisit when:
```

---

### D-001 — 2026-08-17 — PyTorch as the only modelling framework
- Decision: all models, losses, Jacobians in PyTorch (≥ 2.1); `torch.func` for
  Jacobians; numpy/scipy for data generation and reference integrators.
- Alternatives considered: JAX (nicer `jacfwd`/`vmap`, but MPS support weak and owner
  tooling is torch-based); scikit-learn-only baselines (insufficient for GRU/Transformer/SSM).
- Rationale: one framework for all baselines and compiled models keeps budget parity and
  latency measurement comparable; MPS available as optional device.
- Consequences: `pytorch-engineering` skill governs shapes `(B,T,D)`, checkpoint format,
  determinism.
- Revisit when: a required op (e.g. batched `eig` for stability) is a bottleneck.

### D-002 — 2026-08-17 — YAML configs → frozen dataclasses; config hash is the protocol id
- Decision: every knob (split, horizons, loss weights, budgets, score weights, device)
  lives in YAML under `configs/{datasets,models,experiments,compiler}/`, loaded into
  dataclasses in `nssc.utils.config`; `config_hash` over the resolved config identifies
  the protocol.
- Alternatives: Hydra/OmegaConf (more features, more magic; harder to hash reliably);
  argparse-only (silent defaults drift).
- Rationale: CLAUDE.md requires that any protocol change be explicit in configuration —
  a hash over the fully resolved dataclass enforces it mechanically.
- Consequences: unknown keys are errors; code defaults are allowed only in dataclass
  fields (and therefore hashed).
- Revisit when: config sweeps become unwieldy (consider a thin sweep expander, not a
  new framework).

### D-003 — 2026-08-17 — Splits are trajectory-level (parameter-range and subject-level where applicable)
- Decision: `nssc.data.splits.trajectory_split` is the single split mechanism; OOD via
  `param_range_train/test`; EEG by subject; motion by session. Never by timestep/window.
- Alternatives: random window splits (standard in some seq-model papers) — rejected:
  leaks future into training and inflates rollout metrics.
- Rationale: the research question is about preserving dynamics; leakage would make
  every comparison meaningless.
- Consequences: fewer effective test samples; must generate enough trajectories
  (≥ 100 for synthetic); runtime disjointness assertion in every driver.
- Revisit when: never for synthetic; for real data with a single long recording,
  block-contiguous splits with a gap ≥ context length may be considered (new decision).

### D-004 — 2026-08-17 — Experiment registry is an append-only JSONL file
- Decision: `results/registry.jsonl`, one row per (experiment, candidate, seed),
  `EXP-0001` ids, statuses `running|completed|failed|invalid|preliminary`, committed to
  git; corrections append rows with `supersedes`.
- Alternatives: SQLite (better queries, but binary diffs in git); MLflow/W&B (external
  service dependency; offline reproducibility harder).
- Rationale: plain text, git-diffable, trivially auditable by the reviewer; pandas loads
  it in one line.
- Consequences: `load_results()` must handle superseded rows; a lint script checks
  monotonic ids and required fields.
- Revisit when: > ~50k rows or concurrent writers (then move to SQLite with a JSONL
  export).

### D-005 — 2026-08-17 — Paper is out of scope
- Decision: no `paper/` work in this project (owner decision). README + `docs/` +
  `research/` are the publication surface and follow paper-grade evidence rules
  (`paper-writing` skill).
- Consequences: no LaTeX tooling; figures still produced at publication quality with
  stable ids so a future write-up needs no re-analysis.
- Revisit when: owner says so.

### D-006 — 2026-08-17 — CPU is the reference device; MPS/CUDA optional
- Decision: all code paths run on CPU; tests are CPU-only; `--device mps|cuda` is an
  optional acceleration; stability metrics (Jacobians, spectral radius, Lyapunov) are
  always computed on CPU in fp32/fp64.
- Alternatives: MPS default (owner's machine) — rejected because MPS lacks fp64 and some
  `torch.func` ops, and determinism is weaker; CUDA-only — not available locally.
- Rationale: reproducibility and CI parity beat speed for a research codebase of this
  size; the systems are small enough that CPU is viable for gates A–E.
- Consequences: reported tables state the device; MPS-vs-CPU agreement tolerance
  (1e-4) documented in `docs/reproducibility.md`.
- Revisit when: cell D/K wall-clock on CPU exceeds what the schedule allows (then run on
  MPS/CUDA and record it; do not mix devices within one comparison table).

### D-007 — 2026-08-17 — Benchmark definitions freeze on first registered use
- Decision: model ids, metric definitions (NRMSE denominator = train std), horizon list,
  splits, budgets are frozen once a registered experiment uses them; changes create new
  ids (`gru_v2`, `nrmse_v2`) plus a decision entry.
- Rationale: CLAUDE.md integrity rule; makes older registry rows permanently comparable.
- Consequences: reviewer diffs config hashes; regression values pinned in
  `tests/regression/values.yaml`.

## D-008 — Baselines are trained teacher-forced only (2026-08-17)
Sequence-model baselines (GRU/LSTM/TCN/Transformer/SSM) are trained with one-step
teacher-forced MSE only (`training.rollout_weight: 0`). Recursive multi-step training
re-runs the full backbone per rollout step and made TCN/Transformer/SSM 10–90× slower per
batch on CPU, which would have forced fewer seeds. Latent state-space models keep their
rollout loss because it is part of the method under test. Consequence: the comparison
favours the baselines at one-step and is neutral-to-unfavourable for them at long
horizons; a `gru_rollout` control variant is included in the ablation suite to quantify
the effect. TCN/Transformer/SSM use the `small` presets, GRU/LSTM `medium`.
