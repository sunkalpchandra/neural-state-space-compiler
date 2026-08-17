# Reproducibility

## Environment
- Python ≥ 3.10 (CI: 3.10, 3.12), PyTorch ≥ 2.1, CPU is the reference device. MPS/CUDA
  are optional accelerations (`--device`); numbers reported in tables state the device.
- `make install` (`pip install -e ".[dev]"`); optional extras `.[eeg]` (mne),
  `.[dashboard]`.
- `pre-commit install` for ruff + hygiene hooks. `make test-fast` before every push.
- Record of the exact environment for a run: `metadata.json` in the checkpoint dir and
  the `hardware` field of the registry row (`torch.__version__`, device name, CPU
  model, `platform`).

## Sources of randomness and how they are pinned
| source | control |
|---|---|
| dataset generation (initial conditions, parameters, obs map, noise) | `DatasetConfig.seed_offsets` + dataset seed; cached by `dataset_hash` |
| split assignment | `trajectory_split(n_traj, spec, seed)` deterministic |
| model init, dropout, data order | `seed_everything(seed)` per run; seeded `torch.Generator` in DataLoader |
| candidate enumeration order | sorted by `candidate_id` |
| bootstrap CIs | `numpy.random.default_rng(0)` |
| torch nondeterministic kernels | `use_deterministic_algorithms(True, warn_only=True)`; CPU exact; MPS/CUDA agreement to 1e-4 documented per run |

## Identifiers
- `experiment_id` (`EXP-0001` …): allocated by the registry, never reused.
- `config_hash`: sha256 of the resolved config (all defaults filled, keys sorted); any
  change to split, preprocessing, architecture, loss, optimizer, seed list, horizon
  list, or score weights changes it. Print with `nssc config resolve <yaml>`.
- `dataset_hash`: hash of the resolved `DatasetConfig` + generator code version string.
- `git_commit` + `git_dirty` flag: recorded per run; reported results must come from
  clean commits.

## Re-running an experiment
    nssc registry show EXP-0012                       # prints config path, hash, seeds, command
    nssc rerun EXP-0012 --seed 3                      # re-executes with the stored resolved config
    scripts/reproduce.sh EXP-0012                     # rerun all seeds and diff metrics vs registry (tolerance in tests/regression/values.yaml)
The rerun refuses to proceed if the current `config_hash` differs from the stored one
(unless `--allow-hash-mismatch`, which registers a *new* experiment id).

## Reproducing tables and figures
    scripts/make_tables.py  --registry results/registry.jsonl --out results/tables
    scripts/make_figures.py --fig all
Both read only the registry / processed dirs, are deterministic, and fail if referenced
runs are missing.

## What is committed vs. generated
- Committed: configs, code, tests, `results/registry.jsonl`, `results/processed/**`
  (metrics.json, compile reports, small), `results/tables/`, `results/figures/`
  (png/pdf), `research/`, `docs/`.
- Gitignored: `results/raw/**` checkpoints (`*.pt`, `*.npz`), `data/cache`, `data/raw`.
  Checkpoints for reported results are archived out-of-repo (path recorded in
  `research/experiment_log.md`) — reproduce with `scripts/reproduce.sh` instead.

## Protocol invariants (must never change silently)
Trajectory-level splits; validation-only selection; seeds 0–4; horizons
1,5,10,25,50,100,250,500; mode labels; NRMSE denominator = train std; budget parity across
compared models. Every one of these is a config field covered by `config_hash`; a change
requires a `research/decisions.md` entry.

## Known non-reproducibility
- MPS: `float64` unsupported, some `torch.func` ops fall back; stability metrics are
  therefore always computed on CPU. Training on MPS vs CPU can differ at ~1e-4 relative;
  reported tables state the device.
- Wall-clock and latency depend on hardware; only comparisons measured on the same
  machine in the same session are meaningful (recorded in `hardware`).
