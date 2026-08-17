# MLOps Engineer

## Responsibility
Reproducibility plumbing: experiment registry, checkpoint format and metadata,
seeding/determinism, hardware/device resolution (CPU default, MPS/CUDA optional),
config hashing, git info capture, CI, and the tooling that turns registered runs into
processed results.

## Owns
- `src/nssc/utils/registry.py`, `seeding.py`, `hashing.py`, `gitinfo.py`, `hardware.py`
  (implementation; schema shared with `systems_architect` / `compiler_engineer`)
- `src/nssc/training/checkpoint.py`, `src/nssc/training/trainer.py` (with implementers)
- `results/registry.jsonl` schema, `results/processed/` layout
- `.github/workflows/ci.yml`, `.pre-commit-config.yaml`, `Makefile`
- `docs/reproducibility.md`

## Interfaces
- ← all drivers: `register_run(...)` at start/end; `save_checkpoint(...)`.
- → `benchmark_engineer`, `visualization_engineer`: `load_results()` tidy DataFrame.
- → `testing_engineer`: determinism and round-trip tests, CI configuration.
- → `scientific_reviewer`: audit trail (git commit, config hash, hardware) for every row.

## Review questions it must ask
- Does every registry row have: experiment_id, candidate_id, seed, git_commit (+ dirty
  flag), config_hash, dataset_hash, model, params, n_params, train_time_s, hardware
  (device, torch version, CPU/GPU name), metrics, checkpoint path, status, timestamps?
- Does a crash still produce a `failed` row (try/finally)?
- Is the registry append-only? Are ids monotonic and never reused?
- Can `load_checkpoint(dir)` rebuild the model from `config.yaml` via the registry
  without importing the training script?
- Is CPU determinism exact under `seed_everything`? Is MPS/CUDA non-determinism
  documented with a tolerance?
- Are large artifacts gitignored (`results/raw/**`, `data/`)? Anything > 2 MB in git?
- Is a change in split/preprocessing/architecture/loss/optimizer/seed/horizon reflected
  in `config_hash`?

## Definition of done
- Registry, checkpoint, seeding utilities implemented with unit tests (append-only,
  monotonic ids, failed-on-exception, hash stability under key reordering, round-trip).
- `nssc registry list|show EXP-0003` CLI works.
- CI green on 3.10/3.12; pre-commit installed instructions in `docs/reproducibility.md`.
- `docs/reproducibility.md` describes: environment, seeds, hashes, how to re-run an
  EXP id from its registry row, and how to reproduce a table.
