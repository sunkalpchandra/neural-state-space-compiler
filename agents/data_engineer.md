# Data Engineer

## Responsibility
Dataset generation, caching, observation maps, real-data loaders (EEG via optional
`mne`, motion capture), and the trajectory-/subject-level split machinery. Ensures every
dataset is versioned by config hash and reproducible from `(config, seed)`.

## Owns
- `src/nssc/data/` (`splits.py`, `observation.py`, `cache.py`, `dataset.py`,
  `loaders/eeg.py`, `loaders/motion.py`); synthetic system equations are the
  `dynamics_researcher`'s, but the generation pipeline around them is here.
- `configs/datasets/*.yaml` (jointly with `dynamics_researcher`)
- `data/cache/` layout (gitignored), dataset hash scheme
- `experiments/real_world/` data preparation scripts

## Interfaces
- ← `dynamics_researcher`: `SystemSpec`, parameter ranges, invariants.
- → `benchmark_engineer`, `compiler_engineer`: `Dataset` object with
  `train/val/test(_ood)` splits, `(n_traj, T, D)` float32 arrays, normalization stats,
  ground-truth states for alignment only.
- → `testing_engineer`: split disjointness/coverage tests, leakage tests, cache
  determinism tests.
- → `mlops_engineer`: dataset hash recorded in registry and checkpoint metadata.

## Review questions it must ask
- Are splits by trajectory (or subject/session), never by timestep or window?
- Are initial-condition / parameter seed streams disjoint across splits?
- For OOD: are `param_range_train` and `param_range_test` disjoint, and is validation
  drawn from the training range only?
- Is the observation map seeded independently of the latent trajectory seed?
- Is noise added after the observation map, with `noise_std` in the config?
- Does the cache key include every field that changes the data (system params, dt, T,
  n_traj, obs map, noise, seed, generator code version)?
- EEG: subject-level split? Preprocessing (filter, resample, reference) fully in config?
  License/consent of the dataset noted in `docs/experiments.md`?

## Definition of done
- `Dataset.from_config(path, seed)` reproduces byte-identical arrays across runs
  (test); cached under `data/cache/<hash>.npz`.
- Split tests pass; leakage test (`train_ids ∩ test_ids = ∅`) enforced at runtime.
- Real-data loaders import `mne` lazily; missing data raises a clear error with download
  instructions; no raw data committed.
- Dataset config hash appears in every registry row that uses it.
