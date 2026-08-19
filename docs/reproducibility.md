# Reproducibility

What this repo actually records, what it does not, and what a reader can and cannot
re-derive from a fresh clone. Verified against the code on 2026-08-19; where behaviour is
planned but absent it is labelled **not implemented** rather than described as if it worked.

## Environment
- Python ≥ 3.10 (`pyproject.toml:11`; CI matrix 3.10 and 3.12, `.github/workflows/ci.yml`),
  PyTorch ≥ 2.1 (`pyproject.toml:16`).
- `make install` → `pip install -e ".[dev]"`. Optional extras: `.[eeg]` (mne ≥ 1.6),
  `.[dashboard]` (fastapi/uvicorn/httpx).
- `pre-commit install` → ruff + ruff-format, end-of-file-fixer, trailing-whitespace,
  check-yaml, check-added-large-files (max 2000 kB) (`.pre-commit-config.yaml`).
- `make test-fast` (= `pytest -q -m "not slow"`) before every push; CI runs
  `ruff check src tests scripts` then the same pytest selection.
- **CPU is the reference device (D-006) but it is not the default.** `default_device()`
  (`src/nssc/utils/env.py:35`) picks CUDA → MPS → CPU. To get the reference device you must pass
  `--device cpu` (available on `train`, `evaluate`, `compile`, `benchmark`) or set
  `training.device` in the run config. `scripts/reproduce.sh` and the batch driver
  `scripts/dev/pipeline_v2.sh` pass `--device cpu` on every long run.
- There is no lockfile and no `pip freeze` capture. The only dependency versions recorded per
  run are `python`, `torch` and `numpy` inside the `hardware` dict.

## What is recorded per run

Every run writes two rows to the append-only ledger `results/registry.jsonl`: a `running` row
at registration and a `completed`/`failed` row at the end. `ExperimentRegistry.records()`
(`src/nssc/utils/experiment_registry.py:76`) keeps the latest row per `experiment_id`, so the
ledger has roughly twice as many lines as runs; a killed process leaves only its `running` row.
Rows are never rewritten and never deleted — a failed run keeps `status='failed'` with the
exception text in `notes`, and gets an entry in `research/failures.md` (CLAUDE.md integrity
rule). Corrections are appended, not applied in place.

**Ledger snapshot, 2026-08-19** (the protocol-v2 re-run is in progress, so these numbers grow;
recompute rather than quote them):

    python - <<'EOF'
    import json, collections
    rows = [json.loads(l) for l in open("results/registry.jsonl") if l.strip()]
    last = {r["experiment_id"]: r for r in rows}
    print("rows", len(rows), "| ids", len(last),
          "| status", collections.Counter(r["status"] for r in last.values()),
          "| dirty", sum(r["git_commit"].endswith("-dirty") for r in rows),
          "| distinct SHAs", len({r["git_commit"].split("-dirty")[0] for r in rows}),
          "| rows with hardware['device']", sum("device" in r["hardware"] for r in rows))
    EOF

At the time of writing: 1334 rows / 665 ids (654 completed, 10 running, 1 failed); 1334 of 1334
rows `-dirty` across 159 distinct SHAs; 29 rows carry `hardware['device']`.

`ExperimentRecord` (`src/nssc/utils/experiment_registry.py:47-68`) — these are the only fields:

| field | content |
|---|---|
| `experiment_id` | `EXP-0001`… allocated by `next_id()` = max existing + 1, under an exclusive lock |
| `git_commit` | `git rev-parse HEAD`, with `-dirty` appended when `git status --porcelain` is non-empty (`utils/env.py:14-32`). One string — there is **no separate `git_dirty` field** |
| `config_hash` | see *Identifiers* below |
| `dataset`, `model`, `seed` | `dataset` is the system/source name; `model` is `model_name(cfg)` e.g. `mlp+residual_mlp@d3` or `baseline:gru` |
| `status` | `running` \| `completed` \| `failed` |
| `metrics` | the `summarize()` subset only (`experiment.py:179-199`): `<split>/<key>` for the keys in `SUMMARY_KEYS`, plus `train/best_val_loss`, `train/epochs_run`, `train/time_s`. Curves and the full metric dict stay on disk |
| `checkpoint` | path to the checkpoint **directory**, or `null` |
| `config` | the full run config as executed, with the dataset `_file` already merged in |
| `param_count`, `train_time_s` | filled on completion |
| `hardware` | `hardware_info()` (`utils/env.py:43`): `platform`, `machine`, `python`, `torch`, `numpy`, `cuda` (bool), `mps` (bool), `gpu` (only when CUDA is present) — plus `device` when the caller passes it (`register(..., device=...)`; `experiment.py:111`, `baselines/run.py:136`) |
| `created_at`, `updated_at` | unix timestamps |
| `tags`, `notes` | tags such as `suite:synthetic_core`, `ds:lorenz63`, `m:gru_medium`, `stage:screen`, `cand:<id>`; `notes` carries the exception text of failed runs |

On disk, each run's `output_dir` holds `metrics.json` (the *full* metrics, including
`curves.*`, and — for runs written after the 2026-08-18 change — `config/protocol_version`
and `config/ignored_keys`), `history.json`,
`error.json` on failure, and `checkpoint/{model.pt, config.yaml, metadata.json}`
(`training/checkpoint.py:23-30`).

## What is **not** recorded
- **The resolved dataclass defaults.** `config_hash` is taken over the run config *as written*
  (`experiment.py:86-92`), not over `asdict(TrainerConfig)/asdict(EvalConfig)`. Two consequences,
  both verified: writing a value that equals the dataclass default (`weight_decay: 1e-05`,
  `amp: false`) *changes* the hash, and changing a default inside `TrainerConfig` changes what
  runs without changing any hash. D-002 promises the opposite ("code defaults are allowed only in
  dataclass fields and therefore hashed") — **that decision is not implemented** (review R-18).
- **Unknown config keys are dropped, not rejected.** `_dc` (`experiment.py:48-55`) keeps only keys
  that are dataclass fields and returns the rest; they are written to `metrics.json` under
  `config/ignored_keys` but do **not** reach the registry row, because `summarize()` filters them
  out. A typo in `training:` therefore changes `config_hash` while changing nothing about the run.
- **A dataset hash.** `grep -rn dataset_hash src/` finds nothing. The builder does compute
  `metadata['version'] = stable_hash(resolved dataset config)` (`data/builder.py:106`), but that
  value is never stored in the ledger, the checkpoint metadata or the compile report. What
  `config_hash` covers is only the dataset keys present in the YAML; `resolve_config` afterwards
  fills `substeps`, `ic_scale` and `kuramoto_sin_cos` from code defaults, outside the hash.
- **Dataset cache identity.** The EEG BCI loader caches to
  `data/cache/eegbci/eegbci_<stable_hash(resolved cfg)>.npz` (`data/real/eegbci.py:132-134`) and
  reuses that file whenever it exists. Nothing checksums the downloaded raw EDF files, and
  `data/cache/` is gitignored, so a cache entry's only provenance is its config hash.
- **Environment inside the checkpoint.** `checkpoint/metadata.json` contains exactly
  `experiment_id`, `seed`, `norm_stats`, `dataset`, `metrics_summary` (`experiment.py:158-160`) —
  no git commit, no torch version, no device. The module docstring of `training/checkpoint.py`
  mentions "git commit" as an example of free-form metadata; nothing writes it.
- **The device, in every row written before 2026-08-18.** `hardware['device']` was only added
  then (`research/failures.md` F-006), so the overwhelming majority of the ledger — every row
  behind the currently published tables — records CPU/MPS *availability* but not which device the
  run used. Only rows written by the current code have it — see the ledger snapshot above.
- **The protocol version, in the ledger.** `metrics['config/protocol_version']` and
  `metrics['config/ignored_keys']` are written to the on-disk `metrics.json` (`experiment.py:148-149`)
  but `summarize()` keeps only `SUMMARY_KEYS`, so they never reach a registry row — no row in the
  ledger carries any `config/*` key. Telling a v1 row from a v2 row therefore means either
  reading the run's `metrics.json` or recomputing `run_config_hash` on the stored config.
- **The device in tables.** No table generator emits it: `grep -n device src/nssc/evaluation/tables.py
  src/nssc/evaluation/aggregate.py` is empty. D-006's "reported tables state the device" is
  **not implemented**.

## Sources of randomness and how they are pinned
| source | control |
|---|---|
| dataset generation (initial conditions, parameters, observation map, noise, missing mask) | the dataset config's `seed` (default 0, `data/builder.py` `DEFAULTS`), threaded into `system.simulate(..., seed=...)` and `nssc.utils.seeding.rng` |
| split assignment | `trajectory_split(n_traj, fractions, seed)` (`data/splits.py:12`) — a `default_rng(seed)` permutation; `seed`/`fractions` come from the dataset config's `split:` block, default `0` / `(0.7, 0.15, 0.15)` (`data/dataset.py:99-102`). Real loaders may pin `metadata['split_indices']` instead (subject-level for EEG) |
| model init, dropout, batch order | `seed_everything(seed)` at the top of `run_experiment` (`utils/seeding.py:12`) seeds python/numpy/torch |
| bootstrap CIs | `bootstrap_ci(values, n_boot=2000, alpha=0.05, seed=0)` → `default_rng(0)`; called without a seed argument (`evaluation/aggregate.py:44, 83`) |
| Lyapunov tangent vector | `torch.Generator(device="cpu").manual_seed(0)` (`stability/lyapunov.py:21`) — device-independent by construction |
| candidate enumeration order | `itertools.product(latent_dims, encoders, dynamics, hidden)` (`search/space.py:99`); `latent_dims` is sorted, the encoder/dynamics order is the YAML order. Deterministic, but **not** sorted by `candidate_id` |
| torch nondeterministic kernels | `use_deterministic_algorithms(True, warn_only=True)` + `cudnn.deterministic` (`utils/seeding.py:24-27`). `warn_only` means a nondeterministic op degrades to a warning rather than an error |

Caveat on batch order: `make_loaders` builds each `DataLoader` **without** a `generator=`
argument (`data/dataset.py:198-208`), so train shuffling draws from the global torch RNG. It is
reproducible for a fixed (config, seed, code path), but the permutation depends on how much RNG
the preceding model construction consumed — change the architecture and the batch order changes
too. There is no seeded per-loader generator.

## Identifiers
- `experiment_id` (`EXP-0001`…): allocated by `next_id()` and appended under an exclusive
  `flock` on `results/registry.jsonl.lock` (`utils/experiment_registry.py:30-44, 107-132`), so
  concurrent processes cannot be handed the same id. Regression test:
  `tests/unit/test_utils_registry_experiments.py::test_concurrent_register_never_collides`.
  Ids are never reused and rows are never deleted. Two ids allocated before this fix *are*
  duplicated — see *Known caveats*.
- `config_hash`: `stable_hash` = sha256 of the canonical JSON of the object, truncated to 12 hex
  characters (`utils/hashing.py:28`). For latent runs the hashed payload is the run config minus
  `output_dir` and `tags`, with the dataset `_file` merged in and `_protocol: PROTOCOL_VERSION`
  added (`experiment.py:86-92`). For baselines it is `baseline_config_hash`
  (`baselines/run.py:104-112`): dataset and model resolved, `output_dir`/`tags` excluded, and
  **no** protocol key.
- `PROTOCOL_VERSION` (`experiment.py:38`) is currently `2`. It encodes the semantics of a latent
  run and is bumped deliberately so that older rows cannot be reused for the new protocol; v2
  (2026-08-18) evaluates validation loss at the full `rollout_horizon` instead of the current
  curriculum horizon (`TrainerConfig.val_fixed_horizon`, default `True`, `training/trainer.py:36`;
  `research/failures.md` F-007). Because baseline hashes omit it, a bump
  invalidates latent-run reuse but not baseline reuse.
- **Not implemented:** `nssc config resolve <yaml>`, `nssc registry show <id>`, `nssc rerun`,
  `--allow-hash-mismatch`, a `dataset_hash`, and a `git_dirty` field. Print a hash from Python
  instead: `python -c "from nssc.experiment import run_config_hash; from nssc.utils.config import
  load_config; print(run_config_hash(load_config('configs/experiments/lorenz63_mlp_resmlp.yaml').to_dict()))"`.

## Re-running an experiment

Commands that exist (`nssc --help`): `profile`, `train`, `evaluate`, `registry`, `compile`,
`benchmark`, `visualize`, `report`, `tables`, `pareto`, `failures`, `smoke`, `dashboard`, `data`.

    nssc registry --status completed --tag suite:synthetic_core --limit 20   # id, status, dataset, model, seed, params, test nrmse@50, commit[:8]
    nssc train --config configs/experiments/lorenz63_mlp_resmlp.yaml --seed 3 --device cpu
    nssc evaluate --experiment EXP-0123 --split test --device cpu            # re-evaluates from the checkpoint on disk

There is no CLI that replays a stored config. The row carries it, so do it in Python:

    from nssc.experiment import run_experiment
    from nssc.utils.experiment_registry import ExperimentRegistry
    rec = ExperimentRegistry().get("EXP-0123")
    run_experiment(dict(rec["config"]) | {"output_dir": "/tmp/rerun"})

This registers a **new** experiment id; it does not overwrite or supersede the original row.

Verified on the smoke run (`results/registry_smoke.jsonl`, EXP-0001): replaying the stored config
reproduced `test/recursive/nrmse@10 = 1.1412205323476385` exactly — **and** produced a different
`config_hash` (`0fcecdc9beb0` stored → `07fd5d1c648a` now), because that row predates the
`PROTOCOL_VERSION` bump. Hashes are not comparable across a protocol bump even when the numbers
are identical; compare metrics, not hashes, across that boundary.

Nothing refuses to run on a hash mismatch — that check is **not implemented**.

## Resuming a compile or a suite

**Compile** (`nssc compile`, resume on by default; `--no-resume` deletes `search_state.json`,
`compiler/compiler.py:155-159`):
1. `SearchState` — a JSON file in the compile output dir, keyed `"<stage>|<candidate_id>|<seed>"`
   (`search/state.py:24`). `StagedSearch._run_one` returns the cached entry whenever its status is
   `completed` or `failed` (`search/staged.py:66-68`). It does **not** check that the checkpoint
   still exists, and it does **not** verify that the config or git commit still match — the state
   file stores neither (`search/state.py`: `runs`, `stages`, `meta`, `created`, `updated` only).
2. Registry reuse (`reuse_registry`, default `True`, `compiler/compiler.py:111`): a candidate with
   no cached entry is looked up by `run_config_hash(cfg)` + seed, and a completed row is reused if
   its `checkpoint` path exists (`search/staged.py:71-81`). The lookup hash is now the canonical
   helper, so it matches what `run_experiment` registers — it hashed the raw config until
   2026-08-18 and silently missed every row after the protocol bump (`research/failures.md` F-008).
   One limit remains: the path check is on the checkpoint *directory*, which `config.yaml` and
   `metadata.json` satisfy on their own, so a run whose `model.pt` is gone still counts as reusable
   and only fails later, at `load_checkpoint`.

**Suites** (`nssc benchmark --suite <name>`): `run_suite` skips a run when
`registry.find_by_hash(hash, seed)` already has a `completed` row (`search/runner.py:95-101`).
The registry path is not overridable from the CLI — only `run_suite(..., registry=...)` in
Python takes one. The `PROTOCOL_VERSION` bump invalidated every pre-v2 latent hash while leaving
baseline hashes untouched, so re-running a suite today retrains the latent models and replays the
baselines from the ledger. Snapshot for `synthetic_core` on 2026-08-19, mid-rerun: 10 of 60 latent
runs and 90 of 90 baseline runs would be skipped. Check before committing hours of compute —
`run_suite` has a dry-run mode that prints one line per run and `skip (done: EXP-…)` for anything
it would replay:

    from nssc.search.runner import run_suite
    run_suite("configs/experiments/benchmarks/synthetic_core.yaml", dry_run=True,
              overrides=["output_dir=/tmp/drycheck"])

The `overrides` argument is not optional in practice: `run_suite` writes `suite_results.json`
into the suite's `output_dir` unconditionally, dry run included (`search/runner.py:118-120`), so
without it a dry run overwrites the committed
`results/raw/benchmarks/<suite>/suite_results.json` with a partial list.

## What `scripts/reproduce.sh` really does

Five targets; anything else prints `unknown target` and exits 1. There is **no** `EXP-id` mode.
The script `cd`s to the repo root and exports `OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}`.

| target | what runs |
|---|---|
| `smoke` (default) | `pytest -q -m "not slow"`; `nssc smoke` (writes to `results/registry_smoke.jsonl`); `nssc compile --config configs/compiler/tiny.yaml --device cpu` |
| `compile` | `nssc compile --config configs/compiler/{lorenz63,vanderpol,lorenz63_highdim}.yaml --device cpu` |
| `ablations` | `nssc compile` for each of the five `configs/compiler/ablations/*.yaml` (not listed in the script's own header) |
| `benchmark` | `nssc benchmark --suite synthetic_core --device cpu`, then `nssc tables --suite synthetic_core --reference mlpae_resmlp_d3` |
| `figures` | `python scripts/generate_report.py` → per-compile figures, per-suite figures + tables, and `results/SUMMARY.md` |

Gaps, all open:
- `compile` and `ablations` do **not** pass `--no-resume`. `search_state.json` is committed and
  `*.pt` is not, so on a fresh clone every cached entry is replayed and then `compile()` calls
  `load_checkpoint(best_run["checkpoint"])` (`compiler/compiler.py:135`) →
  `torch.load(.../model.pt)` → `FileNotFoundError`. To actually reproduce a compile, add
  `--no-resume` (review R-02).
- `benchmark` covers only `synthetic_core`. `compiled_vs_manual`, `ablation_stability_reg`,
  `baseline_rollout_control` and `real_eegbci` have no target, and neither do `nssc pareto`,
  `nssc failures` or the OOD/uncertainty tables that exist under `results/tables/`.
- `scripts/generate_report.py` knows one reference model (`REFERENCE = {"synthetic_core":
  "mlpae_resmlp_d3"}`, line 24), so other suites' tables are rendered without paired tests.
- **Not implemented:** `scripts/make_tables.py`, `scripts/make_figures.py`,
  `tests/regression/values.yaml`. Regression tolerances are inline assertions in
  `tests/regression/test_benchmark_values.py` and `tests/regression/test_known_dynamics_values.py`.

`scripts/reproduce.sh figures` does work from committed JSON alone: suite figures read
`metrics.json` from each run dir and compile figures read `compile_report.json` /
`search_state.json`. Only the per-run "selected model" figure set needs a checkpoint
(`visualization/figures.py:134` loads `model.pt`); it is skipped with a warning when absent.

## What is committed vs. generated

Gitignored (`.gitignore`), i.e. absent from a fresh clone:
`*.pt` **everywhere** (all model weights), `results/raw/**/*.npz|*.npy`, `*.ckpt`, `*.log`,
`results/logs/`, `results/raw/EXP-*/`, `results/raw/smoke/`, `results/registry_smoke.jsonl`,
`data/cache/`, `data/raw/`.

Committed by the run-committing helpers, which use `git add -f` and therefore reach into
gitignored directories:
- `scripts/dev/commit_runs.py` — one commit per run, adding `metrics.json`, `history.json`,
  `error.json`, `checkpoint/config.yaml`, `checkpoint/metadata.json`, `checkpoint/config.json`.
  Weights are never added.
- `scripts/dev/autocommit_runs.sh` — runs `commit_runs.py` on a loop and commits
  `results/registry.jsonl` as a ledger snapshot, then pushes.

Everything else under `results/` is tracked because no ignore rule matches it:
`results/compile/**/{compile_report.json, compile_report.md, compiled_model.yaml,
compiler_config.yaml, profile.json, candidates.json, search_state.json}`, `results/tables/`,
`results/figures/`, `results/SUMMARY.md`, plus `configs/`, `src/`, `tests/`, `research/`, `docs/`.
`results/processed/` exists but is empty — nothing writes to it.

**What gitignoring `*.pt` means for reproduction.** The `model.pt` files (643 on this machine on
2026-08-19; `find results -name model.pt | wc -l`) exist on the machine that ran the experiments
and nowhere else: they are not committed and they are **not archived
out-of-repo** (no archive path appears in `research/experiment_log.md` or anywhere else). So on a
fresh clone:
- every metric, table and figure that reads JSON can be regenerated;
- nothing that needs weights can — `nssc evaluate`, per-run figures, and the final
  `load_checkpoint` step of `nssc compile` all fail;
- **checkpoints must be retrained** (`scripts/reproduce.sh compile --no-resume`,
  `scripts/reproduce.sh benchmark`) before any of those work.

## Protocol invariants (must never change silently)

Stated as the configs and code actually have them:

| invariant | actual value | where |
|---|---|---|
| splits | trajectory-level, default `(0.7, 0.15, 0.15)` at split seed 0; EEG uses explicit `metadata['split_indices']` | `data/splits.py:12`, `data/dataset.py:83-103` (D-003) |
| selection | validation only — early stopping and best-state restore monitor the val loss (`training/trainer.py:180-194`; note it falls back to the *train* loss when a config has no val split), and `MultiObjectiveScorer` ranks on `val/*` keys (`compiler/scorer.py:67`). Test metrics are computed but never selected on | `training/trainer.py`, `compiler/scorer.py` |
| seeds | `[0, 1, 2, 3, 4]` for `synthetic_core`, `compiled_vs_manual`, `ablation_stability_reg`, `baseline_rollout_control`; `[0, 1, 2]` for `real_eegbci`. Compile stages differ by design: `screen` and `fine` use seed `[0]`, `final` uses `[0, 1, 2]` | `configs/experiments/benchmarks/*.yaml`, `configs/compiler/*.yaml` |
| rollout horizons | suites evaluate `[1, 5, 10, 25, 50, 100, 250]` — **not** 500; `real_eegbci` uses `[1, 5, 10, 25, 50, 100]`; the compiler default config uses `[1, 5, 10, 25, 50, 100, 250, 500]` and its screen stage narrows to `[1, 5, 10, 25, 50]` | same files |
| mode labels | latent models emit `recon/`, `teacher_forced/`, `teacher_forced_ctx/` (the same one-step error restricted to positions `t >= context-1`, so context-window baselines and latent models are compared on the positions both can predict) and `recursive/`; baselines emit the same one-step pair plus `recursive/` and, when configured, `direct/` | `evaluation/evaluator.py:85-93`, `baselines/evaluate.py:84-106` |
| NRMSE denominator | train std — data are normalised with train statistics and evaluation passes `sigma = ones` (`experiment.py:131`), so `nrmse` divides by the training standard deviation (`metrics/prediction.py:29-39`) | D-007 |
| budget parity | **not enforced anywhere in code.** Parameter budgets are whatever the suite YAML's model and baseline size presets produce; comparability is a manual property of the config | — |

Seeds, horizons, splits and the model/baseline definitions are config fields and therefore inside
`config_hash`. The NRMSE denominator, the mode labels and budget parity are **code**, not config,
so a change to them would not show up as a hash change — that is what `PROTOCOL_VERSION` and
D-007 (definitions freeze on first registered use) exist to catch. Any change to the rows above
requires a `research/decisions.md` entry.

## Known caveats (honest limits of the record)

- **Every ledger row was written from a dirty working tree.** Every row carries a `-dirty`
  suffix on `git_commit`, across ~160 distinct SHAs (exact counts in the ledger snapshot above).
  This is a direct
  consequence of the commit style: `scripts/dev/autocommit_runs.sh` commits run artifacts and a
  registry snapshot every ~10 minutes *while* suites are executing, so the working tree is
  essentially never clean and the recorded SHA changes from seed to seed inside a single suite.
  **`git_commit` therefore pins the source tree only approximately**; no patch or diff hash is
  recorded, so the exact code state behind a given number is not recoverable from the row alone.
  The earlier policy statement that "reported results must come from clean commits" was never
  met and has been removed rather than left as an aspiration (review R-15, open).
- **Two experiment-id collisions exist in the ledger.** `EXP-0139` and `EXP-0245` were each handed
  to more than one run when parallel compile/benchmark processes read `next_id()` before either
  appended (`research/failures.md` F-005). Fixed on 2026-08-18 by the `flock` in
  `ExperimentRegistry.register`; the collided rows are deliberately **not** deleted (integrity
  rule). Any analysis that groups by `experiment_id` over pre-fix rows should group by
  `(experiment_id, config_hash, seed)` instead; `nssc.evaluation.aggregate` groups by tags and
  seed and is unaffected.
- **A handful of ids are stuck at `running`** (10 in the snapshot above, plus whatever is in
  flight).
  Those are processes killed or crashed outside `run_experiment`'s `try` block; they are not
  failures, carry no metrics, and are excluded by every `status == "completed"` filter.
- **Device agreement between CPU and MPS has never been measured.** The previous claim of "~1e-4
  relative agreement" is unsupported: there is no test, no recorded comparison, and no table that
  states its device. What *is* true is that MPS lacks float64, so the Jacobian eigendecomposition
  is forced to CPU float64 (`stability/spectral.py:13`). The Lyapunov estimate and the
  norm-growth diagnostic run on whatever device the model is on
  (`stability/lyapunov.py`, `stability/analysis.py`) — only their random tangent vector is
  device-independent. Treat cross-device numbers as uncompared until someone measures them.
- **`warn_only=True`** means a nondeterministic kernel produces a warning, not an error; runs can
  silently lose bitwise determinism on an accelerator.
- **Wall-clock and latency are machine- and thread-dependent** and the launchers disagree:
  `scripts/reproduce.sh` sets `OMP_NUM_THREADS=4`, `scripts/dev/detach.sh` sets
  `OMP_NUM_THREADS=2` / `MKL_NUM_THREADS=2`. Only timings measured on the same machine, in the
  same session, under the same thread settings are comparable — and the thread count is not
  recorded in the ledger.
