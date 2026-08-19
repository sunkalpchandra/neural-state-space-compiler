# Failures

Failed experiments, broken runs, and process failures are scientific evidence. They are
never deleted from the registry (status `failed` or `invalid`) and each gets an entry
here. Include negative results that were *not* crashes: hypotheses that were falsified
belong in `experiment_log.md`, but any *waste* (wrong config, leakage found, wasted
compute) belongs here so it is not repeated.

## Entry template

```
### YYYY-MM-DD — <EXP-xxxx or "process"> — <short title>
- Category: crash | NaN/divergence | leakage | protocol error | wrong config | infra | tooling | integrity
- What happened: <facts, error message / metric evidence>
- Root cause: <as far as known>
- Cost: <wall-clock, seeds affected>
- Detected by: <test / reviewer / manual inspection / registry audit>
- Fix / prevention: <code change, new test, new checklist item; commit sha>
- Registry status set to: failed | invalid (with note)
- Do not: <the one-line lesson>
```

---

_No entries yet._

## F-001 — 2026-08-17 — Scorer: exact-zero best term dominated the ranking
- Where: `nssc.compiler.scorer` (log-ratio normalisation), Lorenz-63 compile screen stage.
- Symptom: `pca+linear@d3` (PCA d=3 on D=3 → reconstruction NRMSE ≈ 1e-9) ranked **1st** with
  50-step rollout NRMSE 1.001 (useless), because every other candidate's recon term became
  log(0.02 / 1e-9) ≈ 17.
- Root cause: log-ratio to the pool minimum has no floor.
- Fix: `error_floor` (default 0.01 NRMSE) inside the log-ratio; regression test
  `tests/unit/test_scorer.py::test_exact_zero_reconstruction_does_not_dominate`.
- Consequence: the screen stage was re-ranked from the cached runs (no training repeated);
  the one fine-stage run started under the buggy ranking stays in the registry (EXP kept).

## F-002 — 2026-08-17 — Explicit latent_dims lists were clipped to obs_dim
- `resolve_latent_dims` clipped explicit lists to `≤ obs_dim`, so the first Lorenz-63 compile
  screened only d∈{2,3} (42 candidates) instead of {2,3,4,8} (84). Fixed to clip only `auto`;
  the running compile picked up the missing candidates after restart (cached runs reused).

## F-003 — 2026-08-17 — Host slept during background runs
- 7 h wall-clock, 48 min CPU: the laptop idle-slept. All long jobs now run under `caffeinate -i`.

## F-004 — 2026-08-17 — Suite resume re-ran baselines after a restart
- `run_suite` hashed the unresolved baseline config while `run_baseline_experiment` registers the
  hash of the resolved one (size preset expanded), so completed baseline runs were not found on
  resume and were repeated (duplicate persistence/GRU rows EXP-0245..0249 remain in the registry).
- Fix: `baseline_config_hash` / `run_config_hash` helpers used by the runner; regression test.

## F-005 — 2026-08-18 — Experiment-id collisions under concurrent runs
- Two ids (EXP-0139, EXP-0245) were handed to more than one run because parallel compile/benchmark
  processes read `next_id()` from the same ledger before either appended. 649 ids, 2 collided.
- Fix: `ExperimentRegistry.register` now takes an exclusive `flock` on `results/registry.jsonl.lock`
  around read-max-id + append (`tests/unit/test_utils_registry_experiments.py::test_concurrent_register_never_collides`).
- The two collided ids are **left in the ledger** (integrity rule: nothing is deleted). Any analysis
  that groups by `experiment_id` should therefore group by `(experiment_id, config_hash, seed)` for
  rows written before this fix; `nssc.evaluation.aggregate` groups by tags + seed and is unaffected.

## F-006 — 2026-08-18 — The device a run used was never recorded
- `hardware_info()` recorded *capabilities* (cuda/mps available) but not the device actually used, so
  a CPU run and an MPS run were indistinguishable in the ledger. Fixed: `register(..., device=...)`
  stores `hardware['device']`; both `run_experiment` and `run_baseline_experiment` pass it.
- Rows written before this fix have no `hardware['device']`; all of them were `--device cpu` runs
  launched by `scripts/dev/detach.sh` / `pipeline_*.sh`, except the EEG smoke run noted in F-00x.
