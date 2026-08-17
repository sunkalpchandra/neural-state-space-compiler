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
