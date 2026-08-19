# Protocol-v1 results (superseded 2026-08-18, kept as evidence)

These tables and the Lorenz-63 compile report were produced **before** the fix in
`research/failures.md` F-007 (validation was evaluated at the ramping curriculum horizon, so
best-checkpoint selection could restore an under-trained epoch). They are kept verbatim because
failed/superseded experiments are evidence, not garbage.

* Latent-model rows here are a **lower bound**: 289/639 latent runs restored a pre-ramp checkpoint.
* Baseline rows are unaffected (`BaselineTrainer` always validated at the full horizon) and are
  reused unchanged by the v2 re-runs — the registry recognises them by config hash.
* The v2 replacements live in `results/tables/` and are labelled `protocol v2`.
