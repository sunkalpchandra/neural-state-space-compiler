# Experiment log

Chronological. One entry per experiment (or per meaningful partial result). Facts and
EXP ids only; interpretation is labeled as such. Never delete entries; superseded entries
get a `Superseded by:` line.

## Entry template

```
### YYYY-MM-DD — EXP-xxxx[, EXP-yyyy] — <short title>
- Hypothesis / cell: Hn / <A–L>
- Config: configs/experiments/<name>.yaml (config_hash <hash>), dataset <name> (<dataset_hash>)
- Git commit: <sha> (clean)
- Seeds: 0–4 (or: preliminary, seeds 0–1)
- Hardware / wall-clock: <device>, <h:mm> total
- Setup: what varied, what was held fixed, budgets, modes, horizons, context length
- Result (facts): key numbers as mean ± std (n=5), CI where used, mode + horizon labeled;
  diverged fraction; table/figure ids (results/tables/..., F3)
- Interpretation: supported / not supported / mixed / inconclusive — and why, in one paragraph
- Hypothesis status change: Hn: <old> → <new> (or none)
- Follow-ups / open questions added: <ids in open_questions.md>
- Reviewer: <accept | accept with changes | reject> — <one line>
- Failures during this experiment: <link to failures.md entry or none>
```

---

### 2026-08-17 — EXP-0001… — Lorenz-63 compiler run (in progress)
- Hypothesis / cell: H1 precondition, H2/H5 (selection), cell C
- Config: configs/compiler/lorenz63.yaml (84 candidates: d∈{2,3,4,8} × {pca,linear,mlp,tcn,gru} × {linear,residual_mlp,koopman,neural_ode(1 substep),ssm}); stages screen(12 ep, cap 50 batches)→fine(50 ep, top 4)→final(120 ep, seeds 0–2)
- Hardware: Apple Silicon CPU (2 threads), background job; log results/logs/compile_lorenz63.log
- Setup note: neural ODE substeps reduced 4→1 after the first screening runs took ~6 min each (restart resumed the other 8 completed runs)
- Result: pending → results/compile/lorenz63/compile_report.md
- Reviewer: pending

### 2026-08-17 — Van der Pol compiler run (in progress)
- Config: configs/compiler/vanderpol.yaml (d∈{2,3,4}, same encoder/dynamics pool) → results/compile/vanderpol

### 2026-08-17 — EXP-0010… — synthetic_core benchmark suite (in progress)
- Hypothesis / cell: H1 (primary), cell D
- Config: configs/experiments/benchmarks/synthetic_core.yaml — 3 datasets × (4 latent models + 6 baselines) × 5 seeds = 150 runs, 40 epochs, capped 60 batches/epoch, baselines teacher-forced only (D-008)
- Early observation (facts, seeds 0–4 done for lorenz63/linae_linear_d3): linear AE + linear dynamics on Lorenz-63 diverges in recursive rollout (test NRMSE@50 = 20.5 ± 31.2, NRMSE@250 ≈ 1.5e17) while its one-step NRMSE is 0.15 ± 0.05 — the canonical "good one-step, useless dynamics" failure the multi-objective score is designed to catch.
- Result: pending → results/tables/synthetic_core.md

