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


### 2026-08-17 — EXP-0044…EXP-0249 — synthetic_core, Lorenz-63 rows complete (n=5 seeds)
- Hypothesis / cell: H1 (primary), cell D. Config: configs/experiments/benchmarks/synthetic_core.yaml
  (40 epochs, ≤60 batches/epoch, context 20, recursive rollout on test split, baselines TF-only per D-008).
- Result (facts, test recursive NRMSE mean ± std over seeds 0–4; results/tables/synthetic_core.md):

  | model | params | @50 | @100 | @250 |
  |---|---|---|---|---|
  | mlpae_resmlp_d3 (latent, d=3) | 13,833 | 0.0076 ± 0.0010 | 0.017 | 0.075 |
  | lstm_medium | 200,579 | 0.0077 ± 0.0011 | 0.015 | 0.065 |
  | gru_medium | 150,531 | 0.0125 ± 0.0018 | 0.034 | 0.137 |
  | tcn_small | 16,867 | 0.0236 ± 0.0056 | 0.045 | 0.220 |
  | mlpae_mlp_d3 (non-residual) | 13,833 | 0.096 ± 0.058 | 0.306 | 0.664 |
  | ssm_small | 14,691 | 0.098 ± 0.040 | — | — |
  | transformer_small | 42,083 | 0.190 ± 0.044 | 0.822 | 1.253 |
  | pca_linear_d3 | 9 | 1.010 ± 0.012 | 1.045 | 1.128 |
  | persistence | 0 | 1.295 | 1.427 | 1.532 |
  | linae_linear_d3 | 33 | diverged (20 ± 31) | diverged | diverged |

  Paired vs mlpae_resmlp_d3 (NRMSE@50, same seeds, n=5): gru p_t=0.005, tcn 0.003, transformer 0.001,
  lstm 0.85 (no difference); Wilcoxon minimum p=0.062 at n=5 (reported, not over-interpreted).
- Interpretation: on Lorenz-63 with identity observations, a 13.8k-parameter latent state-space
  model (MLP AE d=3 + residual-MLP dynamics, trained with a 20-step rollout loss) matches the
  best large sequence model (LSTM, 14.5× more parameters) and beats GRU/TCN/Transformer/SSM at
  50–250 steps. The residual parameterisation matters (non-residual MLP dynamics is 12× worse
  at @50 and unstable across seeds). Linear latent dynamics diverge (chaos is not linear).
  Caveat: baselines are teacher-forced only (D-008); the rollout-loss control suite is queued.
- Hypothesis status: H1 — *supported on Lorenz-63 (identity obs)*; pending vanderpol, high-dim.
- Figures: results/figures/suites/synthetic_core/horizon_curve_lorenz63.png, pareto_lorenz63.png.

### 2026-08-18 — Lorenz-63 compiler run: screen + fine stages (interim, final stage running)
- Screen (84 candidates, 12 epochs, 1 seed, val): 30 survivors; top-3 gru+residual_mlp@d4, gru+residual_mlp@d3,
  mlp+residual_mlp@d3. Every pca+*, *+linear and most *+ssm candidates pruned (rollout NRMSE ≈ 1 or diverged).
- Fine (30 candidates, 50 epochs, 1 seed, val recursive NRMSE@250 / params / verdict):
  1. linear+koopman@d3 — 0.016 / 5,169 / stable (J=0.17)
  2. linear+residual_mlp@d3 — 0.020 / 4,635 / stable
  3. linear+residual_mlp@d4 — 0.023 / 4,771 / stable
  4. mlp+residual_mlp@d3 — 0.023 / 13,833 / stable   ← the hand-picked reference architecture
  5. tcn+residual_mlp@d3 — 0.026 / 75,721 (pruned; 5.5× params for no gain)
- Observation: with identity observations a *linear* encoder suffices (the state is observed) and the
  compiler prefers the cheapest dynamics family that keeps long-horizon error — Koopman lifting
  (m=12) edges out residual MLP. Selection is on validation only; test numbers come from Experiment L.
- Next: final stage (top-4 × seeds 0–2 × 120 epochs) → compile_report.md.
