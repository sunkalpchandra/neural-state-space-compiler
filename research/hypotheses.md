# Hypotheses

Research question: *Can we automatically compile high-dimensional temporal observations
into a low-dimensional, structured, predictive state-space representation that preserves
the underlying dynamics better than simply fitting a large sequence model?*

Each hypothesis has a falsification criterion fixed **before** its experiments run.
Statuses: `untested | preliminary | supported | not supported | mixed`. Status changes
only via an entry in `experiment_log.md` citing EXP ids with seeds 0–4.

| id | status | matrix cells |
|----|--------|--------------|
| H1 | partially supported (Lorenz-63 identity obs, hand-picked latent model vs 6 baselines, n=5; see experiment_log 2026-08-17) — compiler-selected model pending | B, C, D, J |
| H2 | untested | E |
| H3 | untested | F |
| H4 | untested | G |
| H5 | untested | H, L |
| H6 | untested | I |
| H7 | untested | K |

---

## H1 — Compiled low-dimensional latent SSM vs. large sequence models
**Claim.** A compiled model with `d ≪ D` (d within 2× the estimated intrinsic dimension)
matches or exceeds the long-horizon recursive rollout accuracy of directly fitted large
sequence models (GRU/LSTM/TCN/Transformer/SSM at their best validation-selected size)
while using fewer parameters and lower per-step latency.
**Operationalisation.** Recursive mode, horizons H ∈ {100, 250} (primary), NRMSE and
VPT; systems S1–S9; seeds 0–4; budget parity.
**Falsified if** on a majority of systems the best large sequence model has lower
recursive NRMSE@100 than the compiled model with a paired-by-seed 95 % bootstrap CI of the
difference excluding zero, *or* the compiled model needs ≥ the parameter count of the
best sequence model to match it. Partial outcome (wins on low-n systems, loses on
Lorenz-96 / Gray–Scott) → `mixed`, reported per system.

## H2 — Multi-objective selection vs. validation-MSE selection
**Claim.** Selecting candidates by `J` (recon + one-step + rollout + complexity +
instability) yields models with lower long-horizon error and fewer diverging rollouts
than selecting by validation one-step MSE, from the same candidate pool.
**Falsified if** across S3, S5, S6, S7, S9 the val-MSE-selected models have equal or
lower recursive NRMSE@250 and equal or lower diverged fraction (paired by seed) — i.e. the
extra terms buy nothing — or if J-selection is worse at H ≤ 10 by more than it gains at
H ≥ 100 (documented trade-off, `mixed`).

## H3 — Multi-scale slow/fast latent improves long-horizon stability
**Claim.** On systems with explicit timescale separation (FitzHugh–Nagumo, two-cluster
Kuramoto, Lorenz-96, Gray–Scott), a slow/fast latent partition with the paired
dynamics improves recursive NRMSE at H ≥ 100 and VPT relative to a single-scale latent
at equal `d` and equal parameter count.
**Falsified if** the paired difference in NRMSE@100 is not below zero (CI includes 0)
on the timescale-separated systems, or if the same gain is obtained by a single-scale
model with equal parameters (control condition).

## H4 — Stability regularization reduces rollout divergence
**Claim.** Adding the instability term (spectral-radius / norm-growth penalty) during
training reduces the fraction of diverging 2000-step free rollouts and brings the
latent λ̂₁ closer to the data's λ_ref, without increasing NRMSE@25 by more than 10 %
relative.
**Falsified if** diverged fraction does not decrease monotonically with penalty weight
across {0, 0.01, 0.1, 1.0} on S3/S6/S7, or if the required weight degrades NRMSE@25 by
> 10 % (then the trade-off is reported and status `mixed`).

## H5 — Complexity penalty finds Pareto-efficient models
**Claim.** With `λ4 > 0`, the compiler's chosen model lies on the (rollout error,
parameter count) Pareto front in ≥ 4/5 seeds, and the front's hypervolume from the
compiled candidate pool is not worse than that of the baseline size grid.
**Falsified if** the chosen model is dominated (by a candidate or baseline with both
fewer params and lower NRMSE@100) in ≥ 2/5 seeds on a majority of systems, or if
`λ4 = 0` yields the same choices (penalty is inert).

## H6 — Compiler generalizes to out-of-distribution parameters
**Claim.** Compiled models trained on `param_range_train` degrade less (ratio
NRMSE@100_OOD / NRMSE@100_ID) than the best directly fitted sequence model on
Van der Pol (μ), Lorenz-63 (ρ), Lorenz-96 (F), Kuramoto (K).
**Falsified if** the compiled model's degradation ratio is ≥ that of the best baseline
(paired by seed) on ≥ 3 of the 4 systems, or if all models fail equally (NRMSE_OOD > 1
everywhere — no information; status `not supported / inconclusive`, reported).

## H7 — Transfer to real data (EEG, motion capture)
**Claim.** On EEG (subject-level splits) and motion capture, the compiled model reaches
recursive NRMSE@{10, 25, 50} within the seed-CI of the best baseline while using
≤ 25 % of its parameters, and the compile report's chosen `d` is stable across seeds
(± 1).
**Falsified if** the compiled model is outside the CI of the best baseline at all three
horizons, or `d` varies by more than 2 across seeds (unstable compilation), or if no
model beats persistence at H = 10 (dataset uninformative → report, do not claim).

---

Non-hypotheses (things this repo does *not* claim): that latent coordinates correspond
to physical/biological variables; that any result holds beyond the systems and horizons
listed; that the compiler is faster than hand-tuning.
