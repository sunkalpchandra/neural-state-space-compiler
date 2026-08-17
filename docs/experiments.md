# Experiment plan: matrix A–L and gates A–G

Every experiment names its hypothesis (`research/hypotheses.md`) and matrix cell.
Drivers live under `experiments/{synthetic,real_world,ablations,benchmarks}/`, configs
under `configs/experiments/`. All reported results: seeds 0–4, trajectory-level splits,
modes labeled, horizons 1–500 where T allows, selection on validation only.

## Systems

| id | system | intrinsic n | D (observation) | regime / OOD parameter |
|----|--------|-------------|-----------------|------------------------|
| S1 | harmonic / damped oscillator | 2 | 2, 16 (linear lift), 64 (MLP lift) | ζ ∈ {0, 0.05, 0.1} |
| S2 | nonlinear pendulum | 2 | 2, 16, 64 | γ ∈ {0, 0.1} |
| S3 | Van der Pol | 2 | 2, 16, 64 | μ train [0.5,2], OOD [2.5,4] |
| S4 | Lotka–Volterra | 2 | 2, 16 | — |
| S5 | FitzHugh–Nagumo | 2 (slow/fast) | 2, 16, 64 | I ∈ {0.5, 0.0} |
| S6 | Lorenz-63 | 3 (chaotic, λ₁≈0.9) | 3, 16, 64, delay-embed | ρ train [24,32], OOD {20,35} |
| S7 | Lorenz-96 | N (10/20/40) | N, 2N lift | F train 8, OOD {4, 10} |
| S8 | Kuramoto | N phases (8/32) | 2N (cos, sin) | K ∈ {0.5, 2, 4}; two-cluster ω for slow/fast |
| S9 | Gray–Scott | PDE (32×32 or 64×64) | 2048 / 8192 | (F,k) spots/worms/mitosis |
| S10 | EEG (public dataset, subject-level split) | unknown | channels (≈ 32–64) × band-passed | subjects |
| S11 | motion capture (public) | unknown | joint coordinates (≈ 60–90) | sessions/subjects |

Observation maps: `identity`, `linear_lift`, `mlp_lift`, `delay_embed`, `partial`
(subset of coordinates), each with `noise_std ∈ {0, 0.01, 0.05, 0.1}` (relative to data
std). Defaults per system in `configs/datasets/`.

## Matrix

| cell | name | hypothesis | systems | what varies | primary readout | figures/tables |
|------|------|-----------|---------|-------------|-----------------|----------------|
| A | Sanity & recovery on linear systems | (infra) | S1 | obs map, noise | compiler picks d = 2 and linear dynamics; recon/rollout ≈ noise floor | F1, table A |
| B | Latent-dimension recovery | H1 (precondition), H5 | S1–S6, S8 (K=4 vs 0.5), S9 | d ∈ {1,2,3,4,6,8,12,16,32} × fixed MLP AE + residual | val recursive NRMSE@25 vs d; chosen d vs intrinsic n | F2 |
| C | Dynamics-family selection | H1 | S1–S9 | all dynamics families at d = n and d = 2n | which family wins per system; report justification quality; alignment R² | F9, table C |
| D | Long-horizon rollout vs baselines | **H1** (primary) | S1–S9 | compiled model vs 9 baselines (size grid) | recursive NRMSE at H ∈ {1..500}, VPT, attractor error; params/latency | F3, benchmark tables |
| E | Selection-criterion ablation | **H2** | S3, S5, S6, S7, S9 | `selection.criterion ∈ {val_mse, multi_objective}`, same candidate pool | rollout NRMSE@{50,100,250,500}, diverged fraction, spectral radius | F6 |
| F | Multi-scale slow/fast latent | **H3** | S5, S8 (two-cluster), S7, S9 | single-scale vs multiscale encoder+dynamics at equal d and equal params | NRMSE vs horizon, VPT | F7 |
| G | Stability regularization | **H4** | S3, S6, S7 | `loss_weights.stability ∈ {0, 0.01, 0.1, 1.0}` for MLP/residual/Koopman dynamics | diverged fraction over 2000-step free rollouts, ρ(J), λ̂₁ vs λ_ref, NRMSE@250 | F5 |
| H | Complexity penalty / Pareto | **H5** | S1–S9 | `λ4 ∈ {0, 0.25, 0.5, 1, 2}` | is the chosen model on the (rollout, params) Pareto front; hypervolume; params vs error | F4 |
| I | OOD parameter generalization | **H6** | S3 (μ), S6 (ρ), S7 (F), S8 (K) | train range vs OOD test range | NRMSE@{25,100} ID vs OOD, degradation ratio, compiled vs baselines | F8 |
| J | Observation robustness | H1 (scope) | S1, S3, S6, S9 | obs map ∈ {identity, linear, mlp, delay, partial} × noise ∈ {0..0.1} | metrics vs noise/obs map; chosen d drift | table J |
| K | Real data | **H7** | S10, S11 | compiled vs baselines; subject-level splits | recursive NRMSE@{1..50}, teacher-forced; calibration if Gaussian dynamics | table K, F3-style panel |
| L | Compiler cost & search fidelity | (infra, H5) | S3, S6, S9 | staged vs exhaustive (all candidates full budget) on small pools; kill/resume | did coarse screen discard the eventual best? wall-clock per stage; resume determinism | F10 |

Budget note: cells D and K are the expensive ones (9 baselines × size grid × 5 seeds ×
systems). Run S1, S3, S6 first (Gate D), then extend.

## Gates

Work proceeds gate by gate; a gate closes only when the reviewer accepts.

| gate | criterion | evidence |
|------|-----------|----------|
| **A — Data & tests** | All S1–S9 generators implemented, invariants tested, RK4-vs-scipy test, Lorenz λ₁ regression (0.9 ± 0.15), trajectory-level split tests, dataset cache determinism; `pytest -m "not slow"` green in CI. | test run, `tests/` |
| **B — Baselines & registry** | All 9 baselines + metrics + protocols run on S1 tiny config; registry rows written (incl. a deliberately failing run → `failed`); checkpoint round-trip. | EXP rows, integration test |
| **C — Compiler end-to-end** | `nssc compile` on S1 (D = 16 lift) picks d = 2, linear/affine dynamics, report generated; resume test passes; tiny integration < 60 s CPU. Cell A done. | EXP id, `compile_report.md` |
| **D — Synthetic benchmark core** | Cells B, C, D complete for S1, S3, S6 with seeds 0–4; F2, F3, F9 generated; benchmark tables; H1 status (for these systems) written. | tables, figures, `experiment_log.md` |
| **E — Ablations** | Cells E, F, G, H complete on their systems; F4–F7; H2–H5 statuses written with n = 5 statistics. | as above |
| **F — Generalization & real data** | Cells I, J, K; F8; H6, H7 statuses; limitations section drafted. | as above |
| **G — Reproducibility audit** | Fresh clone + `make install` + `scripts/reproduce.sh EXP-xxxx` reproduces one table cell per gate within tolerance; `docs/reproducibility.md` complete; README results block regenerated by script; reviewer accepts all claims. | audit note in `research/experiment_log.md` |

## Cross-cutting rules (repeat of CLAUDE.md, applied here)
- Splits: trajectory-level; OOD via parameter ranges; subject-level for S10.
- Seeds 0–4; mean ± std (+ bootstrap CI); paired-by-seed comparisons; n stated.
- Never optimize on test; test evaluated once at the end of each cell.
- Failed runs stay in the registry as `failed`, and in `research/failures.md`.
- Benchmark definitions frozen once used; changes = new ids + decision entry.
