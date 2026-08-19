# Experiment plan: matrix A–L and gates A–G

Every experiment names its hypothesis (`research/hypotheses.md`) and matrix cell.

This document is the *plan*. Where the plan and the code disagree, the code wins: every
config key, script, metric and flag below was checked against `src/nssc/`, `configs/`,
`scripts/` and `results/` on 2026-08-19. Cells that are still only a plan are labelled
**not implemented** or **not run** rather than described as if they exist.

## Where the work lives

Almost every cell is **config-driven**, executed by `nssc compile --config <file>` or
`nssc benchmark --suite <name>`, not by a per-cell Python driver:

| kind | location |
|---|---|
| datasets (synthetic + real) | `configs/datasets/*.yaml` |
| single runs | `configs/experiments/*.yaml` |
| multi-model benchmark suites | `configs/experiments/benchmarks/*.yaml` (`ablation_stability_reg`, `baseline_rollout_control`, `compiled_vs_manual`, `real_eegbci`, `synthetic_core`) |
| compiler runs | `configs/compiler/*.yaml` (`default`, `tiny`, `lorenz63`, `lorenz63_highdim`, `lorenz96`, `vanderpol`, `eegbci`) |
| compiler ablations | `configs/compiler/ablations/*.yaml` (`lorenz63_valmse`, `lorenz63_rollout_only`, `lorenz63_nostability`, `lorenz63_nocomplexity`, `lorenz63_multiscale`) |
| the only two experiment drivers | `experiments/synthetic/run_ood.py`, `experiments/synthetic/make_compiled_suite.py` |
| pipelines actually used to run the above | `scripts/dev/pipeline_A.sh`, `scripts/dev/pipeline_B.sh` |

`experiments/real_world/`, `experiments/ablations/` and `experiments/benchmarks/` exist but
are **empty** — there are no drivers there, and none are planned while the suite runner
(`nssc benchmark`) covers the same ground from YAML.

Protocol for reported results: benchmark suites use seeds 0–4; compiler final stages use
seeds 0–2 (see the `stages:` list in each `configs/compiler/*.yaml`). Splits are
trajectory-level (subject-level for EEG). Modes are always labelled `recon`,
`teacher_forced` or `recursive`. Horizons come from `eval.horizons`: the synthetic suites use
**1, 5, 10, 25, 50, 100, 250**, `real_eegbci` stops at 100, and `configs/compiler/default.yaml:12`
additionally lists 500. **No run has ever produced a 500-step number**: `evaluate_model` clips the
rollout to `T − context` (480 for the Lorenz-63 dataset at T = 500, context = 20) and
`rollout_errors` skips any horizon beyond that, so `recursive/nrmse@500` is never emitted.
Selection is on validation only.

### Metric names actually produced

Two different key sets exist and they are not the same:

* **Per-run `metrics.json`** — everything `evaluate_model` emits
  (`src/nssc/evaluation/evaluator.py:83-121`), under a `val`/`test` sub-dict:
  `recon/{mse,nrmse}`, `teacher_forced/{mse,nrmse}`, `recursive/nrmse@k` and
  `recursive/nrmse_step@k` for every configured horizon that fits, `recursive/nrmse_mean`,
  `recursive/divergence_time`, `recursive/{horizon,context}`, `curves.recursive_nrmse`
  (the full per-step curve), `stability/{instability_score,rho_max,lyapunov_max,frac_blowup,verdict}`,
  `params/*`, `flops/dynamics_step`, `latency/step_*`, `latency/encode_*`, `latent_dim`.
* **`results/registry.jsonl`** — only the flat subset in `SUMMARY_KEYS`
  (`src/nssc/experiment.py:179`), prefixed `val/` or `test/`. It deliberately omits
  `nrmse@5`, every `nrmse_step@k` and the curves, which is why a registry row shows
  `@{1,10,25,50,100,250}` while the run's `metrics.json` also has `@5`.

**`recursive/nrmse@k` is cumulative** — the mean per-step NRMSE over steps 1..k
(`src/nssc/metrics/prediction.py:58-72`) — not the error at step k. The instantaneous value is
`recursive/nrmse_step@k`, which is not carried into the registry. Every "NRMSE@250" in
`results/tables/` is therefore a mean over the first 250 steps.

There is **no** VPT metric. The closest thing is `recursive/divergence_time`
(`src/nssc/metrics/prediction.py:75`): the first horizon step whose per-step NRMSE exceeds
`divergence_threshold` (default 1.0), returning `H+1` when it never does — so the `251` that
fills the suites' "div. time" column means *no divergence within the 250-step rollout*, not a
measured time. There is **no** attractor-error metric and **no** hypervolume metric. Pareto analysis
reports a *dominated area* in (log₁₀ params × NRMSE) space
(`src/nssc/evaluation/pareto.py:38`), not hypervolume. Linear alignment R² between latents
and true states exists only as a figure caption (`align_latents`,
`src/nssc/visualization/latent.py:86`), not as a registered metric.

## Systems

The 11 rows with a `registry key` in backticks below are exactly the keys returned by
`python3 -c "import nssc.data.systems; from nssc.utils.registry import SYSTEMS; print(sorted(SYSTEMS))"`.
S6′ is not a separate generator — it is `lorenz63` seen through a different observation map.
S10/S11 are real-data sources (`nssc.data.real.REAL_SOURCES`), not synthetic systems.

| id | registry key | state dim (as configured) | D (as configured) | dataset config | OOD block in config | runs in `results/registry.jsonl` |
|----|--------------|---------------------------|-------------------|----------------|---------------------|----------------------------------|
| S1a | `harmonic` | 2 | 2 | `configs/datasets/harmonic.yaml` | — | none (used lifted to D=8 in `configs/compiler/tiny.yaml`) |
| S1b | `damped_oscillator` | 2 (ζ=0.1) | 2 | `configs/datasets/damped_oscillator.yaml` | — | none |
| S2 | `pendulum` | 2 (γ=0, undamped) | 2 | `configs/datasets/pendulum.yaml` | — | none |
| S3 | `vanderpol` | 2 (μ=1) | 2 | `configs/datasets/vanderpol.yaml` | `mu`: train [0.5,2.0], test [2.5,4.0] | 158 completed (98 compiler + 60 suite) |
| S4 | `lotka_volterra` | 2 | 2 | `configs/datasets/lotka_volterra.yaml` | — | none |
| S5 | `fitzhugh_nagumo` | 2 (slow/fast, I=0.5) | 2 | `configs/datasets/fitzhugh_nagumo.yaml` | `I`: train [0.4,0.7], test {0.0} | none |
| S6 | `lorenz63` | 3 (chaotic, λ₁≈0.905) | 3 | `configs/datasets/lorenz63.yaml` | `rho`: train [24,32], test {20,35} | 299 completed (190 compiler incl. ablations + 109 suite) |
| S6′ | `lorenz63` + random-MLP observation | 3 | **64**, `noise_std` 0.05 | `configs/datasets/lorenz63_highdim.yaml` | inherited from S6 | 173 completed (123 compiler + 50 suite) |
| S7 | `lorenz96` | 8 (N=8, F=8) | 8 | `configs/datasets/lorenz96.yaml` | `F`: train [7,9], test {4,12} | 23 completed, 1 running (compile screen stage unfinished) |
| S8 | `kuramoto` | 8 phases (N=8, K=2) | **16** (cos, sin via `kuramoto_sin_cos`) | `configs/datasets/kuramoto.yaml` | `K`: train [1.5,3.0], test {0.5,4.0} | none |
| S9 | `gray_scott` | **1-D** reaction–diffusion, N=32 grid → state 2N | 64 | `configs/datasets/gray_scott.yaml` | — | none |
| S9′ | `coupled_oscillators` | 8 (N=4 chain) | 8 | `configs/datasets/coupled_oscillators.yaml` | — | none |
| S10 | real: `eegbci` | unknown | 64 channels @ 64 Hz, 8 s → T=512 | `configs/datasets/eegbci.yaml` (+ `eegbci_tiny.yaml`) | subject-level split (train 1–5 / val 6 / test 7,8) | 2 smoke rows on `eegbci_tiny` only (EXP-0043 completed, EXP-0042 `failed`); no suite or compile runs |
| S11 | real: `motion_cmu_mocap` | — | — | — | — | **not implemented**: `nssc/data/real/motion.py` deliberately raises `NotImplementedError` rather than shipping a synthetic stand-in |

Corrections to earlier drafts of this table: Gray–Scott is a **1-D** solver on 32 grid points
(D = 64), not a 32×32 / 64×64 2-D field (D = 2048 / 8192); Lorenz-96 is configured at N = 8
only (no N = 10/20/40 configs); Kuramoto is configured at N = 8 only, with no two-cluster ω
config; and the "2, 16, 64" lift grids per system do not exist — the only lifted dataset is
`lorenz63_highdim.yaml`.

**OOD blocks are documentation only.** `build_dataset` keeps the `ood:` key in the config (so
it is hashed into the dataset version) but never applies it — see the schema note at
`src/nssc/data/builder.py:18`. Training over a parameter *range* is not implemented:
`param_range_split` exists and is unit-tested (`src/nssc/data/splits.py:45`,
`tests/unit/test_data_splits.py:43`) but no config or code path calls it. OOD is instead
evaluated post-hoc on frozen checkpoints by `experiments/synthetic/run_ood.py`
(`param_shifts` and `ic_scales`, `src/nssc/evaluation/ood.py`).

### Observation maps

`OBS_MAPS` (`src/nssc/data/observation.py:193`) has exactly six kinds, used as
`observation: {type: <kind>, ...}`:

| kind | class | parameters |
|---|---|---|
| `identity` | `IdentityObservation` | — |
| `linear` | `LinearObservation` | `obs_dim`, `seed`, `orthogonal` |
| `mlp` | `RandomMLPObservation` | `obs_dim`, `hidden`, `n_layers`, `seed`, `gain` (fixed random tanh MLP, untrained) |
| `polynomial` | `PolynomialObservation` | `degree`, `obs_dim`, `seed` |
| `redundant` | `RedundantObservation` | `repeats`, `alpha`, `seed` |
| `pipeline` | `ObservationPipeline` | `maps: [...]` (sequential composition) |

**Not implemented:** delay embedding and partial (coordinate-subset) observation. Earlier
drafts of this plan named `linear_lift`, `mlp_lift`, `delay_embed` and `partial`; the first
two are really keyed `linear` and `mlp`, and the last two do not exist. Cell J cannot be run
as written until they are added.

Corruptions available alongside the map: `noise_std` (additive iid Gaussian, `add_noise`) and
`missing_rate` (iid NaN + mask, `mask_missing`). `irregular_subsample` exists in the same
module but no config key reaches it — not wired into `build_dataset`.

In practice only two non-identity maps are configured anywhere:
`configs/datasets/lorenz63_highdim.yaml` (`{type: mlp, obs_dim: 64, hidden: 64, seed: 0}`,
`noise_std: 0.05`) and `configs/compiler/tiny.yaml` (`{type: linear, obs_dim: 8, seed: 0}`).
Every other dataset config is `identity`, `noise_std: 0.0`, `missing_rate: 0.0` — the
noise sweep `{0, 0.01, 0.05, 0.1}` is planned, not configured.

## Matrix

Status values: **run** = registered runs + a committed table/report; **partial** = some
systems or some of the planned sweep; **not run** = config exists, no runs;
**not implemented** = no code path.

| cell | name | hypothesis | systems run (planned) | what varies | primary readout | implemented by | status |
|------|------|-----------|----------------------|-------------|-----------------|----------------|--------|
| A | Sanity & recovery on linear systems | (infra) | harmonic lifted to D=8 (S1) | observation map, `noise_std` | compiler picks `dynamics == linear`, `latent_dim ≤ 4`; report generated | `configs/compiler/tiny.yaml`; asserted by `tests/integration/test_compiler.py:24-48` | **partial** — passes as a test (tmp dirs, 1 seed); no seeds 0–4 run, no committed table, no noise sweep |
| B | Latent-dimension recovery | H1 (precondition), H5 | lorenz63, lorenz63_highdim, vanderpol (S1–S9) | `candidates.latent_dims` inside a compile: {2,3,4,8}, {2,3,4,8,16}, {2,3,4} | val `recursive/nrmse@…` vs latent dim; selected d vs profiler intrinsic-dim estimate | the `latent_dims` axis of `configs/compiler/{lorenz63,lorenz63_highdim,vanderpol}.yaml`; figure `latent_dim_sweep.png` per compile dir | **partial** — 3 datasets; the sweep figure is the screen stage (1 seed), not a 5-seed study; no standalone sweep config |
| C | Dynamics-family selection | H1 | lorenz63, lorenz63_highdim, vanderpol (lorenz96 running) | `candidates.dynamics` = linear, residual_mlp, koopman, ssm (+ neural_ode at 1 substep on lorenz63/vanderpol; omitted from the high-dim pool) | which family wins; `compile_report.md` reasons; `family_comparison.png` | same compile configs | **partial** — see `results/compile/*/compile_report.md`; lorenz96 has 23 completed runs and no report yet; alignment R² is a figure caption only |
| D | Long-horizon rollout vs baselines | **H1** (primary) | lorenz63, lorenz63_highdim, vanderpol (S1–S9) | 4 hand-picked latent models vs **6 baselines** (persistence, gru, lstm, tcn, transformer, ssm) | `test/recursive/nrmse@{1..250}`, `recursive/divergence_time`, `params/total`, paired t / Wilcoxon | `configs/experiments/benchmarks/synthetic_core.yaml` → `nssc benchmark --suite synthetic_core` | **run** — 3 datasets × 10 models × 5 seeds = 150 runs; `results/tables/synthetic_core.md`, `results/figures/suites/synthetic_core/` |
| E | Selection-criterion ablation | **H2** | lorenz63 (S3, S5, S6, S7, S9) | `objective.criterion ∈ {multi_objective, val_mse, rollout_only}` over the *same* cached candidate pool | selected candidate, `J`, val `recursive/nrmse@250`, `stability/verdict` | `configs/compiler/ablations/lorenz63_valmse.yaml`, `…/lorenz63_rollout_only.yaml` | **partial** — both run on lorenz63 (`results/compile/ablations/*/compile_report.md`); `val_mse` picks `linear+koopman@d8` (rollout 0.1347) vs `mlp+residual_mlp@d3` (0.0164) for `multi_objective`; other systems not run |
| F | Multi-scale slow/fast latent | **H3** | lorenz63_highdim (S5, S7, S8, S9) | `multiscale` encoder (`slow_dim` 1/2, base tcn) and `multiscale` dynamics vs tcn/mlp + residual_mlp at d ∈ {4,8} | val rollout NRMSE, `stability/verdict`, params | `configs/compiler/ablations/lorenz63_multiscale.yaml` | **partial** — 39 runs, report written; every multiscale candidate is poor here — the selected `multiscale+residual_mlp@d8` has val rollout 1.400, the best is `multiscale+residual_mlp@d4` at 1.101, and a third scores 15.42 with verdict `explodes`; timescale-separated systems (FHN, Kuramoto) not run |
| G | Stability regularization | **H4** | lorenz63 (S3, S6, S7) | `training.loss.stability ∈ {0, 0.01, 0.1, 1.0}` for residual_mlp; `{0, 0.1}` for koopman | `test/recursive/nrmse@{50,100,250}`, `recursive/divergence_time`, `stability/frac_blowup` over `eval.stability_horizon` (300) | `configs/experiments/benchmarks/ablation_stability_reg.yaml` | **run** on lorenz63 — 6 models × 5 seeds, `results/tables/ablation_stability_reg.md`. Two caveats before reading it as an H4 test: the free rollout is **300** steps (`eval.stability_horizon`), not 2000; and `test/stability/frac_blowup` is **0.0 in all 30 runs**, so there is no divergence for the penalty to remove on this system. What the sweep does show is a cost: residual-MLP NRMSE@250 goes 0.0746 (w=0) → 0.0736 (0.01) → 0.0651 (0.1) → 0.618 (w=1.0), with `divergence_time` dropping from 251 to 141.6 ± 69.6 at w=1.0 |
| H | Complexity penalty / Pareto | **H5** | lorenz63 (S1–S9) | `objective.complexity` (the λ₄ of the scorer docstring): 0.1 (default) vs 0.0 | selected candidate + `J`; Pareto membership in (params, NRMSE@50) | `configs/compiler/ablations/lorenz63_nocomplexity.yaml`; `nssc pareto --suite synthetic_core` | **partial** — two points only, not the planned `{0, 0.25, 0.5, 1, 2}` sweep; at 0.0 and 0.1 the selection is unchanged (`mlp+residual_mlp@d3`), i.e. the penalty is inert here. Pareto table: `results/tables/pareto_synthetic_core.md` |
| I | OOD parameter generalization | **H6** | lorenz63 (ρ), vanderpol (μ) (planned: + lorenz96 F, kuramoto K) | frozen checkpoints re-evaluated at shifted parameters and widened initial conditions (`ic_scale` 2, 4) | `recursive/nrmse@50` per condition + degradation ratio vs in-distribution | `experiments/synthetic/run_ood.py --suite synthetic_core --dataset …` | **run** for 2 systems — `results/tables/ood_synthetic_core_{lorenz63,vanderpol}.md`. Training on a parameter *range* is not implemented (see above), so this is shift-at-test-time only |
| J | Observation robustness | H1 (scope) | — | obs map × `noise_std` | metrics vs noise/obs map; selected d drift | — | **not run**, and partly **not implemented**: `delay` and `partial` maps do not exist, and no config sweeps `noise_std` |
| K | Real data | **H7** | — (planned: eegbci, motion capture) | latent models vs baselines, subject-level splits | `test/recursive/nrmse@{1..100}`, teacher-forced; calibration only for `gaussian` dynamics | `configs/experiments/benchmarks/real_eegbci.yaml`, `configs/compiler/eegbci.yaml` (both queued in `scripts/dev/pipeline_A.sh`) | **not run** — zero `suite:real_eegbci` rows and no eegbci compile; the real-data path itself works end to end (`configs/experiments/eegbci_smoke.yaml` → EXP-0043 completed on `eegbci_tiny`, 1 subject per split, explicitly not a result). Motion capture source not implemented |
| L | Compiler cost & search fidelity | (infra, H5) | lorenz63, vanderpol, lorenz63_highdim | staged screen→fine→final; resume after kill | `n_runs` and wall-clock per compile report; which candidates the screen discarded (`search_state.json` stage rankings) | `configs/compiler/*.yaml` + `results/compile/*/search_state.json`; resume asserted by `tests/integration/test_compiler.py:52-73` | **partial** — cost and stage funnels are recorded (lorenz63: 127 runs / 22.1 h; vanderpol: 98 runs / 39.5 h; lorenz63_highdim: 123 runs / 3.7 h). The **staged-vs-exhaustive control is not implemented**: there is no exhaustive mode and no run that gives every candidate the full budget, so "did the coarse screen discard the eventual best?" (open question Q-002) is still unanswered |

### Naming caution: two conflicting letter schemes

Several configs and driver docstrings use experiment letters that do **not** match this
matrix. They are the same experiments under different labels; nothing needs to be re-run,
but do not trust the letters inside those files:

| file | letter it claims | cell in this matrix |
|---|---|---|
| `configs/experiments/benchmarks/synthetic_core.yaml:1` | "Experiment D / Gate D" | D ✓ |
| `configs/experiments/benchmarks/real_eegbci.yaml:1` | "Experiment K" | K ✓ |
| `configs/experiments/benchmarks/ablation_stability_reg.yaml:1` | "H4 ablation (Experiment K)" | **G** |
| `experiments/synthetic/run_ood.py:2` | "Experiments G/H" | **I** |
| `experiments/synthetic/make_compiled_suite.py:2`, `configs/experiments/benchmarks/compiled_vs_manual.yaml:1` | "Experiment L" | closest to **D/H** (compiled vs manual vs baselines under the benchmark protocol); *not* the search-fidelity study of cell L |
| `configs/compiler/ablations/lorenz63_nostability.yaml:1` | "Ablation D" | E (objective-term ablation) |
| `configs/compiler/ablations/lorenz63_nocomplexity.yaml:1` | "Ablation E" | **H** |
| `configs/compiler/ablations/lorenz63_multiscale.yaml:1` | "ablation C→F" | F ✓ |
| `src/nssc/evaluation/pareto.py:1` | "Experiment I / §26" | **H** |

Budget note: cells D and K are the expensive ones. `synthetic_core` alone is 150 runs;
the three completed compiles are 348 runs and ~65 h wall-clock on one CPU laptop
(the 22 h lorenz63 compile was a swap-throttled 8 GB laptop with ~1/3 of that time asleep —
F-003 in `research/failures.md`).
A baseline **size grid** is available (`small`/`medium`/`large` presets in
`configs/models/baselines/*.yaml`) but has never been run: every suite pins one size per
baseline (gru/lstm `medium`; tcn/transformer/ssm `small`).

## Gates

Work proceeds gate by gate; a gate closes only when the reviewer accepts.

| gate | criterion | evidence |
|------|-----------|----------|
| **A — Data & tests** | All registered generators implemented, invariants tested, RK4 accuracy/consistency tests, Lorenz λ₁ regression (0.905 ± 0.1), trajectory-level split tests, dataset determinism + version-hash test; `pytest -q -m "not slow"` green in CI. | test run, `tests/` |
| **B — Baselines & registry** | All 7 registered baselines + metrics + protocols run on a tiny config; registry rows written (incl. a deliberately failing run → `failed`); checkpoint round-trip. | EXP rows, integration test |
| **C — Compiler end-to-end** | `nssc compile --config configs/compiler/tiny.yaml` (harmonic, D = 8 linear lift) picks `latent_dim ≤ 4` and `dynamics == linear`, report generated; resume test passes; tiny integration run on CPU. Cell A done. | EXP id, `compile_report.md` |
| **D — Synthetic benchmark core** | Cells B, C, D complete for lorenz63, vanderpol, lorenz63_highdim with seeds 0–4; compile figures + suite figures generated; benchmark tables; H1 status (for these systems) written. | tables, figures, `experiment_log.md` |
| **E — Ablations** | Cells E, F, G, H complete on their systems; ablation reports and tables; H2–H5 statuses written with n = 5 statistics where the cell is a benchmark suite (compiler ablations aggregate 3 seeds). | as above |
| **F — Generalization & real data** | Cells I, J, K; H6, H7 statuses; limitations section drafted. | as above |
| **G — Reproducibility audit** | Fresh clone + `make install` + `scripts/reproduce.sh {smoke,compile,ablations,benchmark,figures}` reproduces one table cell per gate within tolerance; `docs/reproducibility.md` complete; README results block regenerated by `python scripts/generate_report.py`; reviewer accepts all claims. | audit note in `research/experiment_log.md` |

`scripts/reproduce.sh` takes exactly one of `smoke | compile | ablations | benchmark |
figures`; anything else prints `unknown target` and exits 1. There is **no** `reproduce.sh
EXP-xxxx` mode — re-running a single registered experiment is `nssc evaluate --experiment
EXP-xxxx` (re-evaluates a checkpoint; it does not retrain).

## Cross-cutting rules (repeat of CLAUDE.md, applied here)
- Splits: trajectory-level; subject-level for EEG. OOD is applied at evaluation time
  (parameter shift / widened ICs), not as a train/test parameter-range split.
- Seeds 0–4 for benchmark suites, mean ± std (+ bootstrap CI); paired-by-seed comparisons;
  n stated. Compiler final stages use seeds 0–2 and the report says so.
- Never optimize on test; test evaluated once at the end of each cell.
- Failed runs stay in the registry as `failed`, and in `research/failures.md`.
- Benchmark definitions frozen once used; changes = new ids + decision entry (D-007).

## Gate status (updated 2026-08-19)

| gate | status | evidence |
|---|---|---|
| A — Data & tests | **closed with a caveat** | 11 generators + invariant tests (`tests/unit/test_data_systems.py`), RK4 tests (energy conservation, RK4-beats-Euler, substeps-match-finer-dt — there is **no** RK4-vs-scipy test), trajectory-level split + leakage tests (`tests/unit/test_data_splits.py`), determinism + config-version test (`tests/unit/test_data_dataset.py:32`); CI runs `ruff check src tests scripts` + `pytest -q -m "not slow"` on Python 3.10 and 3.12 (`.github/workflows/ci.yml`). **Caveat:** the Lorenz λ₁ regression (`tests/regression/test_known_dynamics_values.py:28`, asserts `abs(lam - 0.905) < 0.1`) is `@pytest.mark.slow`, so CI does not execute it. |
| B — Baselines & registry | **closed with a caveat** | 7 baselines registered (`persistence, mean, gru, lstm, tcn, transformer, ssm`); metrics + `recursive`/`direct` protocols; failing run recorded not raised (`tests/integration/test_end_to_end.py:45`, against a temp registry); checkpoint round-trip (`tests/integration/test_end_to_end.py:36-41`). **Caveats:** `mean` is registered but appears in no suite config and has no registered runs, so "all baselines run" is 6/7; and the single `failed` row in the committed `results/registry.jsonl` is not the deliberate one — it is EXP-0042, the EEG smoke run that hit `Cannot convert a MPS Tensor to float64` (hence `training.device: cpu` in `configs/experiments/eegbci_smoke.yaml`). |
| C — Compiler end-to-end | **closed with a caveat** | `configs/compiler/tiny.yaml` selects `dynamics == linear`, `latent_dim ≤ 4` on linearly-lifted harmonic data, writes the report files, and resumes without re-running cached candidates (`tests/integration/test_compiler.py:24,52,77`). **Caveat:** all three of those tests are `@pytest.mark.slow` and therefore excluded from the CI command. |
| D — Synthetic benchmark core | **in progress** | `synthetic_core` complete: 3 datasets × 10 models × 5 seeds (`results/tables/synthetic_core.md`, `results/figures/suites/synthetic_core/`). Compiles complete for lorenz63, vanderpol, lorenz63_highdim (`results/compile/*/compile_report.md`); lorenz96 compile has 23 completed runs and no report yet; H1 is "partially supported" in `research/hypotheses.md` — supported on lorenz63/vanderpol (identity obs), *not* supported by the hand-picked latent model on lorenz63_highdim, where the compiler's own pick (`gru+residual_mlp@d8`, val rollout 0.9235) is only 0.3% better than a 256-parameter `pca+linear@d16` (0.9265) and no candidate gets val rollout NRMSE below 0.92. |
| E — Ablations | **partial** | Cells E, F, G, H all have at least one run on lorenz63: `results/compile/ablations/{lorenz63_valmse,lorenz63_rollout_only,lorenz63_nostability,lorenz63_nocomplexity,lorenz63_multiscale}/compile_report.md` and `results/tables/ablation_stability_reg.md`. Missing before this gate can close: the other systems named in the cells, the `objective.complexity` sweep (only 0.0 vs 0.1 exist), and H2–H5 status entries in `research/hypotheses.md` (all four still `untested`). Remaining jobs are queued in `scripts/dev/pipeline_A.sh` / `scripts/dev/pipeline_B.sh`. |
| F — Generalisation & real data | **partial** | Cell I run on two systems (`results/tables/ood_synthetic_core_{lorenz63,vanderpol}.md`). Cell J not run (delay/partial observation maps not implemented). Cell K not run at all: `configs/compiler/eegbci.yaml` and `configs/experiments/benchmarks/real_eegbci.yaml` exist and are queued in `pipeline_A.sh`, but the registry has no EEG suite or compile rows — only the two `eeg_smoke` rows on `eegbci_tiny` — and the motion-capture source raises `NotImplementedError`. H6/H7 remain `untested`. |
| G — Reproducibility audit | **open** | Known blocker (R-02 in `research/review_2026-08-18.md`): on a fresh clone `scripts/reproduce.sh compile` resumes from the committed `search_state.json` and then fails loading a gitignored `model.pt`, and `reproduce.sh benchmark` skips all 150 `synthetic_core` runs as already-completed in the committed registry. `nssc compile` does accept `--no-resume`, but `reproduce.sh` does not pass it, and `nssc benchmark` has no registry override. |
