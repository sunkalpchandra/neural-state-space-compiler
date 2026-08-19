# The nssc compiler

`StateSpaceCompiler` (`src/nssc/compiler/compiler.py:49`) turns a dataset config into a
`CompiledModel` — a trained `LatentModel` with a chosen latent dim `d`, encoder, decoder and
dynamics family — plus a `CompileReport` that justifies the choice. Everything below is driven by
`configs/compiler/*.yaml` (loaded as a plain `Config` dict, with `_base_` inheritance and
`--set a.b=c` overrides; see `src/nssc/utils/config.py:98`).

This document describes the code as it exists. Behaviour that was designed but never implemented
is marked **not implemented** rather than described in the present tense.

## Public API

```python
from nssc.compiler import StateSpaceCompiler

comp     = StateSpaceCompiler(cfg, device=None, registry=None, log=print)
profile  = comp.fit()                       # dict, compiler.py:63
cands    = comp.propose()                   # list[CandidateSpec], compiler.py:88
result   = comp.search()                    # {"history", "final", "state_path"}, compiler.py:104
compiled = comp.compile(search_result=None) # CompiledModel, compiler.py:114
compiled = comp.run(resume=True)            # fit + propose + compile(search()), compiler.py:155
metrics  = comp.evaluate(compiled, split="test")   # compiler.py:164
```

`comp.base_run_cfg()` (`compiler.py:98`) builds the shared run config handed to every candidate
(the `dataset`, `windows`, `training`, `eval` and `tags` blocks).

`CompiledModel` (`compiler.py:36`) is a dataclass with `model`, `spec`, `report`, `checkpoint`,
`output_dir` and a `rollout(x_context, horizon)` helper. There is no `profile()`, `candidates()`,
`stability()`, `score()` or `report()` method, and `compile()` returns a `CompiledModel`
(which carries `.report`), not a tuple.

## Stages

| # | stage | input → output | implementation |
|---|-------|----------------|----------------|
| 0 | `fit` (profile) | dataset cfg → profile dict (`profile.json`) | `nssc.compiler.profiler.profile_dataset`: shape/dt, PCA variance curve and `pca_dims_for_variance` at 0.90/0.95/0.99/0.999, Levina–Bickel MLE (k=10, k=20) + Grassberger–Procaccia correlation dimension, autocorrelation time, smoothness, dominant period, spectral flatness, second-difference noise estimate, stationarity, linear one-step and recursive-10-step R², Rosenstein Lyapunov proxy → `suggested_latent_dims` and a `recommendations` dict (`candidate_latent_dims`, `likely_linear`, `likely_chaotic`, `noisy`, `long_memory`, `nonstationary`). No training. The profile is computed on the **train split only** — `build_dataset(dcfg)` → `ds.split()["train"]` → `profile_dataset(train)` (`compiler.py:72-82`) — and the resulting dict carries `computed_on = {split: "train", n_traj: <train>, of_total: <all>}`. Cached: if `profile.json` already exists it is loaded and *not* recomputed (`compiler.py:67-71`). |
| 1 | `propose` | profile + `candidates` cfg → `list[CandidateSpec]` (`candidates.json`) | `nssc.search.space.generate_candidates`: `itertools.product(latent_dims, encoders, dynamics, hidden_dims)`. See the candidate rules below. |
| 2..k | `search` — one entry per `stages:` item | survivors → survivors | `nssc.search.staged.StagedSearch`. Each stage trains every surviving candidate for `epochs` on `seeds` via `nssc.experiment.run_experiment` (so every newly trained run gets an `EXP-####` registry row and, unless the stage sets `save_ckpt: false`, a checkpoint), ranks with the multi-objective score below on **validation** metrics, then prunes with `keep_top` / `keep_frac` while never discarding candidates within `score_tolerance` of the best. Stability metrics come from the evaluator (`nssc.stability.analyze_stability`, enabled by `eval.stability`, default true). Identical run configs already completed in the registry are reused when `reuse_registry` is true (default). |
| k+1 | `compile` | final-stage ranking → `CompiledModel` (`compiled_model.yaml`, `compile_report.{json,md}`) | winner = the highest-ranked final-stage candidate with at least one completed seed; its best-seed checkpoint (lowest validation rollout NRMSE at `rollout_key`) is loaded. `reasons` are computed from the final ranking by `build_reasons` (`report.py:81`). |
| — | `evaluate` (optional) | compiled model → held-out metrics | `StateSpaceCompiler.evaluate(compiled, split="test")`. **Not called by `run()` and not reachable from `nssc compile`** — it is a Python-API call. |

`run(resume=True)` executes stages 0, 1, 2..k, k+1 in that order (`compiler.py:155`).

### Candidate rules (`src/nssc/search/space.py`)

- `latent_dims: auto` uses `recommendations.candidate_latent_dims` (falling back to
  `suggested_latent_dims`, then to `[2, 4, 8, 16, 32]`), keeping only dims in `1..obs_dim`
  (`space.py:70`). An explicit list is honoured as given, including overcomplete `d > D`.
- `encoders` / `dynamics` entries are either a string or `{name, kwargs}`.
- The decoder is taken from the optional `decoders` mapping (encoder name → decoder spec);
  otherwise `pca → pca`, `linear → linear`, everything else → `mlp` (`space.py:17`).
- `hidden_dims: [h, ...]` sets `kwargs.hidden_dims = [h, h]` on any component whose family is one
  of `mlp`, `residual_mlp`, `koopman`, `neural_ode`, `gaussian`, `multiscale`, and tags the
  candidate `h<h>` (`space.py:104-108`).
- Multi-scale consistency: if both encoder and dynamics define `slow_dim` they must agree, and
  `slow_dim < latent_dim` (`space.py:110-114`).
- `pca` is paired only with `linear`/`affine` dynamics while `pca_only_linear` is true
  (its default) — PCA is frozen, so SGD dynamics on top of it are off by default (`space.py:115`).
- `exclude: [{encoder, dynamics, latent_dim}, ...]` drops partial matches; `max_candidates`
  truncates the list (`space.py:117-124`).
- `CandidateSpec.id` = `f"{encoder}+{dynamics}@d{latent_dim}-{stable_hash(spec, 6)}"`, e.g.
  `mlp+residual_mlp@d3-5fa02e`; `CandidateSpec.name` is the same string without the hash
  (`space.py:32-38`). The hash covers kwargs and tags, so two candidates differing only in
  hidden sizes get different ids but the same `name`.

Worked example: `configs/compiler/lorenz63.yaml` has 4 latent dims × 5 encoders × 5 dynamics ×
1 hidden size = 100 combinations, minus the 16 `pca` × non-linear-dynamics pairs = **84
candidates**, which is what `results/compile/lorenz63/compile_report.md` reports for the screen
stage.

### Stage keys

`run_cfg_for` (`staged.py:50`) forwards only these keys from a stage entry: `epochs`, `lr`,
`rollout_horizon`, `early_stopping_patience`, `max_batches_per_epoch`, `loss` (into `training`),
and `eval` (merged into the eval block). `name` and `seeds` control the loop, `save_ckpt` controls
checkpointing, and `keep_top` / `keep_frac` / `score_tolerance` control pruning. Any other key in
a stage entry is silently ignored.

Default stages (`configs/compiler/default.yaml:18-22`): `screen` (15 epochs, seed 0,
`keep_frac: 0.4`, `score_tolerance: 0.1`, cheap eval) → `fine` (60 epochs, seed 0, `keep_top: 4`)
→ `final` (150 epochs, seeds 0–2, no pruning). `configs/compiler/lorenz63.yaml` overrides these to
12 / 50 / 120 epochs with `max_batches_per_epoch: 50` on the screen stage.

Pruning (`staged.py:119`) operates on the score-sorted list of candidates with a **finite** J:
candidates with `J = inf` (no completed seed) are always dropped. `keep_frac` keeps
`max(1, ceil(n · frac))`, `keep_top` caps the count, and `score_tolerance` then *raises* the count
back to include every candidate with `J ≤ J_best + tolerance` (an absolute tolerance on J).

## Multi-objective score

Implementation: `nssc.compiler.scorer.MultiObjectiveScorer` (`scorer.py:98`). Metrics are read
with the `val/` prefix — **the selection path never reads `test/` keys**
(`grep -n test src/nssc/compiler/scorer.py src/nssc/search/{staged,space,state}.py` returns
nothing; the only `test/` reference under `search/` is a log line in the benchmark-suite runner,
`runner.py:111`, which is not part of the compiler).

For a candidate `c` scored against the pool of candidates being ranked in the same stage:

    J(c) = λ_recon·L_recon + λ_1step·L_1step + λ_rollout·L_rollout
         + λ_complexity·C_complexity + λ_stability·C_instability
         + λ_blowup·frac_blowup

with, for `f = error_floor` (default **0.01**):

    L_x(c)        = log( (max(x_c, 0) + f) / (max(min_c' x_c', 0) + f) )
    C_complexity  = log( (params_c + 1) / (min_c' params_c' + 1) ) / log 10
    C_instability = val/stability/instability_score      (0.0 if absent)
    frac_blowup   = val/stability/frac_blowup            (0.0 if absent)

The `error_floor` matters: without it a candidate whose best-in-pool error is ~0 (e.g. PCA
reconstruction at `d = D`) would push everyone else's log-ratio towards infinity. It is
configurable as `objective.error_floor`. The complexity term uses a fixed floor of 1.0 parameter
and is divided by `log 10`, so one unit = one decade of parameters (`scorer.py:113-127`).

Metric keys consumed: `val/recon/nrmse`, `val/teacher_forced/nrmse`, the rollout key,
`val/params/total`, `val/stability/instability_score`, `val/stability/frac_blowup`.

- **Rollout key.** With `rollout_horizon_key: auto` (the default) `pick_rollout_key`
  (`scorer.py:58`) selects `val/recursive/nrmse@H` for the largest H present in *every* pooled
  candidate's metrics, falling back to `val/recursive/nrmse_mean`. An explicit key may be given
  instead. The chosen key is stored on every ranking row and in `CompileReport.rollout_key`.
- **NaN handling.** A non-finite log-ratio term (`recon`, `one_step`, `rollout`, `complexity`) is
  replaced by a fixed penalty of **5.0** — never treated as free (`scorer.py:136-140`). The
  instability and blow-up terms cannot be NaN: `_get(..., 0.0)` maps a missing or non-finite value
  to 0.0, so a candidate whose stability analysis did not run is *not* penalised on those two
  terms.
- **Seed aggregation.** `aggregate_seeds` (`scorer.py:74`) takes the mean over completed seeds of
  every scalar summary metric and records `n_seeds` / `n_failed`. Stability is aggregated
  **worst-case, not by majority** (`scorer.py:84-94`): `val/stability/verdict` is the worst verdict
  over seeds under the order `stable < locally_expanding < collapses < explodes < failed`, and the
  aggregate additionally carries `val/stability/verdict_by_seed` (the per-seed list),
  `val/stability/n_unstable_seeds` and `val/stability/frac_blowup_max`. Note that the two scored
  stability terms are still seed **means** — `instability_score` and `frac_blowup` are averaged like
  any other scalar; the worst-case fields are reporting only. A candidate with zero completed seeds
  gets `J = inf` and sorts last.
- **Instability score.** `src/nssc/stability/analysis.py:63-68`:
  `2·frac_blowup + frac_collapse + 0.5·max(0, ρ_max − 1) + 0.25·max(0, log norm_ratio_end)`,
  or **5.0 outright** when `norm_ratio_end` is not finite (the conditional in the source applies to
  the whole sum).

Defaults come from the `ScoreWeights` dataclass (`scorer.py:28`) and are echoed verbatim into the
report's "Selection weights" block:

| field | default | config key |
|---|---|---|
| `reconstruction` | 1.0 | `objective.reconstruction` |
| `one_step` | 1.0 | `objective.one_step` |
| `rollout` | 2.0 | `objective.rollout` |
| `complexity` | 0.1 | `objective.complexity` |
| `stability` | 1.0 | `objective.stability` |
| `blowup_penalty` | 10.0 | `objective.blowup_penalty` |
| `rollout_horizon_key` | `"auto"` | `objective.rollout_horizon_key` |
| `error_floor` | 0.01 | `objective.error_floor` |
| `criterion` | `"multi_objective"` | `objective.criterion` |
| `extra` | `{}` | `objective.extra` |

`configs/compiler/default.yaml:23-30` sets the first six to exactly these values and leaves
`rollout_horizon_key` and `error_floor` at their dataclass defaults. Unknown keys under
`objective:` are silently dropped by `ScoreWeights.from_config` (`scorer.py:42`).

Alternative criteria for ablations (`objective.criterion`): `val_mse`
(`J = ½·L_recon + ½·L_1step`; H2 ablation) and `rollout_only` (`J = L_rollout`). Both still use the
log-ratio normalisation and the 5.0 NaN penalty. Weights are configuration; changing them is a new
experiment — see `configs/compiler/ablations/`, where `lorenz63_valmse`, `lorenz63_rollout_only`,
`lorenz63_nocomplexity` and `lorenz63_nostability` change only the `objective` block (and so reuse
the Lorenz-63 training runs from the registry), while `lorenz63_multiscale` changes the candidate
set and therefore trains new runs.

## Staged search: state, resume and registry reuse

### What the state file actually holds

`SearchState` (`src/nssc/search/state.py`) is a JSON file at `<output_dir>/search_state.json` with
four top-level keys plus timestamps:

- `runs`: `"<stage>|<candidate_id>|<seed>"` → the run result minus its full `metrics` blob, i.e.
  `experiment_id`, `config_hash`, `model`, `seed`, `output_dir`, `checkpoint`, `wall_time_s`,
  `status`, `summary` (flat `val/*` and `test/*` scalars), `candidate` (the full spec) and `stage`.
  A row adopted from the registry instead carries `reused: true` and has no `wall_time_s` — e.g.
  `results/compile/ablations/lorenz63_valmse/search_state.json` has 104 reused rows out of 126.
- `stages`: stage name → `{n_candidates, seeds, ranking, survivors}`, where `ranking` is the full
  per-candidate list for that stage (`candidate_id`, `name`, `rank`, `score`, `terms`, `agg`).
  Eliminated candidates stay in `ranking`; nothing is deleted.
- `meta`: written as `{}` — nothing populates it.
- `created` / `updated`: epoch seconds.

### What resume guarantees

- `_run_one` (`staged.py:64`) skips any `(stage, candidate, seed)` whose recorded status is
  `completed` **or** `failed`. A failed run is therefore *not* retried on resume.
- `nssc compile` resumes by default; `--no-resume` deletes `search_state.json`
  (`compiler.py:155-159`) and nothing else. In particular `profile.json` is still reused, and the
  per-run output directories under `runs/` are overwritten in place.
- Rankings are recomputed from the cached results each time a stage runs, so a resumed search
  replays pruning deterministically from the same numbers.
- Candidate enumeration order is `itertools.product` order (latent dims sorted ascending,
  encoders and dynamics in config order), and Python's stable sort preserves it for score ties.

### What resume does NOT guarantee

- **An interrupted single run is not resumed mid-training.** If the process dies during training,
  no state entry is written, so the next invocation re-executes that run from epoch 0 into the same
  `runs/<stage>/<candidate_id>/seed<N>/` directory, overwriting the partial `history.json`,
  `metrics.json` and `checkpoint/`. A new `EXP-####` id is allocated for the retry and the killed
  run's registry row stays at `status: "running"` forever.
- **A cached `completed` entry is trusted without checking that its checkpoint still exists**
  (`staged.py:66-68`). On a fresh clone the committed `search_state.json` makes the search skip all
  127 Lorenz-63 runs, and `compile()` then fails in `load_checkpoint` because `.gitignore`'s bare
  `*.pt` rule keeps `model.pt` out of the repo (review finding R-02). Only the registry-reuse
  fallback checks the filesystem, and only that the checkpoint *directory* exists — not `model.pt`.
- **No config or dataset hash is verified on resume** — not implemented. Editing `training`,
  `eval`, `windows` or the dataset block and re-running with the default `--resume` silently
  reuses runs trained under the old settings, because the state key contains only
  stage / candidate id / seed. Candidate ids do change when model kwargs change, so those
  candidates are simply retrained; everything else is stale-reusable. Use `--no-resume` (or a
  fresh `output_dir`) after changing the protocol.
- Per-stage `pending|running|done` status, elimination reasons, per-stage wall-clock, git commit
  and config hash in the state file: **not implemented** (`meta` is empty; the stage record has
  only the four keys listed above).
- A summary registry row for the compile run itself (e.g. `model = "nssc_compiled"`):
  **not implemented** — `grep -rn nssc_compiled src` returns nothing. Only the individual
  candidate runs appear in `results/registry.jsonl`.

### Registry reuse (a separate mechanism)

With `reuse_registry: true` (the default, `compiler.py:111`) `StagedSearch._run_one` computes
`run_config_hash(cfg)` (`staged.py:72` → `experiment.py:81`) — the resolved-dataset config with
`output_dir` and `tags` removed, plus `_protocol: PROTOCOL_VERSION` — and looks for a `completed`
registry row with the same hash **and** seed whose checkpoint directory still exists. If found,
that run is adopted without retraining and marked `reused`. This is what lets the ablation configs
re-score the Lorenz-63 runs without re-training them (`tests/integration/test_compiler.py:77`).

Because the hash goes through the same helper `run_experiment` registers with, a reused row is
guaranteed to have been trained from the same config *text*. Two gaps remain: dataclass defaults
(`TrainerConfig`, `EvalConfig`, `LossWeights`) are not hashed field-by-field, so a change to a code
default is only excluded from reuse by manually bumping `PROTOCOL_VERSION` (`experiment.py:38`,
currently 2); and keys that `_dc()` cannot map are dropped rather than rejected, though they are
recorded in `metrics["config/ignored_keys"]` (`experiment.py:143`). This is the residual of review
finding R-18. Separately, `.gitignore` excludes every `*.pt`, so a fresh clone has checkpoint
directories without weights — see the resume caveat above (R-02).

Every candidate run is tagged `compiler`, `stage:<name>` and `cand:<candidate_id>` in the registry
(`staged.py:59`), so `nssc registry --tag compiler` lists them.

## Outputs

Everything lands in the `output_dir` config key (e.g. `results/compile/lorenz63/`) — **not** in
`results/processed/`, which is empty:

    profile.json           DatasetProfile.to_dict()
    candidates.json        list of CandidateSpec dicts
    search_state.json      the state described above
    runs/<stage>/<candidate_id>/seed<N>/
        checkpoint/{model.pt, config.yaml, metadata.json}
        history.json, metrics.json          (error.json instead, if the run failed;
                                             a registry-reused run creates no directory here)
    compile_report.json    CompileReport.to_dict()
    compile_report.md      CompileReport.to_markdown()
    compiler_config.yaml   the fully resolved compiler config
    compiled_model.yaml    {model: <model config>, checkpoint: ..., experiment_id: ...}
                           — written only when a winning checkpoint was found

Figures are produced separately by `nssc visualize --compile-dir <dir>` (into
`<output>/compile_<dirname>/`) or by `python scripts/generate_report.py`, which writes
`results/figures/compile/<name>/` — the figure set is `compiler_decision`, `stage_funnel`,
`pareto`, `latent_dim_sweep`, `family_comparison`, plus the per-run set for the selected
checkpoint (`src/nssc/visualization/figures.py:239`).

## Report format

`CompileReport` (`src/nssc/compiler/report.py:18`) is a dataclass with exactly these fields:

`selected`, `selected_metrics`, `ranking`, `stage_summaries`, `profile`, `weights`, `reasons`,
`n_runs`, `n_failed`, `wall_time_s`, `dataset`, `checkpoint`, `rollout_key`.

`to_markdown()` (`report.py:44`) emits **five** sections — see
`results/compile/lorenz63/compile_report.md` for a real instance (that file was generated before
the worst-seed stability aggregation landed, so its stability bullet still shows the older
majority-verdict wording; re-render it with `nssc report --compile-dir` after a re-compile):

1. **Header** — `# Compile report — <dataset system>`, then the selected latent dimension,
   encoder (+ decoder), dynamics, `val/params/total`, and `Runs: <n_runs> (<n_failed> failed) in
   <wall_time_s/60> min`.
2. **`## Reason`** — bullets from `build_reasons` (`report.py:81`), computed from the final
   ranking: rollout-NRMSE ratio against the best candidate whose *dynamics* family is `linear` or
   `affine`; rollout and parameter-count deltas versus the runner-up; the smallest candidate's
   rollout ratio; a stability line (`stability (worst seed):` verdict, ρ_max, λ_max per step, mean
   and max blow-up fraction, and `n_unstable_seeds / n_seeds`); validation recon and one-step
   NRMSE; and the seed count when more than one seed completed. Every number is read from the
   ranking — no free text.
3. **`## Selection weights`** — the `ScoreWeights` dict, verbatim.
4. **`## Final ranking`** — one row per final-stage candidate: rank, name, J, rollout NRMSE at
   `rollout_key`, one-step, recon, params, ρ_max, stability verdict; followed by the stage funnel
   bullets (`stage <name>: N candidates → M survivors`) from `stage_summaries`.
5. **`## Dataset profile (excerpt)`** — `obs_dim`, `n_traj`, `n_steps`, `suggested_latent_dims`
   and the `recommendations` hints.

`nssc report --compile-dir <dir>` re-renders these same five sections from `compile_report.json`
(`report.py:140`); it does not read `search_state.json`.

Two field semantics that are easy to misread:

- `wall_time_s` is `now − search_state["created"]`, i.e. elapsed time since the search state was
  first written. Across an interrupted-and-resumed search it includes the idle gap — the Lorenz-63
  report's "1324.4 min" is elapsed time, not summed compute.
- `n_runs` counts entries in `search_state["runs"]` across all stages, including registry-reused
  runs that were not retrained.

**Not implemented** in the report (these were specified before the code and do not exist): a
header with git commit / hardware / config hash / date; a candidate-space enumeration section; a
per-stage, per-candidate search trace with `kept|eliminated` and an elimination reason; explicit
raw-vs-normalised score breakdowns or Pareto membership flags; test metrics or latency for the
chosen model; a caveats list; and a reproduce-command section. The per-stage rankings *are*
recorded in `search_state.json` under `stages.<name>.ranking`, so the trace exists as data even
though the markdown does not render it.

## Families available to the compiler (registry names)

Verified with
`python3 -c "import nssc.representations, nssc.dynamics; from nssc.utils.registry import ENCODERS, DECODERS, DYNAMICS; print(sorted(ENCODERS.keys()), sorted(DECODERS.keys()), sorted(DYNAMICS.keys()))"`:

- **Encoders** (`ENCODERS`): `pca`, `linear`, `mlp`, `tcn`, `gru`, `lstm`, `ssm`, `multiscale`.
- **Decoders** (`DECODERS`): `pca`, `linear`, `mlp`.
- **Dynamics** (`DYNAMICS`): `linear`, `affine`, `mlp`, `residual_mlp`, `koopman`, `neural_ode`,
  `ssm`, `gaussian`, `multiscale`.

These are the exact strings accepted in `candidates.encoders` / `candidates.dynamics` /
`candidates.decoders`; anything else raises `KeyError: unknown key '<x>'. Available: [...]` from
`Registry.get` (`src/nssc/utils/registry.py:39`). Registration sites: `@ENCODERS.register(...)` in
`src/nssc/representations/{pca,linear,mlp,tcn,rnn,ssm,multiscale}.py` (`rnn.py` holds both `gru`
and `lstm`), `@DECODERS.register(...)` in `src/nssc/representations/{pca,linear,mlp}.py`, and
`@DYNAMICS.register(...)` in `src/nssc/dynamics/{linear,mlp,koopman,neural_ode,ssm,gaussian,multiscale}.py`
(`linear.py` holds `linear` and `affine`, `mlp.py` holds `mlp` and `residual_mlp`).
`configs/compiler/default.yaml` exercises a subset (`pca, linear, mlp, tcn, gru, ssm` ×
`linear, affine, residual_mlp, koopman, neural_ode, ssm`).

Family-specific code: `nssc/compiler/{compiler,scorer,profiler}.py`, `nssc/search/staged.py` and
`nssc/search/state.py` contain no family names, so new families are picked up through the registry
alone (see the extension guide in `docs/architecture.md`). Two places do special-case families and
must be checked when adding one: `nssc/search/space.py` (default decoder map, the `hidden_dims`
applicability list, the `pca_only_linear` rule, the `slow_dim` consistency checks) and
`nssc/compiler/report.py:92`, which treats `linear`/`affine` dynamics as "the linear baseline"
when writing the reasons.

## CLI

Verified against `src/nssc/cli/main.py` and `nssc <cmd> --help`:

    nssc compile   --config configs/compiler/lorenz63.yaml [--output DIR] [--set a.b=c]
                   [--device cpu] [--no-resume]
    nssc profile   --config configs/datasets/lorenz63.yaml [--output profile.json]
    nssc report    --experiment EXP-0353 | --compile-dir results/compile/lorenz63
    nssc registry  [--status completed] [--tag compiler] [--limit 50]
    nssc visualize --compile-dir results/compile/lorenz63 [--output results/figures]
    nssc evaluate  --experiment EXP-0353 [--split test] [--context N] [--device cpu]
    nssc smoke                             # tiny single-run smoke experiment (used by `make smoke`)

`-c`, `-o`, `-s`, `-e` are the short forms of `--config`, `--output`, `--set`, `--experiment`.
Notes on flags that do *not* exist: `nssc compile` has no `--seed` and no `--dry-run`; resume is a
default-on boolean flag (`--resume` / `--no-resume`), not an opt-in; `nssc profile` takes
`--config`, not `--dataset`; `nssc registry` has no `list` / `show` subcommands.

Compiler configs live in `configs/compiler/`: `default.yaml`, `tiny.yaml` (CI, < 1 min CPU),
`lorenz63.yaml`, `lorenz63_highdim.yaml`, `lorenz96.yaml`, `vanderpol.yaml`, `eegbci.yaml`, and
`ablations/*.yaml`. There is no `configs/experiments/compile_*.yaml`.

`nssc compile` never touches the test split for selection and never prints test metrics. To get
held-out numbers for a compiled model, either call `StateSpaceCompiler.evaluate(compiled,
split="test")` from Python, or run `nssc evaluate --experiment <id>` with the `experiment_id`
recorded in `compiled_model.yaml`.

## Known limitations

1. **The committed `profile.json` files predate the train-split fix and are still
   whole-dataset profiles.** `fit()` now profiles `ds.split()["train"]` only (`compiler.py:72-82`),
   implementing the code fix recorded for review finding R-17 — the profile drives the candidate
   latent-dim grid whenever `candidates.latent_dims: auto` (`space.py:70-79`), so computing it on
   all trajectories put held-out data into model selection. But `fit()` still short-circuits on an
   existing `profile.json`, and all nine committed ones (`results/compile/**/profile.json`) report
   `n_traj: 100` — every trajectory, e.g. 100 of 100 for Lorenz-63, whose split is 0.7/0.15/0.15 —
   and none has a `computed_on` key. They are pre-fix artefacts and will be reloaded as-is unless
   deleted. The affected compile runs all set explicit `latent_dims`, so their *selections* never
   consulted the profile; the profile excerpt in their reports is nonetheless test-inclusive.
   Delete `profile.json` (or the whole output dir) to regenerate a train-split profile.
2. **Test metrics are computed for every candidate run.** `run_experiment` evaluates both `val`
   and `test` and stores both in the registry row and in `search_state.json` `summary`
   (`experiment.py:127-132`, `SUMMARY_KEYS` at `experiment.py:173`). Nothing in the selection
   path reads them, but they are present on disk, so treat any test number found there as
   "already computed", not as a fresh held-out evaluation.
3. **Resume has no protocol guard** and does not resume a run mid-training — see
   "What resume does NOT guarantee" above.
4. **Registry reuse can still adopt a run trained under different code defaults**: the config
   hash covers the config text plus `PROTOCOL_VERSION`, not the individual dataclass defaults, so
   excluding a protocol change from reuse requires bumping that constant by hand (residual R-18).
5. **The report omits provenance** (git commit, hardware, config hash, reproduce command) and the
   per-candidate search trace; both are partially recoverable from `results/registry.jsonl` and
   `search_state.json` respectively.
6. **Only one seed's checkpoint is kept as "the" compiled model.** `compiled_model.yaml` points at
   the best-validation-rollout seed of the winner; the ranking that selected it is a mean over
   seeds. Per-seed variance is in `search_state.json`, not in the report.
7. **All seeds share one train/val/test split** (the split seed comes from the dataset config), so
   seed-to-seed variance reflects initialisation and batch order only (R-39).

## Research-integrity rules for this subsystem

- Selection is on validation only. Never add a `test/` key to the scorer, to
  `pick_rollout_key`, or to any pruning rule.
- Weights and criteria are configuration. Changing them is a new experiment with a new
  `output_dir`, not an edit to an existing compile run.
- Eliminated and failed candidates stay in `search_state.json` and in the per-stage rankings.
  Failed runs keep `status: "failed"` in the registry and are written up in
  `research/failures.md`; they are never deleted.
- Every number quoted from a compile run must come from `compile_report.json`,
  `search_state.json` or `results/registry.jsonl` — never from a re-run that was not registered.
- Do not claim a selected latent dimension corresponds to a physical variable count unless an
  experiment in this repo demonstrates it; the profiler's `suggested_latent_dims` is a heuristic.
