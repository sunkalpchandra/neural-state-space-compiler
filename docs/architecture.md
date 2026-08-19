# nssc architecture

`nssc` compiles a high-dimensional multivariate time series `x_{1:T}`, `x_t ∈ R^D`, into
a compact latent state-space model

    z_t     = E_φ(x_t)           encoder / state inference,  z_t ∈ R^d
    z_{t+1} = F_θ(z_t, u_t)      latent dynamics
    x̂_t     = D_ψ(z_t)           decoder

by profiling the data, enumerating candidate `(d, encoder, dynamics, decoder)`
configurations from the component registries, training each under a staged, resumable
search, and scoring them with a multi-objective criterion (see `docs/compiler.md`).

Two caveats up front, both load-bearing for how the rest of this document reads:

* `d ≪ D` is the *intent*, not an enforced invariant. `resolve_latent_dims`
  (`src/nssc/search/space.py:70`) clips **auto** latent dims to `obs_dim`, but an explicit
  `latent_dims:` list is honoured as-is, including overcomplete latents.
* `E_φ` is written above as a function of `x_t`, not `x_{≤t}`, because three of the eight
  registered encoders (`linear`, `mlp`, `pca`) are **pointwise**: `z_t` depends on `x_t`
  alone. The other five (`gru`, `lstm`, `ssm`, `tcn`, `multiscale`) do consume history.
  All eight declare `is_causal = True`. The flags `Encoder.is_causal` /
  `Encoder.is_pointwise` / `Encoder.requires_fit`
  (`src/nssc/representations/base.py:15-17`) are the source of truth, and
  `tests/unit/test_representations.py:101,116` parametrizes the causality and
  pointwise-ness tests over them.

## 1. Dataflow

```
configs/datasets/*.yaml    configs/models/{encoders,dynamics,baselines}/*.yaml
configs/compiler/*.yaml    configs/experiments/*.yaml, .../benchmarks/*.yaml
         │
         ▼  nssc.utils.config.load_config   (`_base_` deep-merge + `--set a.b=c`)
 ┌───────────────────────┐
 │ nssc.data.builder     │ build_dataset(cfg) → TrajectoryDataset (N,T,D)
 │  systems / observation│ .split()      trajectory-level train/val/test
 │  splits / real loaders│ .normalize()  with *train* statistics only
 └──────────┬────────────┘
            │
            ▼
 ┌───────────────────────┐  DatasetProfile.to_dict()
 │ nssc.compiler.profiler│──────────────┐
 │  profile_dataset()    │              │
 └───────────────────────┘              ▼
                              ┌──────────────────────────┐
 ENCODERS / DECODERS /   ────▶│ nssc.search.space        │
 DYNAMICS registries          │  generate_candidates()   │
 (nssc.utils.registry)        └────────────┬─────────────┘
                                           │ list[CandidateSpec]
                                           ▼
                              ┌──────────────────────────┐
                              │ nssc.search.staged       │ stages: screen → fine → final
                              │  StagedSearch.run()      │ resumable: search_state.json
                              └────────────┬─────────────┘
                                           │ one run per (candidate, seed)
                                           ▼
 ┌─────────────────────────────────────────────────────────────────────────────┐
 │ nssc.experiment.run_experiment(cfg)   ← the single-run entrypoint, also used │
 │   models.builder.build_latent_model     by `nssc train` and benchmark suites │
 │   training.Trainer.fit                                                       │
 │   evaluation.evaluate_model   modes: recon | teacher_forced | recursive      │
 │        └─ metrics.* + stability.analyze_stability                            │
 │   training.save_checkpoint         → <output_dir>/checkpoint/                │
 │   utils.experiment_registry        → results/registry.jsonl (register → …)   │
 └───────────────────────────────┬─────────────────────────────────────────────┘
                                 │ flat per-run summary (nssc.experiment.summarize)
                                 ▼
                       ┌──────────────────────┐
                       │ compiler.scorer      │  .rank() → rows sorted by J
                       │ MultiObjectiveScorer │  J = Σ λ_i · term_i (see docs/compiler.md)
                       └──────────┬───────────┘
                                  ▼
                  ┌───────────────────────────────────────────┐
                  │ CompiledModel + CompileReport             │
                  │ results/compile/<name>/                   │
                  │   profile.json      candidates.json       │
                  │   search_state.json compiler_config.yaml  │
                  │   compile_report.json / .md               │
                  │   compiled_model.yaml  (pointer, not a    │
                  │       copy of the weights)                │
                  │   runs/<stage>/<cand_id>/seed<k>/         │
                  └───────────────────┬───────────────────────┘
                                      │
  nssc.baselines (SequenceForecaster comparators, no latent bottleneck)
  nssc.search.runner.run_suite  ──────┤ writes into the same registry
                                      ▼
             ┌────────────────────────────────────────────────────────┐
             │ reporting — generated, never hand-edited                │
             │  nssc tables   --suite S → results/tables/S.{md,json}   │
             │  nssc pareto   --suite S → results/tables/pareto_S.*    │
             │  nssc failures           → results/tables/failures.md   │
             │  scripts/generate_report.py                             │
             │    → results/figures/compile/<name>/*.{png,pdf}         │
             │    → results/figures/suites/<suite>/*.{png,pdf}         │
             │    → results/SUMMARY.md                                 │
             └────────────────────────────────────────────────────────┘
```

`scripts/generate_report.py` only writes the `compile/` and `suites/` trees (plus
`results/SUMMARY.md`). The third tree, `results/figures/reference/<EXP-id>/`, is the
per-experiment figure set for a single reference run, produced by
`nssc visualize --experiment <EXP-id> -o results/figures/reference` —
`visualize_experiment` appends `/<experiment>` to whatever `-o` it is given
(`visualization/cli_hooks.py:20`), so the default `nssc visualize -e EXP-0044` would
instead land in `results/figures/EXP-0044/`.

`results/processed/` exists but is currently empty — per-run artefacts live next to their
checkpoints under `results/raw/**` and `results/compile/**`.

## 2. Package boundaries (`src/nssc/`)

The middle column is what the code *actually* imports today (verified by grepping
`from nssc.` across each package), not an aspiration.

| package | imports from `nssc` | must not import |
|---|---|---|
| `utils/` (config, seeding, registry, hashing, io, env) | — | anything else in nssc |
| `metrics/` | — (imports nothing from nssc today) | anything but `utils` |
| `data/` | utils | representations, dynamics, models, training, compiler |
| `representations/`, `dynamics/` | utils | data, models, training, search, compiler |
| `models/` | representations, dynamics, utils | training, evaluation, search, compiler |
| `training/` | models, utils | evaluation, search, compiler, cli |
| `stability/` | dynamics, models | training, search, compiler |
| `uncertainty/` | metrics, models | search, compiler |
| `evaluation/` | data, metrics, models, stability, training, utils, `experiment` | compiler, search, cli |
| `experiment.py` | data, evaluation, models, training, utils | compiler, search, cli |
| `baselines/` | data, metrics, models, representations, utils, `experiment` | compiler, search, cli |
| `search/` | compiler (scorer), experiment, utils, baselines (lazily) | cli, visualization |
| `compiler/` | data, evaluation, experiment, models, search, training, utils | cli, visualization |
| `visualization/` | evaluation, experiment, stability, training, uncertainty, utils | — |
| `cli/` | everything (all imports inside command bodies) | — |

Notes on the two non-obvious edges:

* **`compiler` ↔ `search` is a genuine cycle.** `compiler/compiler.py:28-29` imports
  `nssc.search.space` and `nssc.search.staged`, while `search/staged.py:25` imports
  `nssc.compiler.scorer`. It is broken by a lazy `__getattr__` in
  `src/nssc/search/__init__.py` that defers importing `StagedSearch` until first access.
* **`experiment.py` is a module, not a package**, and sits below `search`/`compiler` but
  above `training`/`evaluation`. Several packages reach into it for `prepare_data` /
  `resolve_dataset_cfg` (`evaluation/ood.py:25`, `baselines/run.py:35`,
  `visualization/figures.py:128`, `compiler/compiler.py:26`).

Heavy optional deps are imported inside functions only: `mne` at
`data/real/eegbci.py:140`, `uvicorn` at `dashboard/app.py:458`. They are declared as
extras in `pyproject.toml` (`dashboard`, `eeg`). `dashboard/` is outside the package.

## 3. Core interfaces

Tensors are `(batch, time, dim)` unless stated. Components are constructed with **plain
keyword arguments**, not config dataclasses — there is no `Encoder.config`,
`.n_params()` or `.complexity()`. Parameter counts come from `num_parameters()`.

```python
# src/nssc/representations/base.py
class Encoder(nn.Module):
    is_causal: bool = True       # class attributes, declarative
    is_pointwise: bool = True    # z_t depends only on x_t
    requires_fit: bool = False   # e.g. PCA: closed-form fit before/instead of SGD

    def __init__(self, obs_dim: int, latent_dim: int) -> None: ...
    def forward(self, x: Tensor) -> Tensor            # (B,T,D) -> (B,T,d)
    def encode(self, x: Tensor) -> Tensor             # alias for forward
    @torch.no_grad()
    def fit(self, x: Tensor) -> None                  # closed-form hook; default no-op
    def num_parameters(self) -> int

class Decoder(nn.Module):
    def __init__(self, latent_dim: int, obs_dim: int) -> None: ...
    def forward(self, z: Tensor) -> Tensor            # (B,T,d) -> (B,T,D)
    def decode(self, z: Tensor) -> Tensor             # alias for forward
    def num_parameters(self) -> int

# src/nssc/dynamics/base.py
class Dynamics(nn.Module):
    is_linear: bool = False
    is_stochastic: bool = False

    def __init__(self, latent_dim: int, control_dim: int = 0) -> None: ...
    def step(self, z, u=None) -> Tensor               # (B,d)[,(B,m)] -> (B,d)   REQUIRED
    def forward(self, z, u=None) -> Tensor            # delegates to step
    def rollout(self, z0, horizon, u=None) -> Tensor  # (B,d) -> (B,H,d), base loop
    def step_sequence(self, z, u=None) -> Tensor      # (B,T,d) -> (B,T,d), teacher-forced
    def jacobian(self, z, u=None) -> Tensor           # (B,d) -> (B,d,d), vmap+jacrev default
    def num_parameters(self) -> int
    def extra_losses(self) -> dict[str, Tensor]       # model-specific regularisers; default {}
```

Only `step` is mandatory; `rollout`, `step_sequence` and `jacobian` have working base
implementations. `extra_losses()` is summed into the training objective with per-key
weights from `LossWeights.extra` (`src/nssc/training/losses.py:87-89`).

```python
# src/nssc/models/latent_model.py
class LatentModel(nn.Module):
    def __init__(self, encoder, dynamics, decoder, config: dict | None = None) -> None: ...
    encoder: Encoder; dynamics: Dynamics; decoder: Decoder
    config: dict                                    # the model config dict, for checkpoints
    latent_dim: int; obs_dim: int                   # properties, read off the encoder

    def encode(self, x) -> Tensor                   # (B,T,D) -> (B,T,d)
    def decode(self, z) -> Tensor
    def reconstruct(self, x) -> Tensor              # D(E(x)), (B,T,D)
    def predict_teacher_forced(self, x) -> tuple[Tensor, Tensor, Tensor]
        # (x̂_{2:T} (B,T-1,D), z_{1:T} (B,T,d), ẑ_{2:T} (B,T-1,d))
    def rollout(self, x_context, horizon, u=None) -> tuple[Tensor, Tensor]   # (x̂ (B,H,D), ẑ (B,H,d))
    def rollout_from_latent(self, z0, horizon, u=None) -> tuple[Tensor, Tensor]
    def num_parameters(self) -> dict[str, int]      # {"encoder","dynamics","decoder","total"}
    @torch.no_grad()
    def latent_trajectory(self, x) -> Tensor
```

`predict_teacher_forced` returning a **3-tuple** and `num_parameters` returning a **dict**
are easy to get wrong; `evaluation/evaluator.py:71` (`xn, _, _ = …`) and the
`model.num_parameters()["total"]` call in `run_experiment` show the call sites.

Multi-scale encoders (`ENCODERS["multiscale"]`) and dynamics (`DYNAMICS["multiscale"]`)
split the latent as `z = [z_slow, z_fast]` with `slow_dim` the first coordinates
(`representations/multiscale.py:60`, `dynamics/multiscale.py:44`). Both require
`0 < slow_dim < latent_dim`, and `generate_candidates` drops candidates where an encoder
and dynamics disagree on `slow_dim` (`search/space.py:110-114`). The interface above is
unchanged (`d = slow_dim + fast_dim`).

Compiler-side interfaces:

```python
# src/nssc/compiler/profiler.py — a dataclass of 36 fields, all listed below
@dataclass
class DatasetProfile:
    n_traj, n_steps, obs_dim, dt, total_samples, has_missing, missing_rate, sampling_rate_hz,
    mean_min, mean_max, std_min, std_median, std_max, dynamic_range,
    pca_dims_for_variance, explained_variance_curve, mle_dim_k10, mle_dim_k20,
    correlation_dim, suggested_latent_dims,
    autocorr, autocorr_time, smoothness, dominant_period_steps, spectral_flatness,
    noise_std_estimate, signal_std, noise_ratio_estimate,
    nonstationarity_mean, nonstationarity_std, nonstationary_dim_fraction,
    linear_predictability_r2, linear_r2_at_10_steps,
    lyapunov_proxy, lyapunov_proxy_per_time, recommendations
    def to_dict(self) -> dict; def to_markdown(self) -> str

# src/nssc/search/space.py
@dataclass(frozen=True)
class CandidateSpec:
    latent_dim: int; encoder: str; dynamics: str; decoder: str = "mlp"
    encoder_kwargs / decoder_kwargs / dynamics_kwargs / training_overrides: dict
    tags: tuple[str, ...] = ()
    id: str      # property: "<enc>+<dyn>@d<k>-<stable_hash6>"
    name: str    # property: "<enc>+<dyn>@d<k>"
    def to_dict(self); @classmethod from_dict(cls, d); def model_config(self) -> dict

# src/nssc/compiler/compiler.py
@dataclass
class CompiledModel:
    model: LatentModel | None; spec: CandidateSpec; report: CompileReport
    checkpoint: str | None; output_dir: str
    def rollout(self, x_context, horizon)

class StateSpaceCompiler:
    def __init__(self, cfg, device=None, registry=None, log=print) -> None
    def fit(self) -> dict                       # stage 1: profile (cached in output_dir/profile.json)
    def propose(self) -> list[CandidateSpec]    # stages 2–4: enumerate candidates
    def base_run_cfg(self) -> dict              # the dataset/windows/training/eval block shared by runs
    def search(self) -> dict                    # stages 5–6: StagedSearch + scoring
    def compile(self, search_result=None) -> CompiledModel     # select winner, load ckpt, write report
    def run(self, resume: bool = True) -> CompiledModel        # fit → propose → search → compile
    def evaluate(self, compiled, split="test") -> dict         # held-out eval of the compiled model
```

`compile()` returns a single `CompiledModel`; the report is reachable as
`compiled.report`. There is **no** `profile()`, `candidates()`, `stability()`, `score()`
or `report()` method on the compiler — profiling is `fit()`, candidate enumeration is
`propose()`, and stability/scoring happen inside `evaluate_model` and
`MultiObjectiveScorer` respectively. `nssc compile -c <yaml>` calls `run()`
(`cli/main.py:144`).

## 4. Registry mechanism (`nssc.utils.registry`)

`registry.py` defines one generic `Registry` class and **six module-level instances**.
There are no free `register` / `get` / `list_registered` functions.

```python
from nssc.utils.registry import DYNAMICS      # or ENCODERS, DECODERS, SYSTEMS, DATASETS, BASELINES

@DYNAMICS.register("koopman")                  # key defaults to cls.__name__ if omitted
class KoopmanDynamics(Dynamics): ...

DYNAMICS.get("koopman")                        # -> class (KeyError lists available keys)
DYNAMICS.build("koopman", latent_dim=8)        # -> instance
DYNAMICS.keys()                                # -> sorted list[str]
"koopman" in DYNAMICS                          # __contains__ / __iter__ / __len__ also defined
```

`register` sets `cls.registry_key` and raises `KeyError` on a duplicate key bound to a
different class.

| registry | populated by importing | keys today |
|---|---|---|
| `ENCODERS` | `nssc.representations` | `gru, linear, lstm, mlp, multiscale, pca, ssm, tcn` |
| `DECODERS` | `nssc.representations` | `linear, mlp, pca` |
| `DYNAMICS` | `nssc.dynamics` | `affine, gaussian, koopman, linear, mlp, multiscale, neural_ode, residual_mlp, ssm` |
| `SYSTEMS` | `nssc.data` / `nssc.data.systems` | `coupled_oscillators, damped_oscillator, fitzhugh_nagumo, gray_scott, harmonic, kuramoto, lorenz63, lorenz96, lotka_volterra, pendulum, vanderpol` |
| `BASELINES` | `nssc.baselines` | `gru, lstm, mean, persistence, ssm, tcn, transformer` |
| `DATASETS` | — | **empty**: declared but nothing registers into it |

There are **no** registries for losses, schedules, metrics, figures or observation maps.
Observation maps are looked up through a plain module-level dict,
`nssc.data.observation.OBS_MAPS` (`data/observation.py:193`), via
`ObservationMap.from_config`.

`import nssc` populates nothing — `src/nssc/__init__.py` only sets `__version__`.
Registration is a side effect of importing the defining packages, which
`nssc.models.builder._ensure_registries()` does explicitly (`models/builder.py:19-21`)
before every `build_latent_model` call.

Candidate enumeration reads the compiler config's **`candidates:`** block (not
`candidate_space:`); `generate_candidates` (`search/space.py:82`) accepts:

```yaml
candidates:
  latent_dims: auto            # or an explicit list; "auto" reads the profile's suggestions
  encoders:  [pca, linear, mlp, tcn, gru, ssm]     # str, or {name: x, kwargs: {...}}
  dynamics:  [linear, affine, residual_mlp, koopman, neural_ode, ssm]
  decoders:  {}                # optional encoder → decoder mapping; default mlp (pca→pca, linear→linear)
  hidden_dims: [64]            # optional; applied to mlp-like components, tagged h<k>
  exclude: []                  # optional list of partial {encoder, dynamics, latent_dim} matches
  max_candidates: null         # optional truncation
```

The string `"all"` is **not implemented** — every entry must be a real registry key or
`Registry.get` raises `KeyError`. Two filters are applied by default:
multi-scale `slow_dim` consistency, and `pca_only_linear: true`, which skips PCA paired
with non-linear dynamics (`search/space.py:115`).

Nothing in `nssc/compiler` or `nssc/search` names a concrete family; the defaults live in
`configs/compiler/*.yaml` and in the fallback literals of `generate_candidates`
(`encoders: ["mlp"]`, `dynamics: ["residual_mlp"]`).

Test: `tests/unit/test_utils_registry.py`, plus `tests/unit/test_dynamics.py` and
`tests/unit/test_representations.py`, which parametrize their whole suites over
`DYNAMICS.keys()` / `ENCODERS.keys()` / `DECODERS.keys()`.

## 5. Config system (`nssc.utils.config`)

YAML files load into a plain **`Config(dict)`** with attribute access — *not* frozen
dataclasses. There is no schema validation at load time.

```python
load_config(path, overrides=["training.lr=1e-3"], base=None) -> Config
```

* `_base_: other.yaml` (a path or list of paths, relative to the file) is deep-merged
  *underneath* the file's own keys; bases resolve recursively (`config.py:81-88`).
* CLI overrides are `a.b.c=value` strings parsed with YAML semantics
  (`parse_override`, `config.py:91`), exposed as `--set` / `-s` on `nssc train`,
  `nssc compile`, `nssc benchmark` and `nssc data generate`.
* `Config` adds `get_path("a.b")`, `set_path("a.b", v)`, `to_dict()` and
  `hash()` = `stable_hash(to_dict())` = first 12 hex chars of the SHA-256 of the
  canonical JSON (`utils/hashing.py:28-31`).
* `save_yaml(cfg, path)` writes a resolved config back out.

There is **no `nssc config` command**. To see a resolved config and its hash, use the
library directly, or read the copy every run already writes: `compiler_config.yaml` for a
compile run, and the full `config` field of the registry row for any run.

Sub-trees are consumed by typed dataclasses in the owning subsystem:

| YAML block | dataclass | file |
|---|---|---|
| `training:` | `TrainerConfig` | `training/trainer.py:25` |
| `training.loss:` | `LossWeights` | `training/losses.py:24` |
| `eval:` | `EvalConfig` | `evaluation/evaluator.py:34` |
| `objective:` | `ScoreWeights` | `compiler/scorer.py:29` |

Real field names (a frequent source of drift):

```
TrainerConfig  epochs, lr, weight_decay, grad_clip, scheduler(cosine|plateau|none),
               warmup_epochs, early_stopping_patience, rollout_horizon, rollout_curriculum,
               curriculum_epochs, val_fixed_horizon, rollout_stride, loss{...}, log_every,
               max_batches_per_epoch, device, amp, compile
LossWeights    recon, latent_1step, obs_1step, rollout, stability, extra{key: weight}
EvalConfig     context, horizons, max_horizon, batch_size, stability, stability_horizon,
               latency, latency_horizon, divergence_threshold, extra
ScoreWeights   reconstruction, one_step, rollout, complexity, stability, blowup_penalty,
               rollout_horizon_key, error_floor, criterion(multi_objective|val_mse|
               rollout_only), extra
```

A run config additionally has `dataset:` (the `nssc.data.builder` schema, or
`{_file: configs/datasets/x.yaml}`), `model:` (`latent_dim`, `encoder`, `decoder`,
`dynamics`), `windows: {context, horizon, stride, batch_size}`, `seed:`, `tags:` and
`output_dir:` — see the module docstring of `src/nssc/experiment.py`. A compiler config
adds `candidates:`, `stages:`, `objective:` and `output_dir:` — see
`configs/compiler/default.yaml`.

### Known limitation: unknown keys are dropped, not rejected

`nssc.experiment._dc` (`experiment.py:48-56`) filters an incoming dict down to the
dataclass's declared field names before constructing it. Nothing rejects the leftovers,
so a misspelt key — `epochs: 40` written as `n_epochs: 40`, or `rollout_weight` placed at
`training:` level instead of under `training.loss` — does **not** raise; the run proceeds
on defaults. This is deliberate: a suite's `training:` block feeds both latent runs and
baseline runs, whose dataclasses have different fields, so unknown keys have to be
tolerated. The key is still hashed into `config_hash`, so two runs differing only in a
typo get different ids but identical behaviour.

Partial mitigation, added in response to review finding R-28: `_dc` now *returns* the
ignored keys alongside the object, and `run_experiment` records them as
`metrics["config/ignored_keys"] = {"training": [...], "eval": [...]}`
(`experiment.py:143`). Check that field when a config change appears to have had no
effect. Its reach is limited, and the gaps are worth knowing:

* It lands in `<output_dir>/metrics.json` only. `summarize()` does not copy it, so the
  **registry row does not carry it** — you cannot audit typos from `registry.jsonl`.
* `StateSpaceCompiler.evaluate` discards the list (`compiler/compiler.py:172`,
  `ecfg, _ignored = …`).
* `nssc.baselines.run` keeps its own single-return `_dc` (`baselines/run.py:47-49`), so
  baseline runs report nothing.
* `ScoreWeights.from_config` (`scorer.py:43-46`) still filters `objective:` keys with no
  report at all.

Load-time schema validation is **not implemented**, and neither is a warning on the log.

The places that *do* raise on an unknown key:

* `training.loss` keys — `Trainer.__init__` splats them into `LossWeights(**…)`
  (`trainer.py:52`), so an unknown loss key raises `TypeError`.
* `dataset.params` keys — `DynamicalSystem.__init__` raises `KeyError` on unknown
  parameters (`data/systems/base.py:40-42`).
* Unknown *registry* names anywhere — `Registry.get` raises `KeyError`.

### Protocol version

`nssc.experiment.PROTOCOL_VERSION` (currently `2`) is folded into every latent run's
config hash as `_protocol` (`run_config_hash`, `experiment.py:81-87`) and echoed into
`metrics["config/protocol_version"]`. It versions the *semantics* of a run rather than
its configuration, so bumping it stops the search's registry-reuse path from serving a
v1 run for a v2 request while leaving the old rows in the ledger. `v1 → v2` changed
validation to be evaluated at the full `rollout_horizon` instead of the current
curriculum horizon (`research/failures.md` F-007). Baseline runs have no equivalent:
`baseline_config_hash` (`baselines/run.py:104`) does not include a protocol version.

## 6. Experiment registry (`results/registry.jsonl`)

Append-only JSON lines. Ids `EXP-0001`, `EXP-0002`, … are allocated by
`ExperimentRegistry.next_id()` = max existing + 1, never reused; `register()` takes an
exclusive `flock` on `results/registry.jsonl.lock` while it reads the max id and appends,
so parallel compiles cannot collide.

A run appends **at least twice**: once at `register()` with `status="running"`, then
again at `complete()` / `fail()` / `update()` with the same `experiment_id`.
`ExperimentRegistry.records()` therefore deduplicates, keeping the **last** line per id.
Nothing is ever edited or deleted in place — a failed run stays in the file with
`status="failed"` (CLAUDE.md rule). There is no `supersedes` mechanism; a correction is
a new `experiment_id`.

`ExperimentRecord` (`utils/experiment_registry.py:48`) has exactly these fields. Below is
the real row `EXP-0648`, with `metrics` and `config` abridged (the row carries 35 metric
keys and the full resolved config):

```json
{"experiment_id": "EXP-0648",
 "git_commit": "208cd95d9c20f61d2dfe3eb148c4eff84124b389-dirty",
 "config_hash": "2bd856b8decb",
 "dataset": "lorenz96", "model": "linear+ssm@d8", "seed": 0,
 "status": "completed",
 "metrics": {"val/recon/nrmse": 0.1928, "val/recursive/nrmse@50": 0.6014,
             "val/params/total": 424, "val/latent_dim": 8,
             "val/stability/rho_max": 1.0363, "val/stability/verdict": "stable",
             "test/recursive/nrmse@50": 0.5930, "test/recursive/divergence_time": 51.0,
             "train/best_val_loss": 0.2848, "train/epochs_run": 12, "train/time_s": 6.68},
 "checkpoint": "results/compile/lorenz96/runs/screen/linear+ssm@d8-9b6f01/seed0/checkpoint",
 "config": {"dataset": {...}, "windows": {...}, "training": {...}, "eval": {...},
            "model": {...}, "seed": 0, "tags": [...], "output_dir": "..."},
 "param_count": 424, "train_time_s": 6.682393625000259,
 "hardware": {"platform": "macOS-14.6.1-arm64-arm-64bit", "machine": "arm64",
              "python": "3.12.0", "torch": "2.13.0", "numpy": "2.2.5",
              "cuda": false, "mps": true},
 "created_at": 1787118091.386174, "updated_at": 1787118098.607733,
 "tags": ["cand:linear+ssm@d8-9b6f01", "compiler", "h96", "lorenz96", "stage:screen"],
 "notes": ""}
```

Note this row's `hardware` has **no** `device` key: it predates that addition. Rows
written since then carry it.

Points worth stating plainly, because earlier drafts of this file got them wrong:

* `status` is only `running | completed | failed`. There is no `invalid` or
  `preliminary`.
* There is no `run_id`, `hypothesis`, `matrix_cell`, `git_dirty`, `dataset_hash`,
  `n_params`, `latent_dim`, `started_at`, `finished_at` or `supersedes` field.
  Dirtiness is a `-dirty` suffix on `git_commit` (`utils/env.py:30`); the parameter count
  is `param_count`; timestamps are the float epoch seconds `created_at`/`updated_at`;
  the latent dim is inside `metrics` as `<split>/latent_dim`.
* `hardware` is `hardware_info()` (`utils/env.py:43`) plus a `device` key holding the
  device the run actually used, passed in by the caller
  (`experiment.py:106`, `baselines/run.py:137`). Older rows predate that key.
* `metrics` is the **flat summary**, not the full metrics dict: `summarize()`
  (`experiment.py:182`) keeps the scalar `SUMMARY_KEYS` prefixed by split. The full
  nested metrics — including per-step `curves` — are written to
  `<output_dir>/metrics.json`.
* Provenance identifiers are `config_hash` and the dataset's own
  `metadata['version']` (a `stable_hash` of the resolved dataset config,
  `data/builder.py:24-26`). A separate top-level `dataset_hash` field does not exist.

There is no `load_results()` helper. Reading goes through `ExperimentRegistry`'s own
accessors — `records()`, `get(id)`, `find(**filters)` and `find_by_hash(config_hash, seed)`,
the last being how the staged search and the suite runner skip a run that already
completed.

Aggregation across runs lives in `nssc.evaluation.aggregate`: `group_runs`, `mean_std`,
`bootstrap_ci`, `paired_test`, `summary_table`, `pareto_front`, `format_markdown`,
`load_groups`. `evaluation/tables.py:8` and `evaluation/pareto.py:14` build on those, and
they are in turn what `nssc tables`, `nssc pareto` and `scripts/generate_report.py` call.

## 7. Checkpoint format and on-disk layout

A checkpoint is a **directory**, written by `save_checkpoint`
(`training/checkpoint.py:23`) at `<output_dir>/checkpoint/`:

- `model.pt` — `torch.save(model.state_dict(), …)`, a **bare state dict**. No optimizer
  state, epoch counter or best-val score is stored; training is not resumable from a
  checkpoint (only the *search* is resumable, via `search_state.json`).
- `config.yaml` — `model.config`, i.e. the model config dict plus `obs_dim`, which is
  everything `build_latent_model` needs to rebuild the architecture through the
  registries.
- `metadata.json` — free-form. `run_experiment` stores `experiment_id`, `seed`,
  `norm_stats` (`{mean, std}` as lists), the resolved `dataset` config and
  `metrics_summary` (`experiment.py:151-154`).

```python
load_checkpoint(path, map_location="cpu") -> tuple[LatentModel, dict]
```

It returns **`(model, metadata)`**, with the model rebuilt from `config.yaml`, weights
loaded with `weights_only=True`, and `.eval()` already called. Normalisation statistics
live in the metadata dict and are **not** re-attached to the model — callers that need
un-normalised outputs must apply them themselves.

Baselines use a parallel but distinct format: `save_forecaster_checkpoint`
(`baselines/run.py:80`) writes `model.pt`, `config.json` (note: JSON, with
`baseline`/`kwargs`/`mode`/`direct_horizon`/`obs_dim`/`resolved`) and `metadata.json`,
loaded by `load_forecaster_checkpoint` → `(model, metadata)`.

Where runs land:

| producer | output_dir |
|---|---|
| `nssc train` / `run_experiment` default | `results/raw/<EXP-id>/` |
| benchmark suite (`run_suite`) | `results/raw/benchmarks/<suite>/<dataset>/<model>/seed<k>/` |
| compiler search run | `results/compile/<name>/runs/<stage>/<candidate_id>/seed<k>/` |
| `nssc smoke` | `results/raw/smoke/` (and a *separate* ledger, `results/registry_smoke.jsonl`) |

Each run directory holds `checkpoint/`, `metrics.json`, `history.json` (and `error.json`
on failure). The compiled model is **not copied** anywhere: `compile()` writes
`compiled_model.yaml` in the compile output dir containing the winner's `model` config,
its `checkpoint` path and its `experiment_id` (`compiler/compiler.py:150-151`) — a
pointer into `runs/`. Model weights are gitignored everywhere: `.gitignore` carries a bare
`*.pt` rule plus `results/raw/**/*.{pt,npz,npy}`, `results/raw/EXP-*/` and
`results/raw/smoke/`.

## 8. Extension guide: adding a new dynamics family

1. Create `src/nssc/dynamics/my_family.py`:

   ```python
   from torch import Tensor
   from nssc.dynamics.base import Dynamics
   from nssc.utils.registry import DYNAMICS

   @DYNAMICS.register("my_family")
   class MyFamilyDynamics(Dynamics):
       is_linear = False          # set True if F is linear in z (used by tests + reporting)

       def __init__(self, latent_dim: int, control_dim: int = 0, hidden_dims=(64, 64)) -> None:
           super().__init__(latent_dim, control_dim)
           ...                    # plain kwargs; no config dataclass, no `config_cls`

       def step(self, z: Tensor, u: Tensor | None = None) -> Tensor:
           ...                    # (B,d) -> (B,d).  This is the only required method.

       # Optional overrides — the base class already provides working versions:
       #   rollout(z0, horizon, u)   loops step
       #   step_sequence(z, u)       flattens (B,T,d) and applies step
       #   jacobian(z, u)            torch.func.vmap(jacrev(step))
       #   extra_losses()            -> {} ; return {"my_reg": tensor} to add a regulariser
   ```

   There is no `DynamicsConfig`, `config_cls`, `default_rollout` or `autograd_jacobian`
   helper — those never existed. Use `nssc.dynamics._mlp_util.make_mlp` for the usual MLP
   body, as `dynamics/mlp.py` and `dynamics/multiscale.py` do.

2. Import it in `src/nssc/dynamics/__init__.py` (for the registration side effect) and
   add the class to `__all__`.

3. Add `configs/models/dynamics/my_family.yaml` (optional, alongside the existing nine)
   and, to make it a compiler candidate, add `"my_family"` to `candidates.dynamics` in
   the relevant `configs/compiler/*.yaml`. There is no `"all"` shorthand.

4. Run `pytest tests/unit/test_dynamics.py -k my_family`. That suite parametrizes over
   `DYNAMICS.keys()`, so the new key is picked up automatically for shapes, control
   input, gradient flow, jacobian-vs-finite-differences, `state_dict` round-trip and the
   `is_linear`/`num_parameters` contract. If the default constructor is slow, add an
   entry to the `SMALL` kwargs dict at `tests/unit/test_dynamics.py:19`.

5. Add a docstring with the transition equation and any stability property (e.g. a
   spectral-norm constraint). Family names surface in the compile report through
   `CandidateSpec.name` and in figures through `nssc.visualization.style.family_of`.

6. Add a line to `docs/compiler.md` under "Families available to the compiler",
   and — if it changes how complexity is accounted — check `nssc.metrics.complexity.estimate_flops_per_step`
   handles it (it returns `None` when it cannot estimate).

No edit to `nssc/compiler` or `nssc/search` is needed; if one is, that is a bug in the
compiler.

The same procedure applies to:

* **encoders** — `@ENCODERS.register("name")` on an `Encoder` subclass taking
  `(obs_dim, latent_dim, **kwargs)`, implementing `forward`; set `is_causal` /
  `is_pointwise` / `requires_fit` honestly, since the tests branch on them. Import in
  `src/nssc/representations/__init__.py`. Tests: `tests/unit/test_representations.py`.
* **decoders** — `@DECODERS.register("name")` on a `Decoder` subclass taking
  `(latent_dim, obs_dim, **kwargs)`. Also add a `_DEFAULT_DECODER` pairing in
  `search/space.py:17` if the encoder needs a non-`mlp` partner.
* **systems** — `@SYSTEMS.register("name")` on a `DynamicalSystem` subclass
  (`data/systems/base.py`) setting `name`, `state_dim`, `default_params`, `default_dt`,
  `default_transient` and implementing `f(t, x, params)` and `sample_initial(rng, n)`.
  Import in `src/nssc/data/systems/__init__.py`. Tests: `tests/unit/test_data_systems.py`.
* **baselines** — `@BASELINES.register("name")` on a `SequenceForecaster` subclass
  (`baselines/base.py`) implementing `backbone(x)` and `feature_dim`, then calling
  `self._build_heads()`. Tests: `tests/unit/test_baselines.py`.

There is **no figure registry**. New figures are plain functions in
`nssc/visualization/` wired into `figures_for_experiment` / `figures_for_compile` /
`figures_for_suite` (`visualization/figures.py`), which is what
`scripts/generate_report.py` and `nssc visualize` call.
