# nssc architecture

`nssc` compiles a high-dimensional multivariate time series `x_{1:T}`, `x_t ∈ R^D`, into
a compact latent state-space model

    z_t     = E_φ(x_≤t)          encoder / state inference,  z_t ∈ R^d, d ≪ D
    z_{t+1} = F_θ(z_t, u_t)      latent dynamics
    x̂_t     = D_ψ(z_t)           decoder

by profiling the data, enumerating candidate `(d, encoder, dynamics, decoder)`
configurations from a registry, training them under a staged, resumable search, and
scoring them with a multi-objective criterion (see `docs/compiler.md`).

## 1. Dataflow

```
 configs/datasets/*.yaml           configs/compiler/*.yaml        configs/models/*.yaml
          │                                 │                              │
          ▼                                 ▼                              ▼
 ┌─────────────────┐   (n_traj,T,D)  ┌──────────────────┐        ┌────────────────────┐
 │  nssc.data      │───────────────▶│ DatasetProfiler   │        │ baselines          │
 │  generators /   │  splits: train/ │ (nssc.compiler)  │        │ (experiments/      │
 │  loaders /      │  val/test(_ood) │ D, T, spectrum,  │        │  benchmarks)       │
 │  observation    │                 │ PCA energy, ACF, │        └─────────┬──────────┘
 │  splits / cache │                 │ est. intrinsic d │                  │
 └─────────────────┘                 └────────┬─────────┘                  │
                                              │ DatasetProfile             │
                                              ▼                            │
                                     ┌──────────────────┐                  │
   nssc.utils.registry  ───────────▶ │ CandidateGenerator│                 │
   (encoders, dynamics, decoders,    │ d-grid × encoder  │                 │
    systems, losses, schedules)      │ × dynamics × dec  │                 │
                                     └────────┬─────────┘                  │
                                              │ list[Candidate]            │
                                              ▼                            │
                                     ┌──────────────────┐   per candidate  │
                                     │ StagedSearch     │──┐               │
                                     │ (nssc.search)    │  │ ┌──────────────────────┐
                                     │ coarse → fine →  │  └▶│ Trainer (nssc.training)│
                                     │ long-horizon →   │    │ LatentModel = E+F+D    │
                                     │ stability → final│◀───│ losses, ckpt           │
                                     └────────┬─────────┘    └──────────┬───────────┘
                                              │                          │ checkpoints
                              ┌───────────────┼───────────────┐         ▼
                              ▼               ▼               ▼    results/raw/EXP-*/
                     ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
                     │ Evaluator    │ │ Stability    │ │ metrics      │
                     │ (evaluation) │ │ Analyzer     │ │ (nssc.metrics)│
                     │ tf/recursive/│ │ (stability)  │ │ recon,kstep, │
                     │ direct       │ │ J, ρ(J), λ₁  │ │ nrmse,vpt,.. │
                     └──────┬───────┘ └──────┬───────┘ └──────┬───────┘
                            └────────────────┼────────────────┘
                                             ▼
                                   ┌───────────────────┐
                                   │ MultiObjectiveScorer│  J = Σ λ_i · term_i
                                   └─────────┬─────────┘
                                             ▼
                          ┌──────────────────────────────────────┐
                          │ CompiledModel + CompileReport         │
                          │ results/raw/EXP-*/compiled/           │
                          │ results/processed/EXP-*/compile_report│
                          │ results/registry.jsonl (one row/run)  │
                          └──────────────────────────────────────┘
                                             │
                     nssc.visualization ──────┴────── scripts/make_tables.py
                     results/figures/F1..F10          results/tables/*.md|csv
```

## 2. Package boundaries (`src/nssc/`)

| package | may import | must not import |
|---|---|---|
| `utils/` (config, seeding, registry, hashing, gitinfo, hardware) | stdlib, numpy, torch, yaml | anything else in nssc |
| `data/` | utils | representations, dynamics, compiler |
| `representations/`, `dynamics/` | utils | compiler, search, training |
| `metrics/`, `stability/`, `uncertainty/` | utils, dynamics interfaces | compiler, search |
| `training/` | utils, metrics, representations, dynamics | compiler, search |
| `evaluation/` | utils, metrics, stability, training (load ckpt) | compiler, search |
| `compiler/` | everything above | cli, visualization |
| `search/` | compiler, training, evaluation, utils | cli |
| `visualization/` | utils (registry loading), pandas, matplotlib | torch models |
| `cli/` | everything | — |

Heavy optional deps (`mne`, `fastapi`, `uvicorn`, `plotly`) are imported inside
functions only. `dashboard/` is outside the package.

## 3. Core interfaces

All tensors `(batch, time, dim)` unless stated. Every module takes a config dataclass
and exposes `.config`, `.latent_dim`, `.obs_dim`, `.n_params()`.

```python
class Encoder(nn.Module):
    """x (B,T,D) -> z (B,T,d). Causal unless config.causal is False (then only valid
    for teacher-forced reconstruction and labeled as such)."""
    def encode(self, x: Tensor) -> Tensor: ...
    def encode_context(self, x: Tensor) -> Tensor:   # (B,c,D) -> z_c (B,d), state at end of context
        ...

class Decoder(nn.Module):
    """z (B,T,d) -> x_hat (B,T,D)."""
    def decode(self, z: Tensor) -> Tensor: ...

class Dynamics(nn.Module):
    """Latent transition F_theta."""
    def step(self, z: Tensor, u: Tensor | None = None) -> Tensor:          # (B,d)[,(B,m)] -> (B,d)
    def rollout(self, z0: Tensor, horizon: int, u: Tensor | None = None) -> Tensor:  # -> (B,H,d)
    def jacobian(self, z: Tensor) -> Tensor:                                # (B,d) -> (B,d,d)
    # optional: def sample(self, z, u=None) -> z_next for Gaussian dynamics; .is_stochastic

class LatentModel(nn.Module):
    """Bundles encoder + dynamics + decoder + normalization stats."""
    encoder: Encoder; dynamics: Dynamics; decoder: Decoder
    def reconstruct(self, x) -> Tensor                      # D(E(x)), (B,T,D)
    def predict_teacher_forced(self, x) -> Tensor           # D(F(E(x)_t)) for t<T, (B,T-1,D)
    def rollout(self, x_context, horizon) -> tuple[Tensor, Tensor]  # (x_hat (B,H,D), z (B,H,d))
    def n_params(self) -> int; def complexity(self) -> ComplexityInfo  # params, latent_dim, flops/step
```

Multi-scale slow/fast encoders return `z` with `config.slow_dims` / `config.fast_dims`
partitions and the paired dynamics module updates slow coordinates every `k` steps; the
interface above is unchanged (`d = d_slow + d_fast`).

Compiler-side interfaces:

```python
@dataclass
class DatasetProfile:  D, T, n_traj, sampling_dt, pca_energy_curve, est_intrinsic_dim,
                       autocorr_time, spectral_slope, is_stationary, noise_est, notes

@dataclass
class Candidate:       candidate_id, latent_dim, encoder: str, encoder_cfg, dynamics: str,
                       dynamics_cfg, decoder: str, decoder_cfg, train_cfg

class StateSpaceCompiler:
    def profile(dataset) -> DatasetProfile
    def candidates(profile) -> list[Candidate]           # via CandidateGenerator + registry
    def search(candidates, dataset) -> SearchState        # StagedSearch, resumable
    def evaluate(model, dataset) -> dict                  # Evaluator, all modes/horizons
    def stability(model, dataset) -> dict                 # StabilityAnalyzer
    def score(metrics, complexity, stability) -> ScoreBreakdown
    def report(state) -> CompileReport                    # md + json
    def compile(dataset) -> tuple[CompiledModel, CompileReport]   # runs all stages
```

Stages, in order: `profile → candidates → staged search (train per candidate) →
evaluate → stability → score → report`. See `docs/compiler.md`.

## 4. Registry mechanism (`nssc.utils.registry`)

```python
from nssc.utils.registry import register, get, list_registered

@register("dynamics", "koopman")
class KoopmanDynamics(Dynamics): ...

get("dynamics", "koopman")            # -> class
list_registered("dynamics")           # -> ["linear", "affine", "mlp", "residual", "koopman", ...]
```

Kinds: `system`, `observation_map`, `encoder`, `decoder`, `dynamics`, `loss`,
`schedule`, `metric`, `figure`. Registration happens at import of the defining module;
`nssc.representations.__init__` and `nssc.dynamics.__init__` import their submodules so
`import nssc` populates the registry. The `CandidateGenerator` reads
`configs/compiler/*.yaml` → `candidate_space: {encoders: [...], dynamics: [...],
latent_dims: [...]}` where entries are registry names or `"all"`. Nothing in
`nssc/compiler` or `nssc/search` names a concrete family.

Test: registering a dummy dynamics in a test module makes it appear in
`StateSpaceCompiler.candidates()` output.

## 5. Config system (`nssc.utils.config`)

YAML files → frozen dataclasses with defaults; unknown keys are errors.

```
DatasetConfig     system, params, param_range_train, param_range_test, dt, T, n_traj,
                  burn_in, obs_map, obs_dim, noise_std, split{train,val,test,n_test_ood},
                  seed_offsets, real{path, subjects, preprocessing}
ModelConfig       encoder{name, **cfg}, dynamics{name, **cfg}, decoder{name, **cfg},
                  latent_dim, normalize
TrainConfig       max_epochs, patience, monitor, batch_size, lr, weight_decay, schedule,
                  rollout_horizon_train, curriculum, loss_weights{recon,onestep,rollout,
                  stability}, grad_clip, amp, compile, device
CompilerConfig    candidate_space, stages{coarse,fine,long,stability}{budget,keep,thresholds},
                  score_weights{lambda1..lambda5}, normalization, horizons, seeds
EvalConfig        modes, horizons, context_length, n_eval_trajectories, attractor_stats
ExperimentConfig  experiment{name,hypothesis,matrix_cell}, dataset, models|compiler,
                  seeds, training, evaluation, selection
```

`config_hash = sha256(canonical_json(resolved_config))[:12]` where canonical JSON sorts
keys and resolves all defaults, so any protocol change changes the hash. `nssc config
resolve X.yaml` prints the resolved config and hash.

## 6. Experiment registry (`results/registry.jsonl`)

Append-only JSON lines, one per run (`experiment × candidate/model × seed`). Ids
`EXP-0001`, `EXP-0002`, … allocated by `next_experiment_id()` (max existing + 1; never
reused). Row schema:

```json
{"experiment_id": "EXP-0012", "run_id": "EXP-0012/gru_h128/seed3",
 "hypothesis": "H1", "matrix_cell": "D", "status": "completed|failed|running|invalid|preliminary",
 "git_commit": "abc123", "git_dirty": false, "config_hash": "9f2c...", "dataset_hash": "1a77...",
 "dataset": "lorenz63", "model": "gru", "candidate_id": "gru_h128", "seed": 3,
 "params": {"hidden": 128, "...": "..."}, "n_params": 51203, "latent_dim": 128,
 "train_time_s": 412.3, "hardware": {"device": "cpu", "cpu": "...", "torch": "2.3.0"},
 "metrics": {"recursive": {"nrmse@1": 0.01, "nrmse@100": 0.42}, "teacher_forced": {...},
             "spectral_radius_mean": 0.98, "diverged_frac": 0.0},
 "checkpoint": "results/raw/EXP-0012/gru_h128/seed3", "note": "",
 "started_at": "2026-08-17T10:00:00Z", "finished_at": "..."}
```

`load_results()` flattens to a tidy DataFrame (`mode, horizon, metric, value` long
format) for tables and figures. Rows are never edited in place; corrections append a new
row with `supersedes: run_id`.

## 7. Checkpoint format

Directory `results/raw/<EXP-id>/<candidate_id>/seed<k>/`:

- `model.pt` — `torch.save({"state_dict": model.state_dict(), "optimizer": ...,
  "epoch": int, "best_val": float, "norm_stats": {"mean": ..., "std": ...}})`
- `config.yaml` — resolved `ModelConfig` + `TrainConfig` (enough to rebuild via registry)
- `metadata.json` — `nssc_version, git_commit, git_dirty, config_hash, dataset_hash,
  seed, device, torch_version, n_params, latent_dim, train_time_s, created_at`

`load_checkpoint(dir, device)` → `LatentModel` (rebuilt from `config.yaml` via
registry, weights loaded, norm stats attached). Compiled model additionally saved under
`results/raw/<EXP-id>/compiled/` with `compile_report.json` next to it.

## 8. Extension guide: adding a new dynamics family

1. Create `src/nssc/dynamics/my_family.py`:
   ```python
   @dataclass
   class MyFamilyConfig(DynamicsConfig): latent_dim: int = 8; hidden: int = 64

   @register("dynamics", "my_family")
   class MyFamily(Dynamics):
       config_cls = MyFamilyConfig
       def step(self, z, u=None): ...        # (B,d) -> (B,d)
       def rollout(self, z0, horizon, u=None): return default_rollout(self, z0, horizon, u)
       def jacobian(self, z): return autograd_jacobian(self.step, z)   # or analytic
   ```
2. Import it in `src/nssc/dynamics/__init__.py`.
3. Add `configs/models/my_family_example.yaml` (optional) and, if it should be a compiler
   candidate, add `"my_family"` to `candidate_space.dynamics` in the relevant
   `configs/compiler/*.yaml` (or leave `"all"`).
4. Run `pytest tests/unit/test_interfaces.py -k my_family` — the parametrized suite
   picks it up automatically (shapes, finiteness, grad, jacobian-vs-finite-diff,
   round-trip, determinism).
5. Add a docstring with the transition equation and any stability property (e.g.
   spectral norm constraint) — the compile report prints it.
6. Add a line to `docs/compiler.md` §"Families" and, if it changes complexity accounting,
   to `nssc.metrics.complexity`.

No edit to `nssc/compiler` or `nssc/search` is needed; if one is, that is a bug in the
compiler.

Same procedure for encoders (`@register("encoder", ...)`, implement `encode`,
`encode_context`), decoders, systems (`@register("system", ...)`, provide `f(x,p)` and
`SystemSpec`), and figures (`@register("figure", "fig11_...")`).
