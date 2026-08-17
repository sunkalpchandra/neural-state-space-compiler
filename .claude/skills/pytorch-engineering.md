# Skill: pytorch-engineering

## Purpose
Engineering rules for all `torch` code in `nssc` (representations, dynamics, training,
evaluation, stability). Keeps shapes, devices, determinism, and checkpoints uniform so
the compiler can swap components without special cases.

## Relevant theory
- Sequence tensors are `(batch, time, dim)` = `(B, T, D)` for observations and
  `(B, T, d)` for latents. Single-step calls take `(B, d)`. Everything else is a
  documented exception.
- Recurrent/latent rollouts are sequential in time; teacher-forced training can be
  parallel over time for non-recurrent encoders. Recursive rollout with gradient
  through H steps is BPTT: memory O(H·B·d), gradient explosion for unstable F_θ —
  which is exactly why the stability term exists.
- Autograd Jacobians: `torch.func.jacrev(F)(z)` for `(d,) → (d,)`, vmapped over batch
  (`torch.func.vmap`) — used by `nssc.stability`. Falls back to finite differences on
  MPS if `torch.func` ops are unsupported.
- Mixed precision is off by default: rollouts amplify fp16 error; stability metrics need
  fp32 (or fp64 for spectral radius checks).

## Project-specific conventions
- Interfaces (see `docs/architecture.md`):
  `Encoder.encode(x: (B,T,D)) -> z: (B,T,d)`,
  `Decoder.decode(z: (B,T,d)) -> x_hat: (B,T,D)`,
  `Dynamics.step(z: (B,d), u: (B,m)|None) -> (B,d)`,
  `Dynamics.rollout(z0: (B,d), horizon: int, u: (B,H,m)|None) -> (B,H,d)`,
  `Dynamics.jacobian(z: (B,d)) -> (B,d,d)`,
  `LatentModel(encoder, dynamics, decoder)`.
- All modules take a config dataclass in `__init__` (`from nssc.utils.config import ...`)
  and expose `.config`, `.latent_dim`, `.obs_dim`, `.n_params()`.
- Devices: `nssc.utils.hardware.resolve_device(pref)` → `"cuda"` > `"mps"` > `"cpu"`,
  overridable by `--device`. MPS is *optional* (decision D-006): every code path must
  run on CPU; MPS-unsupported ops (`float64`, some `torch.func`, `torch.linalg.eig`)
  are computed on CPU copies. Tests run on CPU.
- Determinism: `nssc.utils.seeding.seed_everything(seed)` sets `random`, `numpy`,
  `torch`, `torch.cuda`, `torch.mps` seeds, `torch.use_deterministic_algorithms(True,
  warn_only=True)`, `CUBLAS_WORKSPACE_CONFIG=:4096:8`. DataLoaders use a seeded
  `torch.Generator` and `worker_init_fn`. Two runs with the same seed on CPU must give
  identical metrics (regression test); on MPS/CUDA agreement to 1e-4 is acceptable and
  documented.
- Checkpoint format (`nssc.training.checkpoint`): a directory
  `results/raw/<EXP-id>/<candidate-id>/seed<k>/` containing
  `model.pt` (`{"state_dict": ..., "optimizer": ..., "epoch": int, "best_val": float}`),
  `config.yaml` (resolved model+train config), and `metadata.json`
  (`nssc_version, git_commit, config_hash, seed, device, torch_version, n_params,
  train_time_s, created_at, dataset_hash`). `load_checkpoint(dir)` rebuilds the model
  from `config.yaml` via the registry, then loads weights — never pickles modules.
- Loss functions in `nssc.training.losses` return a dict `{"total": ..., "recon": ...,
  "onestep": ..., "rollout": ..., "stability": ..., "complexity": ...}` so the scorer
  and logs use the same names as `docs/compiler.md`.
- Normalization statistics (per-dim mean/std from *train* trajectories) are stored in
  the checkpoint and applied inside `LatentModel`, so a loaded model consumes raw data.

## Implementation requirements
- Shapes asserted at module boundaries with informative messages (`x.shape[-1] ==
  self.obs_dim`).
- No `.item()` / `.cpu()` inside training loops except at logging points; no per-step
  Python lists of tensors in rollouts — preallocate `(B,H,d)`.
- Rollout training uses configurable `rollout_horizon_train` (e.g. 10–25) with optional
  curriculum; evaluation horizons come from the experiment config.
- Gradient clipping (`clip_grad_norm_`, default 1.0) on all rollout losses.
- `torch.no_grad()` / `torch.inference_mode()` for evaluation; `model.eval()` — GRU
  dropout and BatchNorm otherwise leak train behaviour into rollouts.
- Any custom `nn.Module` must pass: forward finite, backward finite, param count matches
  `n_params()`, save/load round-trip gives identical outputs (unit test template in
  `tests/unit/conftest.py`).
- `torch.compile` and AMP are opt-in flags in the train config, off by default.

## Common failure modes
- Mixing `(T,B,D)` (PyTorch RNN default) and `(B,T,D)`: always `batch_first=True`.
- Rollout that re-encodes decoded outputs (`E(D(z))`) when the protocol says latent
  rollout — the evaluation mode label must match the code path (see `experiment-design`).
- Nondeterminism from `torch.backends.cudnn.benchmark=True`, unordered `set` iteration
  in candidate generation, or `time`-seeded numpy in observation maps.
- MPS silently producing float32 for float64 requests → spectral radius drift; compute
  stability on CPU.
- Checkpoint saved with `torch.save(model)` (pickled class) — breaks after refactors.
- Forgetting to store normalization stats → loaded model evaluates on differently
  scaled data and "regresses".
- In-place ops on tensors needed for Jacobians (`z += ...`) breaking `torch.func`.

## Validation checklist
- [ ] Docstrings state shapes; `(B,T,D)` unless documented.
- [ ] Runs on CPU; MPS/CUDA optional and only via `--device`.
- [ ] `seed_everything` called; CPU determinism test passes.
- [ ] Checkpoint dir has `model.pt`, `config.yaml`, `metadata.json`; round-trip test.
- [ ] Losses returned as named dict; names match `docs/compiler.md`.
- [ ] Eval under `inference_mode` and `model.eval()`.
- [ ] No heavy optional imports at module import time (`mne`, `fastapi`, `plotly`).


## Lessons learned (2026-08-17)
- MPS has no float64: eigen-decompositions (`nssc.stability.spectral`) run on CPU double.
- Neural-ODE candidates with 4 RK4 substeps were 20× slower than residual MLPs in the
  compiler screen; screening uses `n_substeps: 1` (config, not code default).
- Sequence baselines with recursive rollout losses re-run the whole backbone per step
  (TCN/Transformer/SSM 10–90× slower); baselines train teacher-forced only (D-008).
