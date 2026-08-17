# `nssc.training`

### `nssc.training`

Training: losses, trainer loop, checkpoints.

### `nssc.training.checkpoint`

Checkpoint format: a directory containing

    model.pt        torch state_dict of the LatentModel
    config.yaml     model config (encoder/decoder/dynamics/latent_dim/obs_dim)
    metadata.json   free-form metadata (metrics, seed, git commit, norm stats, ...)

``load_checkpoint`` rebuilds the model from ``config.yaml`` via the registries.

#### `load_checkpoint(path: 'str | Path', map_location: 'str | torch.device' = 'cpu') -> 'tuple[LatentModel, dict[str, Any]]'`

#### `save_checkpoint(model: 'LatentModel', path: 'str | Path', metadata: 'dict[str, Any] | None' = None) -> 'Path'`

### `nssc.training.losses`

Composite loss for latent state-space models.

    L = w_recon · ||x − D(E(x))||²
      + w_latent_1step · ||E(x)_{t+1} − F(E(x)_t)||²          (latent consistency)
      + w_obs_1step   · ||x_{t+1} − D(F(E(x)_t))||²           (one-step prediction)
      + w_rollout     · mean_k ||x_{t+k} − D(F^k(E(x)_t))||²  (multi-step, k ≤ H_train)
      + w_stability   · stability regulariser (Jacobian spectral radius surrogate)
      + Σ dynamics.extra_losses()  (e.g. Koopman consistency)

Rollout horizon can follow a curriculum (``rollout_horizon`` grows over epochs).

#### class `LatentDynamicsLoss(weights: 'LossWeights', rollout_horizon: 'int' = 10, rollout_stride: 'int' = 1, stability_target: 'float' = 1.0, stability_samples: 'int' = 16, detach_rollout_targets: 'bool' = True) -> 'None'`

- `stability_penalty(self, model: 'LatentModel', z: 'Tensor') -> 'Tensor'` — Penalise Jacobian spectral radius (via power-iteration-free surrogate: ||J v||/||v||

#### class `LossWeights(recon: 'float' = 1.0, latent_1step: 'float' = 1.0, obs_1step: 'float' = 1.0, rollout: 'float' = 1.0, stability: 'float' = 0.0, extra: 'dict[str, float]' = <factory>) -> None`

LossWeights(recon: 'float' = 1.0, latent_1step: 'float' = 1.0, obs_1step: 'float' = 1.0, rollout: 'float' = 1.0, stability: 'float' = 0.0, extra: 'dict[str, float]' = <factory>)


### `nssc.training.trainer`

Minimal, deterministic trainer for LatentModel.

Features: AdamW, cosine/plateau LR schedule, grad clipping, rollout-horizon
curriculum, early stopping on validation loss, best-checkpoint restore, per-epoch
history, wall-clock accounting, optional closed-form PCA fit before SGD.

#### class `Trainer(model: 'LatentModel', cfg: 'TrainerConfig', device: 'torch.device | None' = None) -> 'None'`

- `evaluate(self, loader: 'DataLoader') -> 'dict[str, float]'`
- `fit(self, train_loader: 'DataLoader', val_loader: 'DataLoader | None' = None, log=None) -> 'dict[str, Any]'`
- `maybe_closed_form_fit(self, loader: 'DataLoader', max_batches: 'int' = 50) -> 'None'` — PCA-style encoders: fit on a sample of training windows before SGD.
- `train_epoch(self, loader: 'DataLoader', epoch: 'int') -> 'dict[str, float]'`

#### class `TrainerConfig(epochs: 'int' = 50, lr: 'float' = 0.001, weight_decay: 'float' = 1e-05, grad_clip: 'float' = 1.0, scheduler: 'str' = 'cosine', warmup_epochs: 'int' = 0, early_stopping_patience: 'int' = 20, rollout_horizon: 'int' = 10, rollout_curriculum: 'bool' = True, curriculum_epochs: 'int | None' = None, rollout_stride: 'int' = 4, loss: 'dict[str, Any]' = <factory>, log_every: 'int' = 10, max_batches_per_epoch: 'int | None' = None, device: 'str | None' = None, amp: 'bool' = False) -> None`

TrainerConfig(epochs: 'int' = 50, lr: 'float' = 0.001, weight_decay: 'float' = 1e-05, grad_clip: 'float' = 1.0, scheduler: 'str' = 'cosine', warmup_epochs: 'int' = 0, early_stopping_patience: 'int' = 20, rollout_horizon: 'int' = 10, rollout_curriculum: 'bool' = True, curriculum_epochs: 'int | None' = None, rollout_stride: 'int' = 4, loss: 'dict[str, Any]' = <factory>, log_every: 'int' = 10, max_batches_per_epoch: 'int | None' = None, device: 'str | None' = None, amp: 'bool' = False)

