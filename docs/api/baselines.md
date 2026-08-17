# `nssc.baselines`

### `nssc.baselines`

Sequence-model baselines (comparators for the central hypothesis) and trivial baselines.

Importing this package populates ``nssc.utils.registry.BASELINES`` with keys
``persistence, mean, gru, lstm, tcn, transformer, ssm``.

#### `build_baseline(key: 'str', obs_dim: 'int', **kw: 'Any') -> 'SequenceForecaster'`

``BASELINES.build(key, obs_dim=obs_dim, **kw)`` (kw includes ``mode``/``direct_horizon``).

### `nssc.baselines.base`

Common interface for "large sequence model" comparators and trivial baselines.

A :class:`SequenceForecaster` maps an observation history directly to future
observations, with **no latent state-space bottleneck**. It is the comparator
for the central hypothesis (compiled latent SSM vs. plain sequence model).

Two prediction modes (fixed at construction):

* ``recursive`` — the model predicts one step ahead; :meth:`forecast` appends the
  prediction to the context and repeats (autoregressive rollout).
* ``direct``    — a second head predicts ``direct_horizon`` steps at once from
  the context; horizons longer than that are covered by block-wise recursion.

Neural subclasses only implement :meth:`backbone` — causal per-position features
``(B,T,F)`` — plus ``feature_dim``; the base class supplies the heads and
everything else. Non-parametric baselines override the ``predict_*`` methods.

Shapes: ``x`` is ``(B, T, D)`` (batch, time, obs_dim) unless stated otherwise.

#### class `SequenceForecaster(obs_dim: 'int', mode: 'str' = 'recursive', direct_horizon: 'int | None' = None, **_: 'object') -> 'None'`

Base forecaster: ``x_{≤t} → x̂_{t+1}`` (and optionally ``x̂_{t+1:t+H}``).

- `backbone(self, x: 'Tensor') -> 'Tensor'` — Causal features ``(B,T,F)``: position ``t`` depends on ``x_{≤t}`` only.
- `config(self) -> 'dict'` — Constructor kwargs needed to rebuild (subclasses extend).
- `forecast(self, x_context: 'Tensor', horizon: 'int') -> 'Tensor'` — ``(B,C,D), H → (B,H,D)``: predictions for the ``H`` steps after the context.
- `num_parameters(self) -> 'int'`
- `predict_direct(self, x_context: 'Tensor') -> 'Tensor'` — ``(B,C,D) → (B,direct_horizon,D)`` (direct mode only).
- `predict_next(self, x: 'Tensor') -> 'Tensor'` — ``(B,T,D) → (B,D)``: prediction of the step following the window.
- `predict_next_sequence(self, x: 'Tensor') -> 'Tensor'` — ``(B,T,D) → (B,T,D)``: entry ``t`` is the prediction of ``x_{t+1}`` from ``x_{≤t}``.

### `nssc.baselines.evaluate`

Evaluate a :class:`SequenceForecaster` with the same protocol/keys as
:func:`nssc.evaluation.evaluator.evaluate_model` (minus recon/stability, which
have no meaning without a latent state).

Keys: ``recursive/nrmse@k``, ``recursive/nrmse_step@k``, ``recursive/nrmse_mean``,
``recursive/divergence_time``, ``recursive/horizon``, ``recursive/context``,
``curves.recursive_nrmse``, ``teacher_forced/mse|nrmse`` (one step ahead from
ground-truth history at every position, like ``evaluate_model``), ``params/total``,
``latent_dim`` (None, or the wrapped model's), ``latency/step_latency_ms_*`` (one
``predict_next`` call) and, in direct mode, ``direct/nrmse@k`` / ``direct/nrmse_step@k``.

#### `evaluate_forecaster(model: 'SequenceForecaster', x: 'Tensor', context: 'int' = 20, horizons: 'Sequence[int]' = (1, 5, 10, 25, 50, 100, 250, 500), sigma: 'np.ndarray | None' = None, device: 'torch.device | None' = None, max_horizon: 'int | None' = None, batch_size: 'int' = 64, divergence_threshold: 'float' = 1.0, latency: 'bool' = True) -> 'dict[str, Any]'`

``x``: (N,T,D) held-out trajectories normalised like the training data.

### `nssc.baselines.latent_wrapper`

Adapt a :class:`nssc.models.LatentModel` to the :class:`SequenceForecaster` interface.

Lets :func:`nssc.baselines.evaluate.evaluate_forecaster` score compiled latent
models and sequence baselines with the identical protocol. Not registered in
``BASELINES`` (it is not a candidate to build from a config).

#### class `LatentModelForecaster(model: 'LatentModel') -> 'None'`

Base forecaster: ``x_{≤t} → x̂_{t+1}`` (and optionally ``x̂_{t+1:t+H}``).

- `forecast(self, x_context: 'Tensor', horizon: 'int') -> 'Tensor'` — ``(B,C,D), H → (B,H,D)``: predictions for the ``H`` steps after the context.
- `num_parameters(self) -> 'int'`
- `predict_next(self, x: 'Tensor') -> 'Tensor'` — ``(B,T,D) → (B,D)``: prediction of the step following the window.
- `predict_next_sequence(self, x: 'Tensor') -> 'Tensor'` — ``(B,T,D) → (B,T,D)``: entry ``t`` is the prediction of ``x_{t+1}`` from ``x_{≤t}``.

### `nssc.baselines.persistence`

Trivial baselines: persistence (repeat last value) and training-set mean.

#### class `MeanBaseline(obs_dim: 'int', **kw: 'object') -> 'None'`

``x̂ = mean_train`` (per dimension). The mean is set by :meth:`fit`; zero trainable params.

- `fit(self, x: 'Tensor') -> 'None'` — ``x``: (N,T,D) or (B,L,D) training data → per-dim mean.
- `forecast(self, x_context: 'Tensor', horizon: 'int') -> 'Tensor'` — ``(B,C,D), H → (B,H,D)``: predictions for the ``H`` steps after the context.
- `predict_direct(self, x_context: 'Tensor') -> 'Tensor'` — ``(B,C,D) → (B,direct_horizon,D)`` (direct mode only).
- `predict_next(self, x: 'Tensor') -> 'Tensor'` — ``(B,T,D) → (B,D)``: prediction of the step following the window.
- `predict_next_sequence(self, x: 'Tensor') -> 'Tensor'` — ``(B,T,D) → (B,T,D)``: entry ``t`` is the prediction of ``x_{t+1}`` from ``x_{≤t}``.

#### class `PersistenceBaseline(obs_dim: 'int', **kw: 'object') -> 'None'`

``x̂_{t+k} = x_t`` for all ``k``. Zero parameters.

- `forecast(self, x_context: 'Tensor', horizon: 'int') -> 'Tensor'` — ``(B,C,D), H → (B,H,D)``: predictions for the ``H`` steps after the context.
- `predict_direct(self, x_context: 'Tensor') -> 'Tensor'` — ``(B,C,D) → (B,direct_horizon,D)`` (direct mode only).
- `predict_next(self, x: 'Tensor') -> 'Tensor'` — ``(B,T,D) → (B,D)``: prediction of the step following the window.
- `predict_next_sequence(self, x: 'Tensor') -> 'Tensor'` — ``(B,T,D) → (B,T,D)``: entry ``t`` is the prediction of ``x_{t+1}`` from ``x_{≤t}``.

### `nssc.baselines.rnn`

GRU / LSTM autoregressive forecasters.

``predict_next`` runs the RNN over the full context and projects the last hidden
state. ``forecast`` (recursive mode) carries the hidden state across steps
instead of re-reading the growing context, which is mathematically identical.

#### class `GRUForecaster(obs_dim: 'int', hidden: 'int' = 64, n_layers: 'int' = 1, dropout: 'float' = 0.0, **kw: 'object') -> 'None'`

Base forecaster: ``x_{≤t} → x̂_{t+1}`` (and optionally ``x̂_{t+1:t+H}``).


#### class `LSTMForecaster(obs_dim: 'int', hidden: 'int' = 64, n_layers: 'int' = 1, dropout: 'float' = 0.0, **kw: 'object') -> 'None'`

Base forecaster: ``x_{≤t} → x̂_{t+1}`` (and optionally ``x̂_{t+1:t+H}``).


### `nssc.baselines.run`

Baseline run pipeline mirroring :func:`nssc.experiment.run_experiment`.

Run config::

    dataset: {...}                       # as in nssc.experiment
    model:   {baseline: gru, kwargs: {...}, mode: recursive|direct, direct_horizon: 30,
              size: small|medium|large}   # size → kwargs preset from configs/models/baselines/<key>.yaml
    training: {...}                      # BaselineTrainerConfig fields (context defaults to windows.context)
    windows: {context: 20, horizon: 30, stride: 5, batch_size: 64}
    eval:    {context: 20, horizons: [...], max_horizon, batch_size, divergence_threshold, latency}
    seed: 0
    tags: [...]
    output_dir: results/raw/<name>

Registered with model name ``baseline:<key>`` (``baseline:<key>/direct`` in direct mode).
Checkpoint: ``<output_dir>/checkpoint/{model.pt, config.json, metadata.json}``.

#### `baseline_model_name(mcfg: 'dict[str, Any]') -> 'str'`

#### `load_forecaster_checkpoint(path: 'str | Path', map_location: 'str | torch.device' = 'cpu') -> 'tuple[SequenceForecaster, dict[str, Any]]'`

#### `load_preset(key: 'str', size: 'str' = 'medium') -> 'dict[str, Any]'`

kwargs for ``key`` at ``size`` from ``configs/models/baselines/<key>.yaml`` (``sizes:`` map).

#### `resolve_model_cfg(mcfg: 'dict[str, Any]', windows: 'dict[str, Any]') -> 'dict[str, Any]'`

Normalise ``model`` config: fill kwargs from ``size``, default direct_horizon to windows.horizon.

#### `run_baseline_experiment(cfg: 'dict[str, Any] | Config', registry: 'ExperimentRegistry | None' = None, device: 'torch.device | None' = None, log=<built-in function print>, save_ckpt: 'bool' = True) -> 'dict[str, Any]'`

Execute one baseline run. Failures are recorded (status='failed'), never raised.

#### `save_forecaster_checkpoint(model: 'SequenceForecaster', mcfg: 'dict[str, Any]', path: 'str | Path', metadata: 'dict[str, Any] | None' = None) -> 'Path'`

### `nssc.baselines.ssm`

Diagonal linear state-space (S4D-lite) forecaster.

Stacks :class:`nssc.representations.ssm.SSMBlock` (the same building block as
the ``ssm`` encoder) behind an input projection and in front of a linear
next-step head. Exactly causal; the recurrence is a chunked parallel scan.

#### class `SSMForecaster(obs_dim: 'int', d_model: 'int' = 64, d_state: 'int' = 16, n_layers: 'int' = 2, expand: 'int' = 2, dropout: 'float' = 0.0, chunk: 'int' = 32, **kw: 'object') -> 'None'`

Base forecaster: ``x_{≤t} → x̂_{t+1}`` (and optionally ``x̂_{t+1:t+H}``).

- `backbone(self, x: 'Tensor') -> 'Tensor'` — Causal features ``(B,T,F)``: position ``t`` depends on ``x_{≤t}`` only.
- `config(self) -> 'dict'` — Constructor kwargs needed to rebuild (subclasses extend).

### `nssc.baselines.tcn`

Causal dilated temporal-convolution forecaster (WaveNet/TCN style).

Reuses :class:`nssc.representations.tcn.TCNBlock`. Receptive field
``1 + (k-1) * sum(dilations)``; pass ``min_receptive_field`` (e.g. the context
length) to append doubling-dilation layers until the field covers it. Positions
outside the receptive field are exactly ignored, so ``max_context`` is set to
the receptive field (truncation in ``forecast`` is lossless).

#### class `TCNForecaster(obs_dim: 'int', channels: 'int' = 64, kernel_size: 'int' = 3, n_layers: 'int' = 4, dilations: 'Sequence[int] | None' = None, activation: 'str' = 'gelu', dropout: 'float' = 0.0, min_receptive_field: 'int | None' = None, **kw: 'object') -> 'None'`

Base forecaster: ``x_{≤t} → x̂_{t+1}`` (and optionally ``x̂_{t+1:t+H}``).

- `backbone(self, x: 'Tensor') -> 'Tensor'` — Causal features ``(B,T,F)``: position ``t`` depends on ``x_{≤t}`` only.
- `config(self) -> 'dict'` — Constructor kwargs needed to rebuild (subclasses extend).

### `nssc.baselines.trainer`

Trainer for :class:`SequenceForecaster` baselines on ``WindowDataset`` windows.

Loss per window ``x (B, L, D)`` with ``L = context + horizon``:

* teacher-forced next-step MSE over all positions ``t >= context - 1``
  (predict ``x_{t+1}`` from ``x_{≤t}``; ``predict_next_sequence``);
* optional multi-step recursive MSE: ``forecast(x[:, :context], h)`` vs
  ``x[:, context:context+h]`` with a linear horizon curriculum ``1 → rollout_horizon``
  (weight ``rollout_weight``; ``0`` disables);
* direct mode: MSE of ``predict_direct(x[:, :context])`` vs the next ``direct_horizon`` steps
  (weight ``direct_weight``) in addition to the teacher-forced term.

Optimisation mirrors :class:`nssc.training.trainer.Trainer`: AdamW, cosine
schedule with warm-up, gradient clipping, early stopping on validation loss,
best-state restore, per-epoch history.

#### class `BaselineTrainer(model: 'SequenceForecaster', cfg: 'BaselineTrainerConfig', device: 'torch.device | None' = None) -> 'None'`

- `compute_loss(self, x: 'Tensor', horizon: 'int') -> 'tuple[Tensor, dict[str, float]]'` — ``x``: (B, L, D) window. Returns (total, components).
- `evaluate(self, loader: 'DataLoader') -> 'dict[str, float]'` — Validation loss at the *full* rollout horizon (no curriculum) for a stable criterion.
- `fit(self, train_loader: 'DataLoader', val_loader: 'DataLoader | None' = None, log=None) -> 'dict[str, Any]'`
- `maybe_closed_form_fit(self, loader: 'DataLoader', max_batches: 'int' = 50) -> 'None'` — Baselines exposing ``fit(x)`` (e.g. mean) are fitted on a sample of training windows.
- `train_epoch(self, loader: 'DataLoader', epoch: 'int') -> 'dict[str, float]'`

#### class `BaselineTrainerConfig(epochs: 'int' = 50, lr: 'float' = 0.001, weight_decay: 'float' = 1e-05, grad_clip: 'float' = 1.0, scheduler: 'str' = 'cosine', warmup_epochs: 'int' = 0, early_stopping_patience: 'int' = 20, context: 'int' = 20, rollout_horizon: 'int' = 10, rollout_weight: 'float' = 1.0, rollout_curriculum: 'bool' = True, curriculum_epochs: 'int | None' = None, direct_weight: 'float' = 1.0, log_every: 'int' = 10, max_batches_per_epoch: 'int | None' = None, device: 'str | None' = None) -> None`

BaselineTrainerConfig(epochs: 'int' = 50, lr: 'float' = 0.001, weight_decay: 'float' = 1e-05, grad_clip: 'float' = 1.0, scheduler: 'str' = 'cosine', warmup_epochs: 'int' = 0, early_stopping_patience: 'int' = 20, context: 'int' = 20, rollout_horizon: 'int' = 10, rollout_weight: 'float' = 1.0, rollout_curriculum: 'bool' = True, curriculum_epochs: 'int | None' = None, direct_weight: 'float' = 1.0, log_every: 'int' = 10, max_batches_per_epoch: 'int | None' = None, device: 'str | None' = None)


### `nssc.baselines.transformer`

Decoder-only (causal) transformer forecaster.

``nn.TransformerEncoder`` with a causal attention mask, learned positional
embeddings up to ``max_len`` (default 512), pre-LayerNorm blocks and a linear
next-step head applied at every position. Contexts longer than ``max_len`` are
truncated to the most recent ``max_len`` steps.

#### class `TransformerForecaster(obs_dim: 'int', d_model: 'int' = 64, n_heads: 'int' = 4, n_layers: 'int' = 2, dim_feedforward: 'int | None' = None, dropout: 'float' = 0.0, max_len: 'int' = 512, **kw: 'object') -> 'None'`

Base forecaster: ``x_{≤t} → x̂_{t+1}`` (and optionally ``x̂_{t+1:t+H}``).

- `backbone(self, x: 'Tensor') -> 'Tensor'` — Causal features ``(B,T,F)``: position ``t`` depends on ``x_{≤t}`` only.
- `config(self) -> 'dict'` — Constructor kwargs needed to rebuild (subclasses extend).
