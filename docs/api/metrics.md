# `nssc.metrics`

### `nssc.metrics`

Centralised metrics. All functions accept torch tensors or numpy arrays.

Shapes: predictions/targets are (B, T, D) unless stated. Returned values are
python floats or dicts of floats so they serialise directly into the registry.

### `nssc.metrics.calibration`

Calibration metrics for probabilistic rollouts (Gaussian predictive distributions).

#### `coverage(mean, std, target, z: 'float' = 1.96) -> 'float'`

Fraction of targets inside mean ± z·std (nominal 95% for z=1.96).

#### `expected_calibration_error_regression(mean, std, target, levels: 'tuple[float, ...]' = (0.5, 0.68, 0.8, 0.9, 0.95, 0.99)) -> 'dict[str, float]'`

Mean |empirical coverage − nominal| across confidence levels + per-level coverage.

#### `gaussian_nll(mean, var, target) -> 'float'`

#### `sharpness(std: 'Tensor | np.ndarray') -> 'float'`

### `nssc.metrics.complexity`

Model complexity and cost metrics: parameters, FLOPs, latency, memory.

#### `count_parameters(module: 'nn.Module', trainable_only: 'bool' = True) -> 'int'`

#### `estimate_flops_per_step(module: 'nn.Module', example: 'torch.Tensor') -> 'int | None'`

Rough multiply-add count for one forward pass via torch.utils.flop_counter.

Returns None if the counter is unavailable or the module uses unsupported ops.

#### `measure_inference_latency(fn: 'Callable[[], object]', n_warmup: 'int' = 5, n_iters: 'int' = 20, device: 'torch.device | None' = None) -> 'dict[str, float]'`

Wall-clock latency (ms) statistics for calling ``fn`` repeatedly.

#### `peak_memory_mb(device: 'torch.device | None' = None) -> 'float | None'`

### `nssc.metrics.prediction`

Reconstruction / prediction / rollout error metrics.

#### `horizon_curve(pred, target, sigma=None) -> 'np.ndarray'`

Per-step NRMSE along the horizon axis. pred/target: (B, H, D) → (H,).

#### `mse(pred, target) -> 'float'`

#### `nrmse(pred, target, sigma: 'float | np.ndarray | None' = None) -> 'float'`

RMSE divided by the (per-dim, then averaged) std of the target.

``sigma`` may be provided from the *training* set to avoid test-set leakage.

#### `r2_score(pred, target) -> 'float'`

#### `rmse(pred, target) -> 'float'`

#### `rollout_divergence_time(pred, target, threshold: 'float' = 1.0, sigma=None) -> 'float'`

First horizon step at which per-step NRMSE exceeds ``threshold`` (H+1 if never).

#### `rollout_errors(pred, target, horizons: 'Sequence[int]' = (1, 5, 10, 25, 50, 100, 250, 500), sigma=None) -> 'dict[str, float]'`

NRMSE accumulated up to each horizon k (mean over steps 1..k) plus per-step at k.

Keys: ``nrmse@k`` (cumulative), ``nrmse_step@k`` (instantaneous). Horizons
beyond the available length are skipped.
