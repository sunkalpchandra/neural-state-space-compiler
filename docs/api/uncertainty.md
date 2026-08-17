# `nssc.uncertainty`

### `nssc.uncertainty`

Probabilistic rollouts and calibration evaluation for stochastic latent dynamics.

### `nssc.uncertainty.rollout`

Uncertainty envelopes for long-horizon rollouts.

For ``GaussianDynamics`` (``is_stochastic``) we propagate Monte-Carlo samples
through the latent transition and decode each sample, giving a predictive
distribution over observations at every horizon step. For deterministic
dynamics we fall back to an *ensemble-of-initial-perturbations* envelope
(perturb the encoded initial latent by ``eps``), which is clearly labelled as
such in the returned dict (``method``).

#### `evaluate_uncertainty(model: 'LatentModel', x: 'Tensor', context: 'int', horizon: 'int', n_samples: 'int' = 32, batch_size: 'int' = 32) -> 'dict[str, Any]'`

Calibration of the predictive envelope on held-out ``x`` (N,T,D).

#### `probabilistic_rollout(model: 'LatentModel', x_context: 'Tensor', horizon: 'int', n_samples: 'int' = 32, eps: 'float' = 0.01) -> 'dict[str, Any]'`

Returns dict with ``mean`` (B,H,D), ``std`` (B,H,D), ``samples`` (S,B,H,D), ``method``.
