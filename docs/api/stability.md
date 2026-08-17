# `nssc.stability`

### `nssc.stability`

Stability diagnostics for latent dynamics: spectra, Lyapunov, norm growth, rollout blow-up.

### `nssc.stability.analysis`

Aggregate stability report for a LatentModel on held-out data.

#### class `StabilityReport(spectral: 'dict[str, float]' = <factory>, lyapunov: 'dict[str, float]' = <factory>, norm_growth: 'dict[str, float]' = <factory>, rollout: 'dict[str, float]' = <factory>, instability_score: 'float' = 0.0, verdict: 'str' = 'unknown') -> None`

StabilityReport(spectral: 'dict[str, float]' = <factory>, lyapunov: 'dict[str, float]' = <factory>, norm_growth: 'dict[str, float]' = <factory>, rollout: 'dict[str, float]' = <factory>, instability_score: 'float' = 0.0, verdict: 'str' = 'unknown')

- `to_dict(self) -> 'dict[str, Any]'`

#### `analyze_stability(model: 'LatentModel', x: 'Tensor', horizon: 'int' = 200, dt: 'float' = 1.0, lyapunov_steps: 'int' = 200, max_points: 'int' = 128) -> 'StabilityReport'`

``x``: (B, T, D) held-out observations (already normalised like training data).

#### `latent_norm_growth(model: 'LatentModel', x: 'Tensor', horizon: 'int') -> 'dict[str, float]'`

Compare ||ẑ_t|| along a free rollout to the norm distribution of encoded data.

### `nssc.stability.lyapunov`

Largest Lyapunov exponent of a discrete latent map via tangent-vector renormalisation.

    λ_max ≈ (1/(K·Δt)) Σ_k log ||J_k v_k|| / ||v_k||

Positive λ indicates local exponential divergence (chaos or instability); for a
learned model of Lorenz-63 in latent time units one expects λ·(1/dt) ≈ 0.9 if the
latent dynamics faithfully reproduce the attractor.

#### `largest_lyapunov_exponent(dynamics: 'Dynamics', z0: 'Tensor', n_steps: 'int' = 500, dt: 'float' = 1.0, n_transient: 'int' = 50, seed: 'int' = 0) -> 'dict[str, float]'`

``z0``: (B, d) initial latents. Returns mean/std over batch of λ_max (per unit ``dt``).

### `nssc.stability.spectral`

Local linearisation spectra of latent dynamics.

#### `jacobian_spectrum(dynamics: 'Dynamics', z: 'Tensor', max_points: 'int' = 256) -> 'dict[str, Tensor]'`

Eigenvalues of ∂F/∂z at up to ``max_points`` latent states.

``z``: (N, d) or (B, T, d). Returns dict with ``eigvals`` (M, d) complex,
``spectral_radius`` (M,), ``points`` (M, d).

#### `spectral_radius_stats(dynamics: 'Dynamics', z: 'Tensor', max_points: 'int' = 256) -> 'dict[str, float]'`
