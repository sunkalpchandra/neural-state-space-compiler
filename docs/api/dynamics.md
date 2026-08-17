# `nssc.dynamics`

### `nssc.dynamics`

Latent transition operators z_{t+1} = F_θ(z_t, u_t).

Import this package to populate the DYNAMICS registry.

#### `build_dynamics(key: 'str', latent_dim: 'int', **kwargs: 'Any') -> 'Dynamics'`

Instantiate a registered dynamics family by key.

### `nssc.dynamics.base`

Base class for latent dynamics.

All dynamics expose a discrete-time ``step`` (one transition), a batched
``rollout`` (recursive application), and a ``jacobian`` for stability analysis.
Continuous-time models (neural ODE) still expose ``step`` with a fixed ``dt``.

#### class `Dynamics(latent_dim: 'int', control_dim: 'int' = 0) -> 'None'`

Base class for all neural network modules.

Your models should also subclass this class.

Modules can also contain other Modules, allowing them to be nested in
a tree structure. You can assign the submodules as regular attributes::

    import torch.nn as nn
    import torch.nn.functional as F


    class Model(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv1 = nn.Conv2d(1, 20, 5)
            self.conv2 = nn.Conv2d(20, 20, 5)

        def forward(self, x):
            x = F.relu(self.conv1(x))
            return F.relu(self.conv2(x))

Submodules assigned in this way will be registered, and will also have their
parameters converted when you call :meth:`to`, etc.

.. note::
    As per the example above, an ``__init__()`` call to the parent class
    must be made before assignment on the child.

:ivar training: Boolean represents whether this module is in training or
                evaluation mode.
:vartype training: bool

- `extra_losses(self) -> 'dict[str, Tensor]'` — Optional model-specific regularisers (e.g. Koopman consistency).
- `forward(self, z: 'Tensor', u: 'Tensor | None' = None) -> 'Tensor'` — Define the computation performed at every call.
- `jacobian(self, z: 'Tensor', u: 'Tensor | None' = None) -> 'Tensor'` — Batched Jacobian ∂F/∂z at ``z``: (B, d) → (B, d, d). Autograd default.
- `num_parameters(self) -> 'int'`
- `rollout(self, z0: 'Tensor', horizon: 'int', u: 'Tensor | None' = None) -> 'Tensor'` — Recursive rollout. ``z0``: (B, d) → (B, horizon, d) of z_1..z_H.
- `step(self, z: 'Tensor', u: 'Tensor | None' = None) -> 'Tensor'` — One transition. ``z``: (B, d) → (B, d).
- `step_sequence(self, z: 'Tensor', u: 'Tensor | None' = None) -> 'Tensor'` — Teacher-forced one-step predictions for a whole sequence.

### `nssc.dynamics.gaussian`

Gaussian (heteroscedastic) wrapper around any deterministic dynamics.

    z_{t+1} ~ N( F_base(z_t, u_t), diag(exp(logvar_θ(z_t))) )

``step`` returns the mean so the wrapped model behaves like a deterministic
dynamics for the rest of the pipeline; ``sample_step`` / ``rollout_samples`` /
``nll`` expose the stochastic interface used by the uncertainty module.

#### class `GaussianDynamics(latent_dim: 'int', control_dim: 'int' = 0, base: 'str' = 'residual_mlp', base_kwargs: 'dict[str, Any] | None' = None, var_hidden_dims: 'Sequence[int]' = (64,), act: 'str' = 'gelu', min_logvar: 'float' = -10.0, max_logvar: 'float' = 4.0, init_logvar: 'float' = -2.0, **kwargs) -> 'None'`

Base class for all neural network modules.

Your models should also subclass this class.

Modules can also contain other Modules, allowing them to be nested in
a tree structure. You can assign the submodules as regular attributes::

    import torch.nn as nn
    import torch.nn.functional as F


    class Model(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv1 = nn.Conv2d(1, 20, 5)
            self.conv2 = nn.Conv2d(20, 20, 5)

        def forward(self, x):
            x = F.relu(self.conv1(x))
            return F.relu(self.conv2(x))

Submodules assigned in this way will be registered, and will also have their
parameters converted when you call :meth:`to`, etc.

.. note::
    As per the example above, an ``__init__()`` call to the parent class
    must be made before assignment on the child.

:ivar training: Boolean represents whether this module is in training or
                evaluation mode.
:vartype training: bool

- `extra_losses(self) -> 'dict[str, Tensor]'` — Optional model-specific regularisers (e.g. Koopman consistency).
- `jacobian(self, z: 'Tensor', u: 'Tensor | None' = None) -> 'Tensor'` — Batched Jacobian ∂F/∂z at ``z``: (B, d) → (B, d, d). Autograd default.
- `logvar(self, z: 'Tensor', u: 'Tensor | None' = None) -> 'Tensor'` — Diagonal log-variance of the transition: (B, d) → (B, d).
- `moments(self, z: 'Tensor', u: 'Tensor | None' = None) -> 'tuple[Tensor, Tensor]'` — (mean, logvar) of z_{t+1} | z_t.
- `nll(self, z: 'Tensor', z_next: 'Tensor', u: 'Tensor | None' = None) -> 'Tensor'` — Mean Gaussian negative log-likelihood of ``z_next`` given ``z``. Accepts (B,d) or (B,T,d).
- `rollout_moments(self, z0: 'Tensor', horizon: 'int', n_samples: 'int' = 10, u: 'Tensor | None' = None) -> 'tuple[Tensor, Tensor]'` — Empirical (mean, std) over ``rollout_samples``: each (B, H, d).
- `rollout_samples(self, z0: 'Tensor', horizon: 'int', n_samples: 'int' = 10, u: 'Tensor | None' = None) -> 'Tensor'` — Monte-Carlo rollouts: (B, d) → (S, B, H, d).
- `sample_step(self, z: 'Tensor', u: 'Tensor | None' = None) -> 'Tensor'`
- `step(self, z: 'Tensor', u: 'Tensor | None' = None) -> 'Tensor'` — One transition. ``z``: (B, d) → (B, d).

### `nssc.dynamics.koopman`

Koopman-style dynamics: lift, apply a linear operator, read back.

    φ(z) = [z, g_θ(z)] ∈ R^m,   z_{t+1} = C K φ(z_t)

with learnable lifting ``g_θ`` (MLP), operator ``K`` (m×m) and readout ``C``.
Because ``z`` is included in ``φ`` the readout can be the exact projection onto
the first ``d`` coordinates (``exact_readout=True``, default) or a learned
linear map. The consistency loss ``||φ(z_next) - K φ(z)||^2`` regularises ``K``
towards a genuine Koopman operator on the lifted space.

#### class `KoopmanDynamics(latent_dim: 'int', control_dim: 'int' = 0, obs_dim_lift: 'int | None' = None, hidden_dims: 'Sequence[int]' = (64, 64), act: 'str' = 'gelu', residual: 'bool' = False, exact_readout: 'bool' = True, consistency_weight: 'float' = 1.0, init_scale: 'float' = 0.01, **kwargs) -> 'None'`

Base class for all neural network modules.

Your models should also subclass this class.

Modules can also contain other Modules, allowing them to be nested in
a tree structure. You can assign the submodules as regular attributes::

    import torch.nn as nn
    import torch.nn.functional as F


    class Model(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv1 = nn.Conv2d(1, 20, 5)
            self.conv2 = nn.Conv2d(20, 20, 5)

        def forward(self, x):
            x = F.relu(self.conv1(x))
            return F.relu(self.conv2(x))

Submodules assigned in this way will be registered, and will also have their
parameters converted when you call :meth:`to`, etc.

.. note::
    As per the example above, an ``__init__()`` call to the parent class
    must be made before assignment on the child.

:ivar training: Boolean represents whether this module is in training or
                evaluation mode.
:vartype training: bool

- `consistency_loss(self, z: 'Tensor', z_next: 'Tensor', u: 'Tensor | None' = None) -> 'Tensor'` — ||φ(z_next) - K φ(z)||^2 averaged over the batch. Accepts (B,d) or (B,T,d).
- `eigenvalues(self) -> 'Tensor'`
- `extra_losses(self) -> 'dict[str, Tensor]'` — Optional model-specific regularisers (e.g. Koopman consistency).
- `lift_fn(self, z: 'Tensor') -> 'Tensor'` — φ(z) = [z, g_θ(z)]: (B, d) → (B, m).
- `lifted_step(self, phi: 'Tensor', u: 'Tensor | None' = None) -> 'Tensor'`
- `readout(self, phi: 'Tensor') -> 'Tensor'`
- `rollout(self, z0: 'Tensor', horizon: 'int', u: 'Tensor | None' = None) -> 'Tensor'` — Recursive rollout. ``z0``: (B, d) → (B, horizon, d) of z_1..z_H.
- `rollout_lifted(self, z0: 'Tensor', horizon: 'int') -> 'Tensor'` — Purely linear rollout in lifted space (no re-lifting): (B, H, d).
- `step(self, z: 'Tensor', u: 'Tensor | None' = None) -> 'Tensor'` — One transition. ``z``: (B, d) → (B, d).

### `nssc.dynamics.linear`

Linear and affine latent dynamics: z' = A z (+ b).

These are the simplest, most interpretable families and the natural baseline for
the compiler (PCA + linear). ``least_squares_fit`` gives a DMD-style closed-form
initialisation from paired latents.

#### class `AffineDynamics(latent_dim: 'int', control_dim: 'int' = 0, **kwargs) -> 'None'`

z_{t+1} = A z_t + b (+ B u_t).

- `fixed_point(self) -> 'Tensor'` — z* = (I - A)^{-1} b (via lstsq for robustness).
- `least_squares_fit(self, z_t: 'Tensor', z_next: 'Tensor', ridge: 'float' = 1e-06) -> 'None'` — Closed-form fit of ``[A | b]`` by augmenting z with a constant 1.
- `step(self, z: 'Tensor', u: 'Tensor | None' = None) -> 'Tensor'` — One transition. ``z``: (B, d) → (B, d).

#### class `LinearDynamics(latent_dim: 'int', control_dim: 'int' = 0, spectral_norm_max: 'float | None' = None, init_scale: 'float' = 0.01, **kwargs) -> 'None'`

z_{t+1} = A z_t (+ B u_t).

``A`` is initialised near the identity (``I + 0.01 * randn``) so untrained
rollouts are stable. ``spectral_norm_max`` adds a soft hinge penalty
``relu(||A||_2 - spectral_norm_max)^2`` via :meth:`extra_losses`.

- `eigenvalues(self) -> 'Tensor'` — Complex eigenvalues of ``A``: (d,).
- `extra_losses(self) -> 'dict[str, Tensor]'` — Optional model-specific regularisers (e.g. Koopman consistency).
- `jacobian(self, z: 'Tensor', u: 'Tensor | None' = None) -> 'Tensor'` — Batched Jacobian ∂F/∂z at ``z``: (B, d) → (B, d, d). Autograd default.
- `least_squares_fit(self, z_t: 'Tensor', z_next: 'Tensor', ridge: 'float' = 1e-06) -> 'None'` — Closed-form (DMD-like) fit of ``A`` from paired latents.
- `spectral_radius(self) -> 'float'`
- `step(self, z: 'Tensor', u: 'Tensor | None' = None) -> 'Tensor'` — One transition. ``z``: (B, d) → (B, d).

### `nssc.dynamics.mlp`

MLP and residual-MLP latent dynamics.

#### class `MLPDynamics(latent_dim: 'int', control_dim: 'int' = 0, hidden_dims: 'Sequence[int]' = (128, 128), act: 'str' = 'gelu', **kwargs) -> 'None'`

z_{t+1} = f_θ([z_t, u_t]) with a plain MLP.

- `step(self, z: 'Tensor', u: 'Tensor | None' = None) -> 'Tensor'` — One transition. ``z``: (B, d) → (B, d).

#### class `ResidualMLPDynamics(latent_dim: 'int', control_dim: 'int' = 0, hidden_dims: 'Sequence[int]' = (128, 128), act: 'str' = 'gelu', dt: 'float' = 1.0, stability_reg: 'float' = 0.0, **kwargs) -> 'None'`

z_{t+1} = z_t + dt · f_θ([z_t, u_t]).

The last layer is zero-initialised so the model starts as the identity.
``stability_reg`` > 0 adds ``stability_reg * mean ||f(z)||^2`` (computed on
the most recent batch passed through ``step``) to :meth:`extra_losses`,
discouraging unbounded growth of the update field.

- `extra_losses(self) -> 'dict[str, Tensor]'` — Optional model-specific regularisers (e.g. Koopman consistency).
- `step(self, z: 'Tensor', u: 'Tensor | None' = None) -> 'Tensor'` — One transition. ``z``: (B, d) → (B, d).
- `update(self, z: 'Tensor', u: 'Tensor | None' = None) -> 'Tensor'` — The residual field f_θ(z, u): (B, d) → (B, d).

### `nssc.dynamics.multiscale`

Slow/fast multi-scale latent dynamics.

The latent is split z = [z_slow, z_fast] (first ``slow_dim`` coordinates slow).

    z_slow' = z_slow + Δ_s f_s(z_slow)              (small Δ_s, or only every ``slow_every`` steps)
    z_fast' = f_f(z_fast, z_slow, u)                (residual MLP)

``mode='rate'``: slow block updates every step with rate Δ_s.
``mode='strided'``: slow block updates (with rate Δ_s·slow_every, so the mean
speed matches) only when ``t % slow_every == 0``; ``step`` accepts an optional
``t`` index and ``rollout`` threads it so the module stays stateless.

#### class `MultiScaleDynamics(latent_dim: 'int', control_dim: 'int' = 0, slow_dim: 'int | None' = None, slow_rate: 'float' = 0.1, slow_every: 'int' = 1, mode: 'str' = 'rate', hidden_dims: 'Sequence[int]' = (64, 64), act: 'str' = 'gelu', fast_dt: 'float' = 1.0, **kwargs) -> 'None'`

Base class for all neural network modules.

Your models should also subclass this class.

Modules can also contain other Modules, allowing them to be nested in
a tree structure. You can assign the submodules as regular attributes::

    import torch.nn as nn
    import torch.nn.functional as F


    class Model(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv1 = nn.Conv2d(1, 20, 5)
            self.conv2 = nn.Conv2d(20, 20, 5)

        def forward(self, x):
            x = F.relu(self.conv1(x))
            return F.relu(self.conv2(x))

Submodules assigned in this way will be registered, and will also have their
parameters converted when you call :meth:`to`, etc.

.. note::
    As per the example above, an ``__init__()`` call to the parent class
    must be made before assignment on the child.

:ivar training: Boolean represents whether this module is in training or
                evaluation mode.
:vartype training: bool

- `rollout(self, z0: 'Tensor', horizon: 'int', u: 'Tensor | None' = None, t0: 'int' = 0) -> 'Tensor'` — Recursive rollout. ``z0``: (B, d) → (B, horizon, d) of z_1..z_H.
- `split(self, z: 'Tensor') -> 'tuple[Tensor, Tensor]'`
- `step(self, z: 'Tensor', u: 'Tensor | None' = None, t: 'int | None' = None) -> 'Tensor'` — One transition. ``z``: (B, d) → (B, d).

### `nssc.dynamics.neural_ode`

Neural ODE latent dynamics with an in-house fixed-step RK4 integrator.

    dz/dt = f_θ(z, u),   z_{t+1} = z_t + ∫_0^dt f_θ dt   (RK4, n_substeps)

Backprop goes straight through the solver (no adjoint), which is exact and
cheap for the small latent dimensions this project targets.

#### class `NeuralODEDynamics(latent_dim: 'int', control_dim: 'int' = 0, hidden_dims: 'Sequence[int]' = (64, 64), act: 'str' = 'tanh', dt: 'float' = 1.0, n_substeps: 'int' = 4, solver: 'str' = 'rk4', **kwargs) -> 'None'`

Base class for all neural network modules.

Your models should also subclass this class.

Modules can also contain other Modules, allowing them to be nested in
a tree structure. You can assign the submodules as regular attributes::

    import torch.nn as nn
    import torch.nn.functional as F


    class Model(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv1 = nn.Conv2d(1, 20, 5)
            self.conv2 = nn.Conv2d(20, 20, 5)

        def forward(self, x):
            x = F.relu(self.conv1(x))
            return F.relu(self.conv2(x))

Submodules assigned in this way will be registered, and will also have their
parameters converted when you call :meth:`to`, etc.

.. note::
    As per the example above, an ``__init__()`` call to the parent class
    must be made before assignment on the child.

:ivar training: Boolean represents whether this module is in training or
                evaluation mode.
:vartype training: bool

- `integrate(self, z: 'Tensor', t_span: 'float', u: 'Tensor | None' = None, n_substeps: 'int | None' = None) -> 'Tensor'` — Integrate from z over ``t_span`` latent time units (u held constant).
- `step(self, z: 'Tensor', u: 'Tensor | None' = None) -> 'Tensor'` — One transition. ``z``: (B, d) → (B, d).
- `trajectory(self, z0: 'Tensor', t_end: 'float', n_points: 'int', u: 'Tensor | None' = None) -> 'Tensor'` — Dense trajectory sampled at ``n_points`` uniform times in (0, t_end]: (B, n_points, d).
- `vector_field(self, z: 'Tensor', u: 'Tensor | None' = None) -> 'Tensor'` — f_θ(z, u): (B, d) → (B, d). Useful for phase-portrait plots.

#### `euler_step(f: 'Callable[[Tensor], Tensor]', z: 'Tensor', h: 'float') -> 'Tensor'`

#### `rk4_step(f: 'Callable[[Tensor], Tensor]', z: 'Tensor', h: 'float') -> 'Tensor'`

### `nssc.dynamics.ssm`

Diagonal (+ low-rank) linear SSM in the latent, driven by a nonlinear feature of z.

    z_{t+1} = diag(a) z_t + (U V^T) z_t + B σ(W z_t + c) + G u_t

``a`` is constrained to |a| < 1 either through ``tanh`` (allows negative /
oscillatory-sign modes) or ``exp(-softplus)`` (positive decays only). With
``rank=0`` and ``feature_dim=0`` this collapses to a stable diagonal linear map.

#### class `SSMDynamics(latent_dim: 'int', control_dim: 'int' = 0, feature_dim: 'int | None' = None, rank: 'int' = 0, act: 'str' = 'tanh', param: 'str' = 'tanh', init_decay: 'float' = 0.95, init_scale: 'float' = 0.01, **kwargs) -> 'None'`

Base class for all neural network modules.

Your models should also subclass this class.

Modules can also contain other Modules, allowing them to be nested in
a tree structure. You can assign the submodules as regular attributes::

    import torch.nn as nn
    import torch.nn.functional as F


    class Model(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv1 = nn.Conv2d(1, 20, 5)
            self.conv2 = nn.Conv2d(20, 20, 5)

        def forward(self, x):
            x = F.relu(self.conv1(x))
            return F.relu(self.conv2(x))

Submodules assigned in this way will be registered, and will also have their
parameters converted when you call :meth:`to`, etc.

.. note::
    As per the example above, an ``__init__()`` call to the parent class
    must be made before assignment on the child.

:ivar training: Boolean represents whether this module is in training or
                evaluation mode.
:vartype training: bool

- `eigenvalues(self) -> 'Tensor'` — Eigenvalues of the linear part diag(a) + U V^T: (d,).
- `linear_operator(self) -> 'Tensor'` — Dense linear part diag(a) + U V^T: (d, d).
- `step(self, z: 'Tensor', u: 'Tensor | None' = None) -> 'Tensor'` — One transition. ``z``: (B, d) → (B, d).
