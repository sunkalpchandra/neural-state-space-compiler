# `nssc.data`

### `nssc.data`

Data subsystem: synthetic systems, observation maps, splits, datasets, builder.

### `nssc.data.builder`

Build a :class:`TrajectoryDataset` from a config dict / YAML.

Config schema (all keys optional except ``system``)::

    system: lorenz63
    params: {rho: 28.0}          # overrides system defaults
    n_traj: 100
    n_steps: 500
    dt: 0.01                     # default: system.default_dt
    substeps: 1
    seed: 0
    transient: 1000              # default: system.default_transient
    observation: {type: identity | linear | mlp | polynomial | redundant | pipeline, ...}
    noise_std: 0.0
    missing_rate: 0.0
    kuramoto_sin_cos: true       # Kuramoto only: observe (cos, sin) instead of raw phases
    split: {fractions: [0.7, 0.15, 0.15], seed: 0}   # used by TrajectoryDataset.split()
    ood: {...}                   # documentation of OOD parameter ranges (hashed, not applied)

Real-world sources bypass the simulator entirely: a config with ``source: eegbci`` (or
any key in :data:`nssc.data.real.REAL_SOURCES`) is dispatched to
:func:`nssc.data.real.build_real_dataset`; see :mod:`nssc.data.real.eegbci`.

The dataset ``metadata['version']`` is ``stable_hash`` of the resolved config so
any change in split/preprocessing shows up as a new dataset version.

#### `build_dataset(cfg: 'dict[str, Any] | str | Path') -> 'TrajectoryDataset'`

Simulate, observe, corrupt and package a dataset. See module docstring.

#### `resolve_config(cfg: 'dict[str, Any]') -> 'dict[str, Any]'`

Fill defaults (including system ``dt``/``transient``) so the hash is stable.

### `nssc.data.dataset`

Trajectory dataset container and torch sliding-window dataset.

Arrays are ``(N, T, D)`` for observations (float32), ``(N, T, d)`` for optional
ground-truth latents (never fed to models), ``(T,)`` for time stamps and an
optional boolean ``mask`` ``(N, T, D)`` (``True`` = observed).

#### class `TrajectoryDataset(x: 'np.ndarray', t: 'np.ndarray', z_true: 'np.ndarray | None' = None, mask: 'np.ndarray | None' = None, metadata: 'dict[str, Any]' = <factory>) -> None`

TrajectoryDataset(x: 'np.ndarray', t: 'np.ndarray', z_true: 'np.ndarray | None' = None, mask: 'np.ndarray | None' = None, metadata: 'dict[str, Any]' = <factory>)

- `compute_stats(self) -> 'dict[str, np.ndarray]'` — Per-dim ``mean``/``std`` ``(D,)`` over all trajectories and steps (NaN-aware).
- `normalize(self, stats: 'dict[str, np.ndarray] | None' = None) -> 'tuple[TrajectoryDataset, dict[str, np.ndarray]]'` — Return ``((x - mean) / std`` copy, stats). Pass train stats for val/test.
- `save(self, path: 'str | Path') -> 'Path'`
- `split(self, seed: 'int | None' = None, fractions: 'tuple[float, float, float] | None' = None) -> 'dict[str, TrajectoryDataset]'` — Trajectory-level train/val/test split (see :func:`trajectory_split`).
- `subset(self, idx: 'np.ndarray', split_name: 'str | None' = None) -> 'TrajectoryDataset'` — Select trajectories ``idx`` (a copy). ``metadata['split']`` records the name.
- `to_torch(self, device: 'str | torch.device' = 'cpu') -> 'dict[str, torch.Tensor]'` — Tensors ``x (N,T,D)``, ``t (T,)``, optional ``z_true``, ``mask``.

#### class `WindowDataset(ds: 'TrajectoryDataset', context: 'int', horizon: 'int', stride: 'int' = 1) -> 'None'`

Sliding windows of length ``context + horizon`` over every trajectory.

Item: ``{'x': (L, D), 'traj': int, 'start': int}`` plus ``'z_true': (L, d)`` and
``'mask': (L, D)`` when present in the source dataset.


#### `make_loaders(ds_splits: 'dict[str, TrajectoryDataset]', context: 'int', horizon: 'int', batch_size: 'int' = 64, stride: 'int' = 1, num_workers: 'int' = 0, shuffle_train: 'bool' = True) -> 'dict[str, DataLoader]'`

Build a ``DataLoader`` per split; only ``train`` is shuffled.

#### `n_windows(n_traj: 'int', n_steps: 'int', length: 'int', stride: 'int') -> 'int'`

Number of sliding windows: ``N * (floor((T - L) / stride) + 1)`` (0 if ``T < L``).

### `nssc.data.integrators`

Vectorised fixed-step ODE integrators (pure numpy).

All integrators take a vector field ``f(t, x) -> dx`` operating on a batch of
states ``x`` of shape ``(N, d)`` and return the sampled trajectory of shape
``(N, T, d)`` where ``T = n_steps`` (the initial condition is included as the
first sample). Integration happens in float64; callers cast as needed.

``substeps`` decouples the sampling interval ``dt`` from the integrator step
``dt / substeps`` for stiff or fast systems (e.g. Lorenz-63 at coarse ``dt``).

#### `euler(f: 'VectorField', x0: 'np.ndarray', dt: 'float', n_steps: 'int', substeps: 'int' = 1, t0: 'float' = 0.0) -> 'np.ndarray'`

Forward-Euler convenience wrapper: see :func:`integrate`. Returns ``(N, T, d)``.

#### `euler_step(f: 'VectorField', t: 'float', x: 'np.ndarray', h: 'float') -> 'np.ndarray'`

One forward-Euler step of size ``h``. ``x``: ``(N, d)``.

#### `integrate(f: 'VectorField', x0: 'np.ndarray', dt: 'float', n_steps: 'int', substeps: 'int' = 1, method: 'str' = 'rk4', t0: 'float' = 0.0) -> 'np.ndarray'`

Integrate ``dx/dt = f(t, x)`` from a batch of initial conditions.

Parameters
----------
f : callable ``(t, x) -> dx`` with ``x`` and ``dx`` of shape ``(N, d)``.
x0 : ``(N, d)`` initial states (any float dtype; cast to float64).
dt : sampling interval between returned samples.
n_steps : number of returned samples ``T`` (first sample is ``x0``).
substeps : integrator steps per sampling interval (``h = dt / substeps``).
method : ``"rk4"`` or ``"euler"``.

Returns
-------
``(N, T, d)`` float64 array of sampled states.

#### `rk4(f: 'VectorField', x0: 'np.ndarray', dt: 'float', n_steps: 'int', substeps: 'int' = 1, t0: 'float' = 0.0) -> 'np.ndarray'`

RK4 convenience wrapper: see :func:`integrate`. Returns ``(N, T, d)``.

#### `rk4_step(f: 'VectorField', t: 'float', x: 'np.ndarray', h: 'float') -> 'np.ndarray'`

One classical Runge-Kutta 4 step of size ``h``. ``x``: ``(N, d)``.

### `nssc.data.observation`

Tier-2 observation maps: latent ``(N, T, d)`` -> observed ``(N, T, D)``.

Every map is deterministic given its ``seed`` and round-trips through
``to_config()`` / ``from_config()``. Maps operate on the trailing dimension so
they accept any leading shape ``(..., d)``. Noise / masking helpers are plain
functions taking an explicit ``numpy.random.Generator``.

#### class `IdentityObservation()`

Base class. Subclasses implement ``__call__``/``to_config``; register in ``OBS_MAPS``.


#### class `LinearObservation(obs_dim: 'int', seed: 'int' = 0, orthogonal: 'bool' = False, in_dim: 'int | None' = None) -> 'None'`

``x = z @ W`` with ``W ~ N(0, 1/d)`` of shape ``(d, D)`` (optionally orthonormal cols).

``d`` is inferred lazily from the first input if not given; the matrix is a
deterministic function of ``(seed, d, D)``.

- `matrix(self, d: 'int') -> 'np.ndarray'`
- `to_config(self) -> 'dict[str, Any]'`

#### class `ObservationMap()`

Base class. Subclasses implement ``__call__``/``to_config``; register in ``OBS_MAPS``.

- `to_config(self) -> 'dict[str, Any]'`

#### class `ObservationPipeline(maps: 'list[ObservationMap | dict[str, Any]] | None' = None) -> 'None'`

Sequential composition of maps; ``to_config`` nests member configs.

- `to_config(self) -> 'dict[str, Any]'`

#### class `PolynomialObservation(degree: 'int' = 2, obs_dim: 'int | None' = None, seed: 'int' = 0) -> 'None'`

Monomials up to ``degree`` (no constant): ``[z, z_i z_j (i<=j), ...]``, then optional
random linear projection to ``obs_dim`` (``None`` keeps all monomials).

- `to_config(self) -> 'dict[str, Any]'`

#### class `RandomMLPObservation(obs_dim: 'int', hidden: 'int | list[int]' = 64, seed: 'int' = 0, n_layers: 'int' = 1, in_dim: 'int | None' = None, gain: 'float' = 1.0) -> 'None'`

Fixed random tanh MLP ``d -> hidden -> ... -> D`` (numpy, no training).

- `to_config(self) -> 'dict[str, Any]'`
- `weights(self, d: 'int') -> 'list[tuple[np.ndarray, np.ndarray]]'`

#### class `RedundantObservation(repeats: 'int' = 4, alpha: 'float' = 0.1, seed: 'int' = 0) -> 'None'`

Tile ``z`` ``repeats`` times then mix with a random near-identity linear map:
``x = tile(z) @ (I + alpha * G)`` -> ``D = repeats * d``.

- `to_config(self) -> 'dict[str, Any]'`

#### `add_noise(x: 'np.ndarray', sigma: 'float', rng: 'np.random.Generator') -> 'np.ndarray'`

Additive iid Gaussian noise with std ``sigma`` (``sigma <= 0`` returns a copy).

#### `irregular_subsample(t: 'np.ndarray', x: 'np.ndarray', keep_rate: 'float', rng: 'np.random.Generator') -> 'tuple[np.ndarray, np.ndarray]'`

Keep a random subset of time indices (shared across the batch), always keeping ``t[0]``.

``t``: ``(T,)``, ``x``: ``(..., T, D)`` -> ``(t_kept, x_kept)`` with ``T' <= T``.

#### `mask_missing(x: 'np.ndarray', rate: 'float', rng: 'np.random.Generator') -> 'tuple[np.ndarray, np.ndarray]'`

Drop entries iid with prob ``rate``. Returns ``(x_with_nan, mask)`` where
``mask`` is ``True`` for observed entries (same shape as ``x``).

### `nssc.data.real`

Real-world data sources (Tier 3). Heavy deps (mne) are imported lazily inside builders.

Dispatch by ``cfg['source']`` via :func:`build_real_dataset`; sources register in
:data:`REAL_SOURCES` as ``name -> callable(cfg) -> TrajectoryDataset``.

#### `build_real_dataset(cfg: 'dict[str, Any]') -> 'TrajectoryDataset'`

#### `is_real_source(cfg: 'dict[str, Any]') -> 'bool'`

True when the dataset config selects a real-world source (``source`` key present).

### `nssc.data.real.eegbci`

PhysioNet EEG Motor Movement/Imagery (EEGBCI) loader → :class:`TrajectoryDataset`.

Source: Schalk et al. (2004) BCI2000 / Goldberger et al. (2000) PhysioNet, 109 subjects,
64-channel EEG at 160 Hz, 14 runs per subject (see ``docs/datasets_real.md``).
Downloaded on demand with :func:`mne.datasets.eegbci.load_data`; ``mne`` is an optional
extra (``pip install nssc[eeg]``) and is imported lazily inside :func:`build_eegbci`.

Config schema (defaults in :data:`DEFAULTS`)::

    source: eegbci
    subjects: [1, 2, 3, 4]         # PhysioNet subject ids (1..109)
    runs: [3, 7, 11]               # run ids; 3,7,11 = real L/R fist, 6,10,14 = imagined hands/feet
    channels: null | int | [names] # null = all 64; int = first n (file order); list = names
    resample_hz: 64                # downsample from 160 Hz (null = keep 160)
    bandpass: [1.0, 30.0]          # FIR band-pass (mne); null = none
    segment_seconds: 8             # each run is cut into fixed-length segments (= trajectories)
    segment_stride_seconds: 8      # stride between segment starts (== length → no overlap)
    per_subject_standardize: true  # z-score per subject+channel with THAT subject's own data
    split: {by: subject, train_subjects: [1, 2], val_subjects: [3], test_subjects: [4]}
    cache_dir: data/cache/eegbci

Output ``x`` is ``(N_segments, T, C)`` float32, ``t`` in seconds, ``z_true=None`` (there is
no ground-truth latent). ``metadata['split_indices']`` holds the subject-level split which
:meth:`TrajectoryDataset.split` honours; per-segment ``subject_of_segment`` /
``run_of_segment`` are listed in ``metadata['per_traj_keys']`` so subsets stay aligned.
Processed datasets are cached as ``<cache_dir>/eegbci_<version>.npz`` keyed by the
``stable_hash`` of the resolved config.

Standardisation note: per-subject z-scoring uses only that subject's own recordings, so
no statistic from a test subject ever touches training data (and vice versa). The
downstream trainer additionally normalises with *train-split* statistics.

#### `build_eegbci(cfg: 'dict[str, Any]', use_cache: 'bool' = True) -> 'TrajectoryDataset'`

Download (if needed), preprocess, segment and package EEGBCI as a TrajectoryDataset.

Returns ``x (N_segments, T, C)`` float32 with subject-level ``metadata['split_indices']``.
Raises ``RuntimeError`` with instructions when ``mne`` or the network is unavailable.

#### `cache_path(cfg: 'dict[str, Any]') -> 'Path'`

``<cache_dir>/eegbci_<version>.npz`` for a *resolved* config.

#### `resolve_eegbci_config(cfg: 'dict[str, Any]') -> 'dict[str, Any]'`

Fill defaults and validate; the result is what gets hashed into ``metadata['version']``.

#### `subject_split_indices(subject_of_segment: 'list[int]', split: 'dict[str, Any]') -> 'dict[str, list[int]]'`

Segment indices per split from the per-segment subject list (subject-level, leak-free).

#### `validate_subject_split(subjects: 'list[int]', split: 'dict[str, Any]') -> 'None'`

Raise if split subject sets overlap, are missing, or reference unloaded subjects.

### `nssc.data.real.motion`

Motion-capture (CMU MoCap) real-data source — NOT IMPLEMENTED (honest stub).

We deliberately ship no synthetic stand-in here: a "motion" dataset that is not real
motion would silently pollute Tier-3 real-world claims. Implementing this source needs:

* Data: CMU Graphics Lab Motion Capture Database (http://mocap.cs.cmu.edu, free for
  research; cite "The data used in this project was obtained from mocap.cs.cmu.edu.
  The database was created with funding from NSF EIA-0196217"). Files are ASF/AMC
  (skeleton + per-frame joint angles, 120 Hz) or the pre-converted BVH / .c3d releases.
* Parsing: an ASF/AMC (or BVH) reader producing per-frame joint-angle vectors
  ``(T, D)`` (D≈62 for the CMU skeleton) — e.g. via a small vendored parser; there is
  no maintained lightweight PyPI dependency we want at package-import time.
* Preprocessing: subject/trial selection by motion category (walk/run/jump ...),
  root-position/orientation removal, downsampling (120 → 30/60 Hz), fixed-length
  segmentation as in :mod:`nssc.data.real.eegbci`.
* Split protocol: subject-level (never mix a subject's trials across splits) with the
  same ``metadata['split_indices']`` mechanism used by the EEG loader.
* Config schema mirroring ``source: eegbci`` (``subjects``, ``trials``/``categories``,
  ``resample_hz``, ``segment_seconds``, ``split``, ``cache_dir``).

Until that exists, ``build_motion`` raises ``NotImplementedError``.

#### `build_motion(cfg: 'dict[str, Any]')`

Placeholder: raises ``NotImplementedError`` (see module docstring for the plan).

### `nssc.data.splits`

Trajectory-level (never timestep-level) dataset splits and OOD parameter ranges.

#### `check_no_leakage(train_idx: 'np.ndarray', val_idx: 'np.ndarray', test_idx: 'np.ndarray') -> 'None'`

Raise ``ValueError`` if any trajectory index appears in more than one split.

#### `param_range_split(train_range: 'tuple[float, float]', test_range: 'tuple[float, float]', n_train: 'int', n_test: 'int', seed: 'int' = 0, name: 'str' = 'param') -> 'dict[str, Any]'`

Sample per-trajectory scalar parameter values for OOD experiments.

Train values are ``U(train_range)``, test values ``U(test_range)``; the ranges
should be disjoint (raises if they overlap). Returns
``{'name', 'train': (n_train,), 'test': (n_test,), 'train_range', 'test_range'}``.
Ranges must come from the dataset config, never from code defaults.

#### `trajectory_split(n_traj: 'int', fractions: 'tuple[float, float, float]' = (0.7, 0.15, 0.15), seed: 'int' = 0) -> 'dict[str, np.ndarray]'`

Random disjoint trajectory index sets ``{'train','val','test'}`` covering ``range(n_traj)``.

Rounding never drops trajectories: the remainder goes to train. With very
few trajectories val/test may be empty; the split is deterministic in ``seed``.

### `nssc.data.systems`

Synthetic dynamical systems; importing this package populates ``SYSTEMS``.

### `nssc.data.systems.base`

Base class for synthetic dynamical systems.

Every system subclasses :class:`DynamicalSystem`, declares its default
parameters, and registers itself in ``nssc.utils.registry.SYSTEMS`` so
configs can reference systems by name. Simulation is vectorised over a batch of
trajectories through :mod:`nssc.data.integrators`.

#### class `DynamicalSystem(params: 'dict[str, Any] | None' = None, dt: 'float | None' = None) -> 'None'`

Abstract ODE system ``dx/dt = f(t, x; params)``.

Subclasses must set the class attributes ``name``, ``state_dim``,
``default_params``, ``default_dt`` and implement :meth:`f` and
:meth:`sample_initial`. ``default_transient`` (burn-in steps) and
``integrator`` (``"rk4"``/``"euler"``) may be overridden.

- `energy(self, x: 'np.ndarray') -> 'np.ndarray | None'` — Optional conserved/characteristic quantity. ``x``: ``(..., d)`` -> ``(...)``.
- `f(self, t: 'float', x: 'np.ndarray', params: 'dict[str, Any]') -> 'np.ndarray'` — Vector field. ``x``: ``(N, d)`` -> ``(N, d)``.
- `fixed_points(self) -> 'np.ndarray | None'` — Optional known fixed points ``(k, d)``.
- `sample_initial(self, rng: 'np.random.Generator', n: 'int') -> 'np.ndarray'` — Draw ``n`` initial states -> ``(n, d)``.
- `simulate(self, n_traj: 'int', n_steps: 'int', dt: 'float | None' = None, seed: 'int' = 0, transient: 'int | None' = None, params: 'dict[str, Any] | None' = None, substeps: 'int | None' = None, ic_scale: 'float' = 1.0) -> 'np.ndarray'` — Simulate ``n_traj`` trajectories of ``n_steps`` samples.

### `nssc.data.systems.coupled_oscillators`

Chain of ``N`` linearly coupled harmonic oscillators, ``d = 2N``.

State layout: ``x = (q_1..q_N, p_1..p_N)``. Each mass has spring ``k`` to
ground and coupling ``kc`` to its chain neighbours (open boundary), plus
optional damping ``c``.

#### class `CoupledOscillators(params: 'dict[str, Any] | None' = None, dt: 'float | None' = None) -> 'None'`

``q' = p, p' = -K q - c p`` with tridiagonal stiffness ``K``.

- `energy(self, x: 'np.ndarray') -> 'np.ndarray'` — Optional conserved/characteristic quantity. ``x``: ``(..., d)`` -> ``(...)``.
- `f(self, t: 'float', x: 'np.ndarray', params: 'dict[str, Any]') -> 'np.ndarray'` — Vector field. ``x``: ``(N, d)`` -> ``(N, d)``.
- `fixed_points(self) -> 'np.ndarray'` — Optional known fixed points ``(k, d)``.
- `sample_initial(self, rng: 'np.random.Generator', n: 'int') -> 'np.ndarray'` — Draw ``n`` initial states -> ``(n, d)``.
- `stiffness(self, params: 'dict[str, Any] | None' = None) -> 'np.ndarray'` — Tridiagonal stiffness matrix ``(N, N)``.

### `nssc.data.systems.fitzhugh_nagumo`

FitzHugh-Nagumo excitable / slow-fast neuron model, ``d = 2``.

#### class `FitzHughNagumo(params: 'dict[str, Any] | None' = None, dt: 'float | None' = None) -> 'None'`

``v' = v - v^3/3 - w + I, w' = eps (v + a - b w)``; ``I=0.5`` oscillatory.

- `f(self, t: 'float', x: 'np.ndarray', params: 'dict[str, Any]') -> 'np.ndarray'` — Vector field. ``x``: ``(N, d)`` -> ``(N, d)``.
- `sample_initial(self, rng: 'np.random.Generator', n: 'int') -> 'np.ndarray'` — Draw ``n`` initial states -> ``(n, d)``.

### `nssc.data.systems.gray_scott`

Gray-Scott 1-D reaction-diffusion (method of lines), ``d = 2N``.

State layout ``x = (u_1..u_N, v_1..v_N)`` on a periodic grid of ``N`` points with
spacing ``h``. Integrated with forward Euler (default) using a 3-point Laplacian;
stability requires ``dt * max(Du, Dv) / h^2 < 0.5``.

Defaults ``(F, k) = (0.03, 0.055)`` give sustained, non-stationary pulsating
patterns in 1-D on 32 points (the 2-D "spots" pair (0.035, 0.065) decays to the
trivial state ``u=1, v=0`` in 1-D on this grid).

#### class `GrayScott1D(params: 'dict[str, Any] | None' = None, dt: 'float | None' = None) -> 'None'`

``u_t = Du u_xx - u v^2 + F(1-u)``, ``v_t = Dv v_xx + u v^2 - (F+k) v``.

- `energy(self, x: 'np.ndarray') -> 'np.ndarray'` — Total ``v`` mass (a characteristic, not conserved, quantity).
- `f(self, t: 'float', x: 'np.ndarray', params: 'dict[str, Any]') -> 'np.ndarray'` — Vector field. ``x``: ``(N, d)`` -> ``(N, d)``.
- `sample_initial(self, rng: 'np.random.Generator', n: 'int') -> 'np.ndarray'` — Homogeneous ``u=1, v=0`` with a random localised perturbation blob.

### `nssc.data.systems.harmonic`

Harmonic and damped harmonic oscillators (state ``(x, v)``, ``d = 2``).

#### class `DampedOscillator(params: 'dict[str, Any] | None' = None, dt: 'float | None' = None) -> 'None'`

``x' = v, v' = -omega^2 x - 2 zeta omega v`` (``zeta > 0`` decays).

- `energy(self, x: 'np.ndarray') -> 'np.ndarray'` — Optional conserved/characteristic quantity. ``x``: ``(..., d)`` -> ``(...)``.
- `f(self, t: 'float', x: 'np.ndarray', params: 'dict[str, Any]') -> 'np.ndarray'` — Vector field. ``x``: ``(N, d)`` -> ``(N, d)``.
- `fixed_points(self) -> 'np.ndarray'` — Optional known fixed points ``(k, d)``.
- `sample_initial(self, rng: 'np.random.Generator', n: 'int') -> 'np.ndarray'` — Draw ``n`` initial states -> ``(n, d)``.

#### class `HarmonicOscillator(params: 'dict[str, Any] | None' = None, dt: 'float | None' = None) -> 'None'`

``x' = v, v' = -omega^2 x``. Energy ``E = v^2/2 + omega^2 x^2/2`` conserved.

- `energy(self, x: 'np.ndarray') -> 'np.ndarray'` — Optional conserved/characteristic quantity. ``x``: ``(..., d)`` -> ``(...)``.
- `f(self, t: 'float', x: 'np.ndarray', params: 'dict[str, Any]') -> 'np.ndarray'` — Vector field. ``x``: ``(N, d)`` -> ``(N, d)``.
- `fixed_points(self) -> 'np.ndarray'` — Optional known fixed points ``(k, d)``.
- `sample_initial(self, rng: 'np.random.Generator', n: 'int') -> 'np.ndarray'` — Draw ``n`` initial states -> ``(n, d)``.

### `nssc.data.systems.kuramoto`

Kuramoto phase oscillators, ``d = N`` (states are phases in radians).

Natural frequencies ``omega_i ~ N(0, 1) * omega_std`` are drawn once from a
generator seeded with ``params['freq_seed']`` so the system is deterministic
given its config. Observe with :func:`observe_sin_cos` (``D = 2N``) to avoid
``2*pi`` wrap discontinuities.

#### class `Kuramoto(params: 'dict[str, Any] | None' = None, dt: 'float | None' = None) -> 'None'`

``theta_i' = omega_i + (K/N) sum_j sin(theta_j - theta_i)``.

- `energy(self, x: 'np.ndarray') -> 'np.ndarray'` — Optional conserved/characteristic quantity. ``x``: ``(..., d)`` -> ``(...)``.
- `f(self, t: 'float', x: 'np.ndarray', params: 'dict[str, Any]') -> 'np.ndarray'` — Vector field. ``x``: ``(N, d)`` -> ``(N, d)``.
- `sample_initial(self, rng: 'np.random.Generator', n: 'int') -> 'np.ndarray'` — Draw ``n`` initial states -> ``(n, d)``.

#### `observe_sin_cos(theta: 'np.ndarray') -> 'np.ndarray'`

Map phases ``(..., N)`` to ``(..., 2N)`` = ``[cos theta, sin theta]``.

#### `order_parameter(theta: 'np.ndarray') -> 'np.ndarray'`

Kuramoto order parameter ``r = |mean_j exp(i theta_j)|``; ``(..., N) -> (...)``.

### `nssc.data.systems.lorenz63`

Lorenz-63 chaotic attractor, ``d = 3``.

#### class `Lorenz63(params: 'dict[str, Any] | None' = None, dt: 'float | None' = None) -> 'None'`

``x' = s(y-x), y' = x(r-z) - y, z' = xy - bz``; sigma=10, rho=28, beta=8/3.

- `f(self, t: 'float', x: 'np.ndarray', params: 'dict[str, Any]') -> 'np.ndarray'` — Vector field. ``x``: ``(N, d)`` -> ``(N, d)``.
- `fixed_points(self) -> 'np.ndarray'` — Optional known fixed points ``(k, d)``.
- `sample_initial(self, rng: 'np.random.Generator', n: 'int') -> 'np.ndarray'` — Draw ``n`` initial states -> ``(n, d)``.

### `nssc.data.systems.lorenz96`

Lorenz-96 chaotic lattice, ``d = N`` (cyclic).

#### class `Lorenz96(params: 'dict[str, Any] | None' = None, dt: 'float | None' = None) -> 'None'`

``x_i' = (x_{i+1} - x_{i-2}) x_{i-1} - x_i + F`` on ``N`` cyclic sites.

- `f(self, t: 'float', x: 'np.ndarray', params: 'dict[str, Any]') -> 'np.ndarray'` — Vector field. ``x``: ``(N, d)`` -> ``(N, d)``.
- `fixed_points(self) -> 'np.ndarray'` — Optional known fixed points ``(k, d)``.
- `sample_initial(self, rng: 'np.random.Generator', n: 'int') -> 'np.ndarray'` — Draw ``n`` initial states -> ``(n, d)``.

### `nssc.data.systems.lotka_volterra`

Lotka-Volterra predator-prey, ``d = 2`` (positive states).

#### class `LotkaVolterra(params: 'dict[str, Any] | None' = None, dt: 'float | None' = None) -> 'None'`

``x' = a x - b x y, y' = d x y - g y``. Conserved ``V = d x - g ln x + b y - a ln y``.

- `energy(self, x: 'np.ndarray') -> 'np.ndarray'` — Optional conserved/characteristic quantity. ``x``: ``(..., d)`` -> ``(...)``.
- `f(self, t: 'float', x: 'np.ndarray', params: 'dict[str, Any]') -> 'np.ndarray'` — Vector field. ``x``: ``(N, d)`` -> ``(N, d)``.
- `fixed_points(self) -> 'np.ndarray'` — Optional known fixed points ``(k, d)``.
- `sample_initial(self, rng: 'np.random.Generator', n: 'int') -> 'np.ndarray'` — Draw ``n`` initial states -> ``(n, d)``.

### `nssc.data.systems.pendulum`

Nonlinear (optionally damped) pendulum, state ``(theta, omega)``, ``d = 2``.

#### class `Pendulum(params: 'dict[str, Any] | None' = None, dt: 'float | None' = None) -> 'None'`

``theta' = omega, omega' = -(g/L) sin(theta) - gamma omega``.

- `energy(self, x: 'np.ndarray') -> 'np.ndarray'` — ``E = omega^2/2 + (g/L)(1 - cos theta)``; conserved when ``gamma = 0``.
- `f(self, t: 'float', x: 'np.ndarray', params: 'dict[str, Any]') -> 'np.ndarray'` — Vector field. ``x``: ``(N, d)`` -> ``(N, d)``.
- `fixed_points(self) -> 'np.ndarray'` — Optional known fixed points ``(k, d)``.
- `sample_initial(self, rng: 'np.random.Generator', n: 'int') -> 'np.ndarray'` — Draw ``n`` initial states -> ``(n, d)``.

### `nssc.data.systems.vanderpol`

Van der Pol relaxation oscillator, ``d = 2``.

#### class `VanDerPol(params: 'dict[str, Any] | None' = None, dt: 'float | None' = None) -> 'None'`

``x' = v, v' = mu (1 - x^2) v - x``; limit cycle for ``mu > 0``.

- `f(self, t: 'float', x: 'np.ndarray', params: 'dict[str, Any]') -> 'np.ndarray'` — Vector field. ``x``: ``(N, d)`` -> ``(N, d)``.
- `fixed_points(self) -> 'np.ndarray'` — Optional known fixed points ``(k, d)``.
- `sample_initial(self, rng: 'np.random.Generator', n: 'int') -> 'np.ndarray'` — Draw ``n`` initial states -> ``(n, d)``.
