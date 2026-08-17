# Skill: dynamical-systems

## Purpose
Ground truth for `nssc.data.synthetic`: the canonical systems, their equations, standard
parameters, integration scheme, and the invariants used to test generators and to judge
whether a compiled latent model "preserves the dynamics". Also the reference for the
stability analysis in `nssc.stability`.

## Relevant theory

All systems are ODEs `ẋ = f(x; p)` sampled at step `dt` for `T` steps, then pushed through
an observation map `g: R^n → R^D` (identity, random linear lift, random MLP lift, delay
embedding, partial observation, plus Gaussian noise `σ_obs`). Intrinsic dimension `n` is
known and is the reference for the latent-dimension recovery experiments.

### Harmonic / damped oscillator (n = 2, linear)
    ẋ = v,   v̇ = −ω² x − 2ζω v
Typical: ω = 1.0, ζ ∈ {0 (harmonic), 0.05, 0.1}, dt = 0.05, T = 1000, x0 ~ U([-2,2]²).
Eigenvalues of the continuous system: `−ζω ± iω√(1−ζ²)`. Discrete map spectral radius
`exp(−ζω dt)` (=1 for ζ=0). Ideal compiled model: d=2, linear dynamics.

### Nonlinear pendulum (n = 2)
    θ̇ = ω,   ω̇ = −(g/L) sin θ − γ ω
Typical: g/L = 1.0, γ ∈ {0, 0.1}, dt = 0.05, θ0 ~ U(−π, π), ω0 ~ U(−2, 2). Energy
`E = ½ω² + (g/L)(1−cos θ)` conserved when γ=0 (regression test: |ΔE|/E < 1e-4 over T
with RK4). Non-linear observation of an intrinsically 2-D system; good test that
Koopman/linear latents fail near separatrix.

### Van der Pol (n = 2, limit cycle)
    ẋ = v,   v̇ = μ(1 − x²) v − x
Typical: μ ∈ {0.5, 1.0, 2.0, 4.0} (parameter-range OOD experiments: train μ ∈ [0.5, 2],
test μ ∈ [2.5, 4]), dt = 0.05, T = 2000. Larger μ → stiffer relaxation oscillations;
period ≈ (3 − 2 ln 2) μ for large μ.

### Lotka–Volterra (n = 2, neutral cycles)
    ẋ = α x − β x y,   ẏ = δ x y − γ y
Typical: α = 1.1, β = 0.4, δ = 0.1, γ = 0.4, dt = 0.01, T = 3000, x0,y0 ~ U(5, 15).
Conserved: `V = δx − γ ln x + βy − α ln y`. Positivity must hold (test: all states > 0).

### FitzHugh–Nagumo (n = 2, excitable / slow–fast)
    v̇ = v − v³/3 − w + I,   ẇ = ε (v + a − b w)
Typical: a = 0.7, b = 0.8, ε = 0.08, I ∈ {0.5 (oscillatory), 0.0 (excitable)}, dt = 0.1,
T = 2000. Explicit slow (w) / fast (v) separation → the target system for H3
(multi-scale latent).

### Lorenz-63 (n = 3, chaotic)
    ẋ = σ (y − x),   ẏ = x (ρ − z) − y,   ż = x y − β z
Standard: σ = 10, ρ = 28, β = 8/3, dt = 0.01, T = 5000 after 1000-step burn-in,
x0 ~ N(0, 1)³ + (0,0,25) offset optional. Largest Lyapunov exponent λ₁ ≈ 0.906
(≈ 0.9; regression test tolerance ±0.1 with T ≥ 1e5 steps of tangent-space integration
or ±0.15 with the Rosenstein/Wolf estimate on shorter series). Lyapunov time ≈ 1.1 time
units ≈ 110 steps at dt = 0.01: recursive rollout error *must* grow ~exp(λ₁ t); a model
that reports flat error at horizon 500 for Lorenz-63 is leaking teacher forcing.
Attractor statistics used for long-horizon evaluation instead of pointwise error:
mean/variance per coordinate, z-histogram, return map of z-maxima.
OOD experiments: ρ ∈ [24, 32] train, ρ ∈ {20, 35} test.

### Lorenz-96 (n = N, chaotic, high-dim)
    ẋ_i = (x_{i+1} − x_{i−2}) x_{i−1} − x_i + F,   i = 1..N (cyclic)
Standard: N ∈ {10, 20, 40}, F = 8 (chaotic; F = 4 quasi-periodic for OOD/sanity),
dt = 0.01 (sample every 5 → 0.05), T = 4000 after burn-in, x0 = F + small perturbation
on one site. λ₁ ≈ 1.67 for N = 40, F = 8 (≈ 0.6 Lyapunov time units). This is the case
where D is genuinely high (D = N or a lift of it) and intrinsic dimension is large
(Kaplan–Yorke dim ≈ 27 for N=40) — the honest test of "d ≪ D".

### Coupled / Kuramoto oscillators (n = N phases)
    θ̇_i = ω_i + (K/N) Σ_j sin(θ_j − θ_i)
Typical: N ∈ {8, 32}, ω_i ~ N(0, 1) (or two clusters ±1 for slow/fast), K ∈ {0.5
(incoherent), 2.0 (partially synced), 4.0 (synced)}, dt = 0.05, T = 2000. Observed as
`(cos θ_i, sin θ_i)` (D = 2N) so the map is smooth. Order parameter
`r = |Σ e^{iθ_j}|/N` is the invariant to check; synced regime has low intrinsic
dimension (r ≈ 1) — a compiler should pick small d at K = 4 and larger d at K = 0.5.

### Gray–Scott reaction–diffusion (PDE → high-D)
    u_t = D_u ∇²u − u v² + F (1 − u),   v_t = D_v ∇²v + u v² − (F + k) v
Typical: D_u = 0.16, D_v = 0.08 (or 2e-5 / 1e-5 with unit-length domain), grid 32×32
or 64×64 periodic, (F, k) ∈ {(0.035, 0.065) spots, (0.030, 0.062) worms/coral,
(0.055, 0.062) mitosis}, dt = 1.0, snapshot every 20–50 steps, T = 500–1000 snapshots.
D = 2·32² = 2048 → the flagship "high-D observation, low-D structure" case for
representation experiments. Integrate with forward Euler + 5-point Laplacian (stable for
dt·D_u/h² < 0.25) — RK4 not required here.

### RK4 integration (default for all ODEs)
    k1 = f(x_t),  k2 = f(x_t + dt/2 k1),  k3 = f(x_t + dt/2 k2),  k4 = f(x_t + dt k3)
    x_{t+1} = x_t + dt/6 (k1 + 2k2 + 2k3 + k4)
Global error O(dt⁴). Use `nssc.data.synthetic.integrate.rk4(f, x0, dt, n_steps,
substeps)` with `substeps ≥ 1` so `dt_sample = dt·substeps` is decoupled from the
integrator step. Compare against `scipy.integrate.solve_ivp(method="DOP853",
rtol=1e-10)` in a unit test on Lorenz-63 for 200 steps (< 1e-3 max abs error).

### Stability / spectral quantities (used by `nssc.stability`)
- Discrete Jacobian `J_t = ∂F_θ/∂z` at points along a trajectory; spectral radius
  `ρ(J)`; for linear latent dynamics `z_{t+1} = A z_t`, `ρ(A) < 1` ⇒ contracting.
- Finite-time Lyapunov estimate: QR of products of Jacobians along a rollout
  (`nssc.stability.lyapunov.benettin`); the true λ₁ of the source system is the
  reference (`≈0.9` L63, `≈1.67` L96-40, `≤0` for damped/limit-cycle systems where the
  leading exponent is 0 along the cycle).
- Norm growth `‖z_t‖ / ‖z_0‖` over long free rollouts (H = 500–2000): divergence to
  inf/NaN or collapse to a fixed point are both instabilities.

## Project-specific conventions
- Generators live in `src/nssc/data/synthetic/<system>.py`, each exposing
  `f(x, p) -> dx` (numpy, vectorized over a leading batch axis) and a
  `SystemSpec` (name, n, default params, dt, burn_in, param ranges for OOD, invariants).
- Registered under `@register("system", "lorenz63")` etc. so the profiler and configs
  reference systems by name.
- Config: `configs/datasets/<system>.yaml` sets `params`, `dt`, `T`, `n_traj`,
  `obs_map`, `obs_dim`, `noise_std`, `split` (train/val/test trajectory counts, and
  `param_range_train` / `param_range_test` for OOD).
- Datasets are generated deterministically from `(system, params, seed)`; cached to
  `data/cache/<hash>.npz` (gitignored). The dataset *version* string in the registry is
  the config hash.
- Trajectory arrays are `(n_traj, T, D)` float32; ground-truth states `(n_traj, T, n)`
  are kept alongside for alignment analyses only, never fed to models.

## Implementation requirements
- All generators vectorized (batch of trajectories in one RK4 call); no Python loops
  over trajectories.
- Every system has a unit test for: finite output, invariant (energy / V / order
  parameter / attractor bounds) within tolerance, determinism under seed, and shape.
- Lorenz-63 has a regression test: λ₁ estimated by `nssc.stability.lyapunov` on the
  ground-truth system ∈ [0.75, 1.05].
- Observation maps are separate objects (`nssc.data.observation`) with their own seed
  so the same latent trajectory can be observed in several ways.
- Burn-in is always discarded before splitting.

## Common failure modes
- Too large `dt` for RK4 on Lorenz (dt = 0.05 without substeps drifts off the attractor).
- Sampling test trajectories from the same initial-condition seed stream as train
  (leakage): use disjoint seed ranges per split.
- Splitting long single trajectories into windows and randomizing windows across splits
  — forbidden; splits are per trajectory (CLAUDE.md).
- Kuramoto observed as raw θ (wraps at 2π → discontinuous targets). Use (cos, sin).
- Comparing pointwise MSE at horizon 500 on chaotic systems and concluding "all models
  fail" — use attractor statistics / valid-prediction-time (time until NRMSE > 0.5)
  for chaotic systems in addition to MSE.
- Gray–Scott with unstable Euler step (checkerboard blow-up): assert dt·D/h² < 0.25.

## Validation checklist
- [ ] Equations and defaults in code match this document (or this document is updated
      *and* the dataset config version bumps).
- [ ] Invariant test present for the system.
- [ ] Intrinsic dimension `n` recorded in `SystemSpec` (used by latent-dim experiments).
- [ ] OOD parameter ranges declared in the dataset config, not in code.
- [ ] Chaotic systems have λ₁ reference recorded and used in evaluation notes.
- [ ] Observation map + noise level appear in the dataset config hash.
