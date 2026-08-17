# Open questions

Numbered, append-only. Each question names the experiment or analysis that would answer
it. Resolved questions get a `Resolved:` line pointing at an experiment-log entry, and
stay here.

### Q-001 — Which intrinsic-dimension estimator should the profiler trust?
PCA-energy elbows undercount for nonlinear manifolds (pendulum, Lorenz); two-NN and
correlation-dimension estimates are noisy at small n_traj. Candidate grid derives from
this estimate, so a bad estimate can exclude the right `d`. → Cell B: compare
`est_intrinsic_dim` (both estimators) to the `d` that minimizes validation NRMSE@25 on
S1–S6; decide whether the grid should always include `{2,3,4,8}` regardless.

### Q-002 — Does the coarse screen discard eventual winners?
Short-budget one-step/NRMSE@10 screening may eliminate slow-starting families (neural
ODE, Koopman with large dictionary) that win at H = 250. → Cell L: exhaustive vs staged
on S3/S6/S9 small pools; measure "regret" of staged choice.

### Q-003 — What is the right long-horizon metric for chaotic systems?
Pointwise NRMSE saturates after ~2 Lyapunov times; VPT and attractor-statistics errors
(marginal W1, mean/std) are proposed. Are they consistent with each other and with the
spectral/Lyapunov stability metrics? → Cells D and G on S6/S7: correlation across seeds
and models between VPT, W1, and λ̂₁ − λ_ref.

### Q-004 — Should `C_instability` be relative to the data's λ_ref or absolute?
An absolute penalty on ρ(J) > 1 penalizes chaotic data unfairly; a relative one needs a
reliable λ_ref estimate on real data. → Cell G (synthetic, λ_ref known) then K.

### Q-005 — Is the multi-scale gain (H3) a capacity effect?
Slow/fast models may simply have a different effective capacity or optimization
trajectory. Control: single-scale model with equal `d` and equal params, plus a
single-scale model with 2× params. → Cell F.

### Q-006 — Teacher-forced vs latent recursive rollout for encoders that need context
GRU/TCN encoders need a context window to produce `z_c`; the compiled model's recursive
rollout starts from that. Should the context length be part of the candidate space, and
how does it interact with `d`? → Cell D sensitivity: context ∈ {10, 50, 200} on S3/S6.

### Q-007 — How should diverged rollouts enter the score and tables?
Currently NRMSE capped at 2.0 in `L_rollout` and reported separately as
`diverged_frac`. Does the cap value change compiler choices? → Cell H sensitivity:
cap ∈ {1, 2, 5}.

### Q-008 — What does the compiler do when the data are near-linear plus noise (S1, D = 64 MLP lift)?
Expected: d = 2, linear/affine dynamics. If it picks MLP dynamics at d = 4, is that the
complexity weight λ4 being too small or the noise term in recon? → Cell A/H.

### Q-009 — Real data: which public EEG and motion datasets, and are subject-level splits
large enough for n = 5 seed statistics?
→ Data engineer to propose in `docs/experiments.md` (license, subjects, sampling rate)
before cell K; principal researcher signs off in `decisions.md`.

### Q-010 — Alignment analysis: linear vs nonlinear alignment for latent interpretability
Linear R² from `z` to ground-truth state may understate a good but nonlinearly-warped
latent. Should F9 report both linear R² and a small-MLP R² (held-out)? Which is the
honest one to caption? → Cell C on S2/S6.
