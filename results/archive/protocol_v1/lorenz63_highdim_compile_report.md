# Compile report — lorenz63

**Selected latent dimension:** 8
**Selected representation:** `gru` (decoder `mlp`)
**Selected dynamics:** `residual_mlp`
**Parameters:** 39632
**Runs:** 123 (0 failed) in 221.3 min

## Reason

- 1.00× lower 250-step rollout NRMSE than the best linear-dynamics candidate (pca+linear@d16: 0.9265 vs selected 0.9235)
- selected has -0.3% rollout NRMSE vs runner-up pca+linear@d16 (0.9235 vs 0.9265)
- +15381% parameter count vs runner-up (39632 vs 256)
- smallest candidate pca+linear@d16 (256 params) has 1.00× the selected model's rollout NRMSE
- stability: verdict=stable, max local spectral radius 1.352, λ_max≈1.111/step, blow-up fraction 0.00
- validation recon NRMSE 0.1209, one-step NRMSE 0.1239
- aggregated over 3 seeds

## Selection weights

```
{'reconstruction': 1.0, 'one_step': 1.0, 'rollout': 2.0, 'complexity': 0.1, 'stability': 1.0, 'blowup_penalty': 10.0, 'rollout_horizon_key': 'auto', 'error_floor': 0.01, 'criterion': 'multi_objective', 'extra': {}}
```

## Final ranking

| rank | candidate | J | rollout | 1-step | recon | params | ρ_max | verdict |
|---|---|---|---|---|---|---|---|---|
| 1 | gru+residual_mlp@d8 | 0.442 | 0.9235 | 0.1239 | 0.1209 | 39632 | 1.352 | stable |
| 2 | pca+linear@d16 | 0.706 | 0.9265 | 0.2049 | 0.1437 | 256 | 0.981 | stable |
| 3 | gru+residual_mlp@d4 | 2.465 | 1.3240 | 0.1260 | 0.1241 | 38600 | 1.433 | stable |
| 4 | gru+residual_mlp@d3 | 3.588 | 2.7540 | 0.1289 | 0.1279 | 38342 | 1.425 | stable |

- stage `screen`: 85 candidates → 26 survivors
- stage `fine`: 26 candidates → 4 survivors
- stage `final`: 4 candidates → 4 survivors

## Dataset profile (excerpt)

- obs_dim=64, n_traj=100, n_steps=500
- suggested latent dims: [2, 4, 6, 8, 10, 13, 16, 20, 26, 32, 40]
- hints: {'candidate_latent_dims': [2, 4, 6, 8, 10, 13, 16, 20, 26, 32, 40], 'likely_linear': False, 'likely_chaotic': True, 'noisy': True, 'long_memory': False, 'nonstationary': False}