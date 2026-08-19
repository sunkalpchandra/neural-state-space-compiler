# Compile report — vanderpol

**Selected latent dimension:** 3
**Selected representation:** `linear` (decoder `linear`)
**Selected dynamics:** `koopman`
**Parameters:** 5162
**Runs:** 98 (0 failed) in 2372.3 min

## Reason

- selected has -42.8% rollout NRMSE vs runner-up linear+koopman@d4 (0.0005 vs 0.0009)
- -7% parameter count vs runner-up (5162 vs 5538)
- stability: verdict=stable, max local spectral radius 1.063, λ_max≈-0.020/step, blow-up fraction 0.00
- validation recon NRMSE 0.0000, one-step NRMSE 0.0001
- aggregated over 3 seeds

## Selection weights

```
{'reconstruction': 1.0, 'one_step': 1.0, 'rollout': 2.0, 'complexity': 0.1, 'stability': 1.0, 'blowup_penalty': 10.0, 'rollout_horizon_key': 'auto', 'error_floor': 0.01, 'criterion': 'multi_objective', 'extra': {}}
```

## Final ranking

| rank | candidate | J | rollout | 1-step | recon | params | ρ_max | verdict |
|---|---|---|---|---|---|---|---|---|
| 1 | linear+koopman@d3 | 0.033 | 0.0005 | 0.0001 | 0.0000 | 5162 | 1.063 | stable |
| 2 | linear+koopman@d4 | 0.111 | 0.0009 | 0.0001 | 0.0000 | 5538 | 1.065 | stable |
| 3 | mlp+residual_mlp@d2 | 0.143 | 0.0007 | 0.0002 | 0.0002 | 13446 | 1.078 | stable |
| 4 | tcn+residual_mlp@d2 | 0.723 | 0.0036 | 0.0003 | 0.0004 | 75334 | 1.080 | stable |

- stage `screen`: 63 candidates → 23 survivors
- stage `fine`: 23 candidates → 4 survivors
- stage `final`: 4 candidates → 4 survivors

## Dataset profile (excerpt)

- obs_dim=2, n_traj=100, n_steps=1000
- suggested latent dims: [1, 2]
- hints: {'candidate_latent_dims': [1, 2], 'likely_linear': True, 'likely_chaotic': False, 'noisy': False, 'long_memory': False, 'nonstationary': False}