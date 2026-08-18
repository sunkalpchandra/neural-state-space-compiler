# Compile report — lorenz63

**Selected latent dimension:** 3
**Selected representation:** `mlp` (decoder `mlp`)
**Selected dynamics:** `residual_mlp`
**Parameters:** 13833
**Runs:** 127 (0 failed) in 1324.4 min

## Reason

- selected has -40.1% rollout NRMSE vs runner-up linear+koopman@d3 (0.0164 vs 0.0273)
- +168% parameter count vs runner-up (13833 vs 5169)
- smallest candidate linear+residual_mlp@d3 (4635 params) has 2.36× the selected model's rollout NRMSE
- stability: verdict=stable, max local spectral radius 1.103, λ_max≈0.432/step, blow-up fraction 0.00
- validation recon NRMSE 0.0006, one-step NRMSE 0.0005
- aggregated over 3 seeds

## Selection weights

```
{'reconstruction': 1.0, 'one_step': 1.0, 'rollout': 2.0, 'complexity': 0.1, 'stability': 1.0, 'blowup_penalty': 10.0, 'rollout_horizon_key': 'auto', 'error_floor': 0.01, 'criterion': 'multi_objective', 'extra': {}}
```

## Final ranking

| rank | candidate | J | rollout | 1-step | recon | params | ρ_max | verdict |
|---|---|---|---|---|---|---|---|---|
| 1 | mlp+residual_mlp@d3 | 0.181 | 0.0164 | 0.0005 | 0.0006 | 13833 | 1.103 | stable |
| 2 | linear+koopman@d3 | 0.752 | 0.0273 | 0.0002 | 0.0000 | 5169 | 1.105 | stable |
| 3 | linear+residual_mlp@d3 | 1.295 | 0.0388 | 0.0003 | 0.0000 | 4635 | 1.104 | stable |
| 4 | linear+residual_mlp@d4 | 32.145 | 15966.6039 | 0.0005 | 0.0000 | 4771 | 1.106 | stable |

- stage `screen`: 84 candidates → 30 survivors
- stage `fine`: 30 candidates → 4 survivors
- stage `final`: 4 candidates → 4 survivors

## Dataset profile (excerpt)

- obs_dim=3, n_traj=100, n_steps=500
- suggested latent dims: [1, 2, 3]
- hints: {'candidate_latent_dims': [1, 2, 3], 'likely_linear': False, 'likely_chaotic': True, 'noisy': False, 'long_memory': False, 'nonstationary': False}