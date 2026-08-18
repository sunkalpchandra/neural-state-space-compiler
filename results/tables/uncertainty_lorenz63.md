# Uncertainty experiment — Gaussian transition dynamics on Lorenz-63 (5 seeds, test split)

Config: configs/experiments/lorenz63_gaussian.yaml (MLP AE d=3, residual-MLP mean + diagonal
log-variance head, latent NLL weight 0.1, 60 epochs). Predictive envelope: 32 Monte-Carlo latent
rollouts decoded, horizon 100 from a 20-step context.

| metric | mean ± std (n=5) |
|---|---|
| Gaussian NLL (per dim, horizon 100) | 9.046 ± 5.351 |
| coverage of nominal 95% interval | 0.661 ± 0.132 |
| regression ECE (levels 0.5–0.99) | 0.258 ± 0.133 |
| sharpness (mean predictive std) | 0.287 ± 0.113 |
| corr(predicted std, actual RMSE) along horizon | 0.960 ± 0.023 |
| recursive NRMSE@50 (mean prediction) | 0.258 ± 0.117 |
| recursive NRMSE@250 | 0.918 ± 0.361 |

Coverage of the 95% envelope by horizon (mean over seeds): h=1: 0.25, 10: 0.59, 25: 0.66, 50: 0.68, 100: 0.71.

Reading: the predictive spread is *informative* (it grows with, and is highly correlated to, the
actual error along the horizon) but **over-confident** (66% empirical coverage at nominal 95%,
worst at short horizons where the learned variance floor is too small). The NLL-trained model's
mean is also less accurate than the deterministic residual-MLP model trained with MSE
(NRMSE@50 0.258 ± 0.117 vs 0.0076 ± 0.0010 in synthetic_core). Calibration would need
either variance inflation on validation (post-hoc) or a heteroscedastic loss schedule — reported
as-is; not tuned on test.
