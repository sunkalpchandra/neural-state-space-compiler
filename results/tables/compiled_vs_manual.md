# Suite `compiled_vs_manual` — mean ± std over seeds (test split)

| dataset | model | n | params | NRMSE@1 | NRMSE@10 | NRMSE@50 | NRMSE@100 | NRMSE@250 | NRMSE mean | div. time | TF NRMSE | params |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| lorenz63 | runnerup_linear_koopman_d3 | 5 | 5169 | 0.0038 ± 0.0051 | 0.0058 ± 0.0026 | 0.0116 ± 0.0078 | 0.0630 ± 0.0994 | 0.1933 ± 0.2528 | 0.1933 ± 0.2528 | 219.6000 ± 70.2125 | 0.0049 ± 0.0084 | 5169 |

## Paired comparisons vs `compiled_mlp_residual_mlp_d3` (NRMSE@50, same seeds)

| dataset | model | n | mean diff (ref − model) | paired t p | Wilcoxon p |
|---|---|---|---|---|---|