# Suite `compiled_vs_manual` — mean ± std over seeds (test split)

| dataset | model | n | params | NRMSE@1 | NRMSE@10 | NRMSE@50 | NRMSE@100 | NRMSE@250 | NRMSE mean | div. time | TF NRMSE | params |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| lorenz63 | runnerup_linear_koopman_d3 | 5 | 5169 | 0.0038 ± 0.0051 | 0.0058 ± 0.0026 | 0.0116 ± 0.0078 | 0.0630 ± 0.0994 | 0.1933 ± 0.2528 | 0.1933 ± 0.2528 | 219.6000 ± 70.2125 | 0.0049 ± 0.0084 | 5169 |
| vanderpol | compiled_linear_koopman_d3 | 5 | 5162 | 0.0007 ± 0.0002 | 0.0026 ± 0.0005 | 0.0050 ± 0.0029 | 0.0066 ± 0.0041 | 0.0113 ± 0.0053 | 0.0113 ± 0.0053 | 251 | 0.0008 ± 0.0001 | 5162 |
| vanderpol | runnerup_linear_koopman_d4 | 5 | 5538 | 0.0007 ± 0.0001 | 0.0025 ± 0.0005 | 0.0064 ± 0.0023 | 0.0154 ± 0.0058 | 0.0858 ± 0.1329 | 0.0858 ± 0.1329 | 246.8000 ± 9.3915 | 0.0007 ± 0.0001 | 5538 |
