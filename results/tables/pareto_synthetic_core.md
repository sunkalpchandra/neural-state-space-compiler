# Pareto analysis — test/recursive/nrmse@50 vs parameter count

## lorenz63  (dominated area 2.405)

| model | kind | params | error (mean ± std) | n | Pareto-efficient |
|---|---|---|---|---|---|
| persistence | baseline | 0 | 1.2950 ± 0.0000 | 5 | **yes** |
| pca_linear_d3 | latent | 9 | 1.0095 ± 0.0118 | 5 | **yes** |
| linae_linear_d3 | latent | 33 | 20.4775 ± 31.1672 | 5 |  |
| mlpae_mlp_d3 | latent | 13833 | 0.0960 ± 0.0580 | 5 |  |
| mlpae_resmlp_d3 | latent | 13833 | 0.0076 ± 0.0010 | 5 | **yes** |
| ssm_small | baseline | 14691 | 0.0977 ± 0.0401 | 5 |  |
| tcn_small | baseline | 16867 | 0.0236 ± 0.0056 | 5 |  |
| transformer_small | baseline | 42083 | 0.1900 ± 0.0440 | 5 |  |
| gru_medium | baseline | 150531 | 0.0125 ± 0.0017 | 5 |  |
| lstm_medium | baseline | 200579 | 0.0077 ± 0.0013 | 5 |  |

## vanderpol  (dominated area 3.645)

| model | kind | params | error (mean ± std) | n | Pareto-efficient |
|---|---|---|---|---|---|
| persistence | baseline | 0 | 1.0772 ± 0.0000 | 5 | **yes** |
| pca_linear_d3 | latent | 4 | 0.3628 ± 0.0027 | 5 | **yes** |
| linae_linear_d3 | latent | 16 | 0.3979 ± 0.0266 | 5 |  |
| mlpae_mlp_d3 | latent | 13446 | 0.0028 ± 0.0008 | 5 |  |
| mlpae_resmlp_d3 | latent | 13446 | 0.0026 ± 0.0009 | 5 | **yes** |
| gru_medium | baseline | 150018 | 0.0158 ± 0.0166 | 4 |  |
