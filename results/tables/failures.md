# Failure analysis

156 runs; category counts: {'poor_long_horizon': 63, 'latent_instability': 6, 'chaotic_divergence': 1, 'poor_reconstruction': 10}

| id | dataset | model | seed | status | categories |
|---|---|---|---|---|---|
| EXP-0009 | lorenz63 | pca+linear@d3 | 0 | completed | poor_long_horizon |
| EXP-0010 | lorenz63 | pca+linear@d3 | 1 | completed | poor_long_horizon |
| EXP-0011 | lorenz63 | pca+linear@d3 | 2 | completed | poor_long_horizon |
| EXP-0012 | lorenz63 | pca+linear@d3 | 3 | completed | poor_long_horizon |
| EXP-0013 | lorenz63 | pca+linear@d3 | 4 | completed | poor_long_horizon |
| EXP-0014 | lorenz63 | linear+linear@d3 | 0 | completed | latent_instability, chaotic_divergence |
| EXP-0015 | lorenz63 | linear+linear@d3 | 1 | completed | latent_instability |
| EXP-0017 | lorenz63 | linear+linear@d3 | 2 | completed | latent_instability |
| EXP-0018 | lorenz63 | linear+linear@d3 | 3 | completed | poor_long_horizon |
| EXP-0019 | lorenz63 | linear+linear@d3 | 4 | completed | latent_instability |
| EXP-0020 | lorenz63 | mlp+mlp@d3 | 0 | completed | ok |
| EXP-0024 | lorenz63 | mlp+mlp@d3 | 1 | completed | ok |
| EXP-0027 | lorenz63 | mlp+mlp@d3 | 2 | completed | ok |
| EXP-0033 | lorenz63 | mlp+mlp@d3 | 3 | completed | ok |
| EXP-0038 | lorenz63 | mlp+mlp@d3 | 4 | completed | ok |
| EXP-0044 | lorenz63 | mlp+residual_mlp@d3 | 0 | completed | ok |
| EXP-0048 | lorenz63 | mlp+residual_mlp@d3 | 1 | completed | ok |
| EXP-0051 | lorenz63 | mlp+residual_mlp@d3 | 2 | completed | ok |
| EXP-0054 | lorenz63 | mlp+residual_mlp@d3 | 3 | completed | ok |
| EXP-0059 | lorenz63 | mlp+residual_mlp@d3 | 4 | completed | ok |
| EXP-0065 | lorenz63 | baseline:persistence | 0 | completed | poor_long_horizon |
| EXP-0066 | lorenz63 | baseline:persistence | 1 | completed | poor_long_horizon |
| EXP-0067 | lorenz63 | baseline:persistence | 2 | completed | poor_long_horizon |
| EXP-0068 | lorenz63 | baseline:persistence | 3 | completed | poor_long_horizon |
| EXP-0069 | lorenz63 | baseline:persistence | 4 | completed | poor_long_horizon |
| EXP-0070 | lorenz63 | baseline:gru | 0 | completed | ok |
| EXP-0078 | lorenz63 | baseline:gru | 1 | completed | ok |
| EXP-0085 | lorenz63 | baseline:gru | 2 | completed | ok |
| EXP-0094 | lorenz63 | baseline:gru | 3 | completed | ok |
| EXP-0101 | lorenz63 | baseline:gru | 4 | completed | ok |
| EXP-0106 | lorenz63 | baseline:lstm | 0 | completed | ok |
| EXP-0114 | lorenz63 | baseline:lstm | 1 | completed | ok |
| EXP-0127 | lorenz63 | baseline:lstm | 2 | completed | ok |
| EXP-0133 | lorenz63 | baseline:lstm | 3 | completed | ok |
| EXP-0139 | lorenz63 | baseline:lstm | 4 | completed | ok |
| EXP-0148 | lorenz63 | baseline:tcn | 0 | completed | ok |
| EXP-0158 | lorenz63 | baseline:tcn | 1 | completed | ok |
| EXP-0167 | lorenz63 | baseline:tcn | 2 | completed | ok |
| EXP-0175 | lorenz63 | baseline:tcn | 3 | completed | ok |
| EXP-0179 | lorenz63 | baseline:tcn | 4 | completed | ok |
| EXP-0182 | lorenz63 | baseline:transformer | 0 | completed | poor_long_horizon |
| EXP-0184 | lorenz63 | baseline:transformer | 1 | completed | poor_long_horizon |
| EXP-0187 | lorenz63 | baseline:transformer | 2 | completed | poor_long_horizon |
| EXP-0191 | lorenz63 | baseline:transformer | 3 | completed | poor_long_horizon |
| EXP-0194 | lorenz63 | baseline:transformer | 4 | completed | poor_long_horizon |
| EXP-0198 | lorenz63 | baseline:ssm | 0 | completed | ok |
| EXP-0202 | lorenz63 | baseline:ssm | 1 | completed | poor_long_horizon |
| EXP-0204 | lorenz63 | baseline:ssm | 2 | completed | ok |
| EXP-0206 | lorenz63 | baseline:ssm | 3 | completed | ok |
| EXP-0208 | lorenz63 | baseline:ssm | 4 | completed | poor_long_horizon |
| EXP-0210 | vanderpol | pca+linear@d2 | 0 | completed | ok |
| EXP-0211 | vanderpol | pca+linear@d2 | 1 | completed | ok |
| EXP-0212 | vanderpol | pca+linear@d2 | 2 | completed | ok |
| EXP-0213 | vanderpol | pca+linear@d2 | 3 | completed | ok |
| EXP-0214 | vanderpol | pca+linear@d2 | 4 | completed | ok |
| EXP-0215 | vanderpol | linear+linear@d2 | 0 | completed | ok |
| EXP-0216 | vanderpol | linear+linear@d2 | 1 | completed | ok |
| EXP-0217 | vanderpol | linear+linear@d2 | 2 | completed | ok |
| EXP-0218 | vanderpol | linear+linear@d2 | 3 | completed | ok |
| EXP-0219 | vanderpol | linear+linear@d2 | 4 | completed | ok |
| EXP-0220 | vanderpol | mlp+mlp@d2 | 0 | completed | ok |
| EXP-0222 | vanderpol | mlp+mlp@d2 | 1 | completed | ok |
| EXP-0223 | vanderpol | mlp+mlp@d2 | 2 | completed | ok |
| EXP-0224 | vanderpol | mlp+mlp@d2 | 3 | completed | ok |
| EXP-0225 | vanderpol | mlp+mlp@d2 | 4 | completed | ok |
| EXP-0226 | vanderpol | mlp+residual_mlp@d2 | 0 | completed | ok |
| EXP-0227 | vanderpol | mlp+residual_mlp@d2 | 1 | completed | ok |
| EXP-0228 | vanderpol | mlp+residual_mlp@d2 | 2 | completed | ok |
| EXP-0230 | vanderpol | mlp+residual_mlp@d2 | 3 | completed | ok |
| EXP-0231 | vanderpol | mlp+residual_mlp@d2 | 4 | completed | ok |
| EXP-0233 | vanderpol | baseline:persistence | 0 | completed | poor_long_horizon |
| EXP-0234 | vanderpol | baseline:persistence | 1 | completed | poor_long_horizon |
| EXP-0235 | vanderpol | baseline:persistence | 2 | completed | poor_long_horizon |
| EXP-0236 | vanderpol | baseline:persistence | 3 | completed | poor_long_horizon |
| EXP-0237 | vanderpol | baseline:persistence | 4 | completed | poor_long_horizon |
| EXP-0238 | vanderpol | baseline:gru | 0 | completed | ok |
| EXP-0240 | vanderpol | baseline:gru | 1 | completed | ok |
| EXP-0242 | vanderpol | baseline:gru | 2 | completed | ok |
| EXP-0244 | vanderpol | baseline:gru | 3 | running | ok |
| EXP-0246 | lorenz63 | baseline:persistence | 1 | completed | poor_long_horizon |
| EXP-0247 | lorenz63 | baseline:persistence | 2 | completed | poor_long_horizon |
| EXP-0248 | lorenz63 | baseline:persistence | 3 | completed | poor_long_horizon |
| EXP-0249 | lorenz63 | baseline:persistence | 4 | completed | poor_long_horizon |
| EXP-0250 | lorenz63 | baseline:gru | 0 | running | ok |
| EXP-0251 | vanderpol | baseline:gru | 3 | completed | ok |
| EXP-0253 | vanderpol | baseline:gru | 4 | completed | ok |
| EXP-0255 | vanderpol | baseline:lstm | 0 | completed | ok |
| EXP-0258 | vanderpol | baseline:lstm | 1 | completed | ok |
| EXP-0259 | vanderpol | baseline:lstm | 2 | completed | ok |
| EXP-0262 | vanderpol | baseline:lstm | 3 | completed | ok |
| EXP-0265 | vanderpol | baseline:lstm | 4 | completed | ok |
| EXP-0266 | vanderpol | baseline:tcn | 0 | completed | ok |
| EXP-0267 | vanderpol | baseline:tcn | 1 | completed | ok |
| EXP-0270 | vanderpol | baseline:tcn | 2 | completed | ok |
| EXP-0274 | vanderpol | baseline:tcn | 3 | completed | ok |
| EXP-0276 | vanderpol | baseline:tcn | 4 | completed | ok |
| EXP-0278 | vanderpol | baseline:transformer | 0 | completed | poor_long_horizon |
| EXP-0280 | vanderpol | baseline:transformer | 1 | completed | poor_long_horizon |
| EXP-0281 | vanderpol | baseline:transformer | 2 | completed | poor_long_horizon |
| EXP-0282 | vanderpol | baseline:transformer | 3 | completed | poor_long_horizon |
| EXP-0284 | vanderpol | baseline:transformer | 4 | completed | poor_long_horizon |
| EXP-0285 | vanderpol | baseline:ssm | 0 | completed | ok |
| EXP-0288 | vanderpol | baseline:ssm | 1 | completed | ok |
| EXP-0291 | vanderpol | baseline:ssm | 2 | completed | ok |
| EXP-0293 | vanderpol | baseline:ssm | 3 | completed | ok |
| EXP-0295 | vanderpol | baseline:ssm | 4 | completed | ok |
| EXP-0297 | lorenz63 | pca+linear@d3 | 0 | completed | poor_reconstruction, poor_long_horizon |
| EXP-0298 | lorenz63 | pca+linear@d3 | 1 | completed | poor_reconstruction, poor_long_horizon |
| EXP-0299 | lorenz63 | pca+linear@d3 | 2 | completed | poor_reconstruction, poor_long_horizon |
| EXP-0300 | lorenz63 | pca+linear@d3 | 3 | completed | poor_reconstruction, poor_long_horizon |
| EXP-0301 | lorenz63 | pca+linear@d3 | 4 | completed | poor_reconstruction, poor_long_horizon |
| EXP-0302 | lorenz63 | linear+linear@d3 | 0 | completed | poor_reconstruction, poor_long_horizon |
| EXP-0303 | lorenz63 | linear+linear@d3 | 1 | completed | poor_reconstruction, poor_long_horizon |
| EXP-0304 | lorenz63 | linear+linear@d3 | 2 | completed | poor_reconstruction, poor_long_horizon |
| EXP-0305 | lorenz63 | linear+linear@d3 | 3 | completed | poor_reconstruction, poor_long_horizon |
| EXP-0306 | lorenz63 | linear+linear@d3 | 4 | completed | poor_reconstruction, poor_long_horizon |
| EXP-0307 | lorenz63 | mlp+mlp@d3 | 0 | completed | poor_long_horizon |
| EXP-0308 | lorenz63 | mlp+mlp@d3 | 1 | completed | poor_long_horizon |
| EXP-0309 | lorenz63 | mlp+mlp@d3 | 2 | completed | poor_long_horizon |
| EXP-0310 | lorenz63 | mlp+mlp@d3 | 3 | completed | poor_long_horizon |
| EXP-0311 | lorenz63 | mlp+mlp@d3 | 4 | completed | poor_long_horizon |
| EXP-0312 | lorenz63 | mlp+residual_mlp@d3 | 0 | completed | latent_instability |
| EXP-0313 | lorenz63 | mlp+residual_mlp@d3 | 1 | completed | poor_long_horizon |
| EXP-0315 | lorenz63 | mlp+residual_mlp@d3 | 2 | completed | latent_instability |
| EXP-0316 | lorenz63 | mlp+residual_mlp@d3 | 3 | completed | poor_long_horizon |
| EXP-0317 | lorenz63 | mlp+residual_mlp@d3 | 4 | completed | poor_long_horizon |
| EXP-0318 | lorenz63 | baseline:persistence | 0 | completed | poor_long_horizon |
| EXP-0319 | lorenz63 | baseline:persistence | 1 | completed | poor_long_horizon |
| EXP-0320 | lorenz63 | baseline:persistence | 2 | completed | poor_long_horizon |
| EXP-0321 | lorenz63 | baseline:persistence | 3 | completed | poor_long_horizon |
| EXP-0322 | lorenz63 | baseline:persistence | 4 | completed | poor_long_horizon |
| EXP-0323 | lorenz63 | baseline:gru | 0 | completed | ok |
| EXP-0324 | lorenz63 | baseline:gru | 1 | completed | ok |
| EXP-0326 | lorenz63 | baseline:gru | 2 | completed | ok |
| EXP-0327 | lorenz63 | baseline:gru | 3 | completed | ok |
| EXP-0329 | lorenz63 | baseline:gru | 4 | completed | ok |
| EXP-0330 | lorenz63 | baseline:lstm | 0 | completed | ok |
| EXP-0333 | lorenz63 | baseline:lstm | 1 | completed | ok |
| EXP-0335 | lorenz63 | baseline:lstm | 2 | completed | ok |
| EXP-0337 | lorenz63 | baseline:lstm | 3 | completed | ok |
| EXP-0338 | lorenz63 | baseline:lstm | 4 | completed | ok |
| EXP-0341 | lorenz63 | baseline:tcn | 0 | completed | ok |
| EXP-0343 | lorenz63 | baseline:tcn | 1 | completed | ok |
| EXP-0347 | lorenz63 | baseline:tcn | 2 | completed | ok |
| EXP-0349 | lorenz63 | baseline:tcn | 3 | completed | ok |
| EXP-0351 | lorenz63 | baseline:tcn | 4 | completed | ok |
| EXP-0355 | lorenz63 | baseline:transformer | 0 | completed | poor_long_horizon |
| EXP-0356 | lorenz63 | baseline:transformer | 1 | completed | poor_long_horizon |
| EXP-0357 | lorenz63 | baseline:transformer | 2 | completed | poor_long_horizon |
| EXP-0358 | lorenz63 | baseline:transformer | 3 | completed | poor_long_horizon |
| EXP-0359 | lorenz63 | baseline:transformer | 4 | completed | poor_long_horizon |
| EXP-0360 | lorenz63 | baseline:ssm | 0 | completed | poor_long_horizon |
| EXP-0363 | lorenz63 | baseline:ssm | 1 | completed | poor_long_horizon |
| EXP-0364 | lorenz63 | baseline:ssm | 2 | completed | ok |
| EXP-0367 | lorenz63 | baseline:ssm | 3 | completed | poor_long_horizon |
| EXP-0368 | lorenz63 | baseline:ssm | 4 | completed | ok |