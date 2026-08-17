# `nssc.experiment`

### `nssc.experiment`

Single-run experiment pipeline: dataset → model → train → evaluate → checkpoint → registry.

A *run config* has the shape::

    dataset: {...}            # nssc.data.builder schema, or {_file: configs/datasets/x.yaml}
    model:   {latent_dim, encoder, decoder, dynamics}
    training: {epochs, lr, rollout_horizon, loss: {...}, ...}   (TrainerConfig fields)
    windows: {context: 20, horizon: 30, stride: 5, batch_size: 64}
    eval:    {context: 20, horizons: [...], ...}                  (EvalConfig fields)
    seed: 0
    tags: [...]
    output_dir: results/raw/<name>     (checkpoint + metrics live here)

#### `prepare_data(dcfg: 'dict[str, Any]', seed: 'int | None' = None) -> 'tuple[dict[str, TrajectoryDataset], dict[str, np.ndarray], TrajectoryDataset]'`

Build → trajectory-level split → normalise with *train* statistics.

#### `resolve_dataset_cfg(dcfg: 'dict[str, Any]') -> 'dict[str, Any]'`

#### `run_experiment(cfg: 'dict[str, Any] | Config', registry: 'ExperimentRegistry | None' = None, device: 'torch.device | None' = None, log=<built-in function print>, save_ckpt: 'bool' = True) -> 'dict[str, Any]'`

Execute one run. Never raises on model failure: failures are recorded with status='failed'.

#### `summarize(metrics: 'dict[str, Any]') -> 'dict[str, Any]'`

Flat, registry-friendly subset: ``<split>/<key>`` for scalar keys.
