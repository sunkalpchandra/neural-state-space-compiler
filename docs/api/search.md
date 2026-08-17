# `nssc.search`

### `nssc.search`

Candidate space, resumable search state and staged search over latent state-space models.

### `nssc.search.runner`

Benchmark suite runner.

A suite YAML::

    name: synthetic_core
    datasets:                       # each entry is a run-config dataset block (or {_file: ...})
      lorenz63: {_file: configs/datasets/lorenz63.yaml}
    windows: {...}  training: {...}  eval: {...}     # shared base
    seeds: [0, 1, 2, 3, 4]
    models:                          # latent state-space models (nssc.experiment)
      mlp_resmlp_d8: {latent_dim: 8, encoder: mlp, dynamics: residual_mlp}
    baselines:                       # sequence forecasters (nssc.baselines)
      gru_medium: {baseline: gru, size: medium}
    per_dataset: {lorenz63: {models: {...}, training: {...}}}   # optional overrides
    output_dir: results/raw/benchmarks/synthetic_core

Runs already present in the registry with the same config hash + seed and
status ``completed`` are skipped, so a suite is resumable and idempotent.

#### `expand_suite(suite: 'dict[str, Any]') -> 'list[dict[str, Any]]'`

Materialise the list of run configs (with ``_kind`` = latent|baseline).

#### `run_suite(path: 'str | Path', overrides: 'list[str] | None' = None, device: 'str | None' = None, log=<built-in function print>, registry: 'ExperimentRegistry | None' = None, dry_run: 'bool' = False, only: 'str | None' = None) -> 'list[dict[str, Any]]'`

### `nssc.search.space`

Candidate specification and generation.

A candidate is a fully specified model config (latent dim, encoder, decoder,
dynamics + kwargs). Candidates are generated from the compiler config's
lists; ``latent_dims: auto`` defers to the dataset profile's suggestions.

#### class `CandidateSpec(latent_dim: 'int', encoder: 'str', dynamics: 'str', decoder: 'str' = 'mlp', encoder_kwargs: 'dict[str, Any]' = <factory>, decoder_kwargs: 'dict[str, Any]' = <factory>, dynamics_kwargs: 'dict[str, Any]' = <factory>, training_overrides: 'dict[str, Any]' = <factory>, tags: 'tuple[str, ...]' = ()) -> None`

CandidateSpec(latent_dim: 'int', encoder: 'str', dynamics: 'str', decoder: 'str' = 'mlp', encoder_kwargs: 'dict[str, Any]' = <factory>, decoder_kwargs: 'dict[str, Any]' = <factory>, dynamics_kwargs: 'dict[str, Any]' = <factory>, training_overrides: 'dict[str, Any]' = <factory>, tags: 'tuple[str, ...]' = ())

- `model_config(self) -> 'dict[str, Any]'`
- `to_dict(self) -> 'dict[str, Any]'`

#### `generate_candidates(cfg: 'dict[str, Any]', profile: 'dict[str, Any] | None' = None, obs_dim: 'int' = 1) -> 'list[CandidateSpec]'`

Cartesian product of latent_dims × encoders × dynamics (× hidden sizes optional).

``cfg`` keys: ``latent_dims`` (list|auto), ``encoders`` (list of str|{name,kwargs}),
``dynamics`` (same), ``decoders`` (optional mapping encoder→decoder spec),
``hidden_dims`` (optional list of hidden sizes applied to mlp-like components),
``exclude`` (list of {encoder,dynamics,latent_dim} partial matches),
``max_candidates`` (int).

#### `resolve_latent_dims(spec: 'Any', profile: 'dict[str, Any] | None', obs_dim: 'int') -> 'list[int]'`

### `nssc.search.staged`

Staged, resumable search over candidates.

    coarse screening → discard clearly inferior → fine evaluation → long-horizon
    validation (more seeds) → complexity/stability analysis → final compilation

Each stage is a dict in the compiler config::

    stages:
      - {name: screen, epochs: 20,  seeds: [0],       keep_top: 8, keep_frac: 0.5}
      - {name: fine,   epochs: 100, seeds: [0],       keep_top: 3}
      - {name: final,  epochs: 200, seeds: [0, 1, 2]}

Every candidate run goes through :func:`nssc.experiment.run_experiment`, so it is
registered in the experiment registry and checkpointed like any other run.

#### class `StagedSearch(base_run_cfg: 'dict[str, Any]', stages: 'list[dict[str, Any]]', weights: 'ScoreWeights', output_dir: 'str | Path', registry: 'ExperimentRegistry | None' = None, device: 'torch.device | None' = None, log=<built-in function print>, reuse_registry: 'bool' = True) -> 'None'`

- `run(self, candidates: 'list[CandidateSpec]') -> 'dict[str, Any]'`
- `run_cfg_for(self, cand: 'CandidateSpec', stage: 'dict[str, Any]', seed: 'int') -> 'dict[str, Any]'`
- `run_stage(self, stage: 'dict[str, Any]', candidates: 'list[CandidateSpec]') -> 'tuple[list[dict[str, Any]], list[CandidateSpec]]'`

### `nssc.search.state`

Persistent, resumable search state (JSON file in the compile output directory).

Keyed by ``(stage, candidate_id, seed)``. If a search crashes after N runs, the
next invocation skips those N and continues.

#### class `SearchState(path: 'str | Path') -> 'None'`

- `get(self, stage: 'str', cand_id: 'str', seed: 'int') -> 'dict[str, Any] | None'`
- `get_stage(self, stage: 'str') -> 'dict[str, Any] | None'`
- `has(self, stage: 'str', cand_id: 'str', seed: 'int') -> 'bool'`
- `put(self, stage: 'str', cand_id: 'str', seed: 'int', result: 'dict[str, Any]') -> 'None'`
- `save(self) -> 'None'`
- `set_stage(self, stage: 'str', info: 'dict[str, Any]') -> 'None'`
- `stage_results(self, stage: 'str') -> 'dict[str, list[dict[str, Any]]]'`
