# `nssc.utils`

### `nssc.utils`

Shared utilities: registry, seeding, config, hashing, environment info.

### `nssc.utils.config`

YAML configuration loading with `_base_` inheritance and dotted overrides.

Configs are plain nested dicts wrapped in :class:`Config` (attribute access +
dict semantics). Typed dataclasses in individual subsystems consume sub-trees.

Supported features
------------------
* ``_base_: path/to/other.yaml`` (relative to the file) — deep-merged underneath.
* CLI overrides ``a.b.c=value`` parsed with YAML semantics.
* :func:`stable_hash` of the resolved config for experiment identity.

#### class `Config(...)`

dict with attribute access; nested dicts are wrapped lazily.

- `get_path(self, dotted: 'str', default: 'Any' = None) -> 'Any'`
- `hash(self) -> 'str'`
- `set_path(self, dotted: 'str', value: 'Any') -> 'None'`
- `to_dict(self) -> 'dict'`

#### `deep_merge(base: 'dict', override: 'dict') -> 'dict'`

#### `load_config(path: 'str | Path | None' = None, overrides: 'list[str] | None' = None, base: 'dict | None' = None) -> 'Config'`

#### `load_yaml(path: 'str | Path') -> 'dict'`

#### `parse_override(s: 'str') -> 'tuple[str, Any]'`

#### `save_yaml(cfg: 'dict', path: 'str | Path') -> 'None'`

### `nssc.utils.env`

Git commit, hardware and library version capture for reproducibility records.

#### `default_device() -> 'torch.device'`

#### `git_commit(repo: 'Path | None' = None) -> 'str'`

#### `hardware_info() -> 'dict[str, Any]'`

### `nssc.utils.experiment_registry`

Experiment registry: append-only JSONL ledger of every run.

Each run receives a monotonically increasing ID ``EXP-0001``. Records are
never deleted; failed runs are kept with ``status='failed'`` (CLAUDE.md rule).

#### class `ExperimentRecord(experiment_id: 'str', git_commit: 'str', config_hash: 'str', dataset: 'str', model: 'str', seed: 'int', status: 'str' = 'running', metrics: 'dict[str, Any]' = <factory>, checkpoint: 'str | None' = None, config: 'dict[str, Any]' = <factory>, param_count: 'int | None' = None, train_time_s: 'float | None' = None, hardware: 'dict[str, Any]' = <factory>, created_at: 'float' = <factory>, updated_at: 'float' = <factory>, tags: 'list[str]' = <factory>, notes: 'str' = '') -> None`

ExperimentRecord(experiment_id: 'str', git_commit: 'str', config_hash: 'str', dataset: 'str', model: 'str', seed: 'int', status: 'str' = 'running', metrics: 'dict[str, Any]' = <factory>, checkpoint: 'str | None' = None, config: 'dict[str, Any]' = <factory>, param_count: 'int | None' = None, train_time_s: 'float | None' = None, hardware: 'dict[str, Any]' = <factory>, created_at: 'float' = <factory>, updated_at: 'float' = <factory>, tags: 'list[str]' = <factory>, notes: 'str' = '')

- `to_dict(self) -> 'dict[str, Any]'`

#### class `ExperimentRegistry(path: 'str | Path' = PosixPath('results/registry.jsonl')) -> 'None'`

- `complete(self, rec: 'ExperimentRecord', metrics: 'dict[str, Any]', **fields: 'Any') -> 'ExperimentRecord'`
- `fail(self, rec: 'ExperimentRecord', error: 'str') -> 'ExperimentRecord'`
- `find(self, **filters: 'Any') -> 'list[dict[str, Any]]'`
- `find_by_hash(self, config_hash: 'str', seed: 'int | None' = None) -> 'list[dict[str, Any]]'`
- `get(self, experiment_id: 'str') -> 'dict[str, Any] | None'`
- `next_id(self) -> 'str'`
- `records(self) -> 'list[dict[str, Any]]'` — Latest record per experiment_id (later lines override earlier).
- `register(self, *, config: 'dict[str, Any]', config_hash: 'str', dataset: 'str', model: 'str', seed: 'int', tags: 'list[str] | None' = None, notes: 'str' = '') -> 'ExperimentRecord'`
- `update(self, rec: 'ExperimentRecord', **fields: 'Any') -> 'ExperimentRecord'`

### `nssc.utils.hashing`

Stable hashing of configurations and arrays.

#### `stable_hash(obj: 'Any', length: 'int' = 12) -> 'str'`

SHA-256 of the canonical JSON serialisation of ``obj``, truncated.

### `nssc.utils.io`

JSON/JSONL helpers with numpy/torch-safe encoding.

#### class `NumpyEncoder(*, skipkeys=False, ensure_ascii=True, check_circular=True, allow_nan=True, sort_keys=False, indent=None, separators=None, default=None)`

Extensible JSON <https://json.org> encoder for Python data structures.

Supports the following objects and types by default:

+-------------------+---------------+
| Python            | JSON          |
+===================+===============+
| dict              | object        |
+-------------------+---------------+
| list, tuple       | array         |
+-------------------+---------------+
| str               | string        |
+-------------------+---------------+
| int, float        | number        |
+-------------------+---------------+
| True              | true          |
+-------------------+---------------+
| False             | false         |
+-------------------+---------------+
| None              | null          |
+-------------------+---------------+

To extend this to recognize other objects, subclass and implement a
``.default()`` method with another method that returns a serializable
object for ``o`` if possible, otherwise it should call the superclass
implementation (to raise ``TypeError``).

- `default(self, o: 'Any') -> 'Any'` — Implement this method in a subclass such that it returns

#### `append_jsonl(obj: 'Any', path: 'str | Path') -> 'None'`

#### `load_json(path: 'str | Path') -> 'Any'`

#### `read_jsonl(path: 'str | Path') -> 'list[Any]'`

#### `save_json(obj: 'Any', path: 'str | Path', indent: 'int' = 2) -> 'None'`

### `nssc.utils.registry`

Lightweight name → class registry.

Every model component (encoder, decoder, dynamics, dataset, metric) registers
itself under a string key so the compiler can enumerate candidates without
hard-coded imports.

Example
-------
>>> DYNAMICS = Registry("dynamics")
>>> @DYNAMICS.register("linear")
... class Linear: ...
>>> DYNAMICS.build("linear", latent_dim=4)

#### class `Registry(name: 'str') -> 'None'`

Abstract base class for generic types.

On Python 3.12 and newer, generic classes implicitly inherit from
Generic when they declare a parameter list after the class's name::

    class Mapping[KT, VT]:
        def __getitem__(self, key: KT) -> VT:
            ...
        # Etc.

On older versions of Python, however, generic classes have to
explicitly inherit from Generic.

After a class has been declared to be generic, it can then be used as
follows::

    def lookup_name[KT, VT](mapping: Mapping[KT, VT], key: KT, default: VT) -> VT:
        try:
            return mapping[key]
        except KeyError:
            return default

- `build(self, key: 'str', **kwargs: 'Any') -> 'T'`
- `get(self, key: 'str') -> 'type[T]'`
- `keys(self) -> 'list[str]'`
- `register(self, key: 'str | None' = None) -> 'Callable[[type[T]], type[T]]'`

### `nssc.utils.seeding`

Deterministic seeding across python, numpy and torch.

#### `rng(seed: 'int') -> 'np.random.Generator'`

Isolated numpy generator (preferred over global state inside generators).

#### `seed_everything(seed: 'int', deterministic: 'bool' = True) -> 'None'`

Seed python, numpy, torch (cpu + all accelerators).

``deterministic=True`` also asks torch to use deterministic kernels where
available; some ops may fall back or warn on MPS.
