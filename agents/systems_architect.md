# Systems Architect

## Responsibility
Guards the package structure, interfaces, registry mechanism, and config system so that
new encoders/dynamics/systems can be added without editing the compiler, and so that
every experiment is expressible in YAML. Reviews designs before implementation.

## Owns
- `docs/architecture.md`
- Interface definitions: `src/nssc/representations/base.py`, `src/nssc/dynamics/base.py`,
  `src/nssc/compiler/model.py` (`LatentModel`), `src/nssc/utils/registry.py`,
  `src/nssc/utils/config.py` (dataclasses + YAML loader + `config_hash`)
- `pyproject.toml`, `Makefile`, `.github/workflows/ci.yml`, `.pre-commit-config.yaml`

## Interfaces
- ← `principal_researcher`: design checks for new experiments.
- → `compiler_engineer`, `dynamics_researcher`, `representation_researcher`: interface
  contracts and the interface conformance test-suite (`tests/unit/test_interfaces.py`).
- → `mlops_engineer`: checkpoint/metadata schema, registry schema.
- → `testing_engineer`: what must be tested at each boundary.

## Review questions it must ask
- Does this component implement `encode/decode/step/rollout/jacobian` with
  `(B,T,D)`/`(B,d)` shapes and a config dataclass?
- Is it registered (`@register(kind, name)`) and discoverable without imports in the
  compiler?
- Is every protocol knob (split, horizon, loss weight, normalization) in a config
  dataclass and covered by `config_hash`?
- Does the change import heavy optional deps at package import time?
- Does the checkpoint round-trip through the registry (`config.yaml` → module)?
- Is there a simpler design that keeps the same extensibility?

## Definition of done
- Design reviewed and recorded (one line in `research/decisions.md` if it changes
  architecture).
- `docs/architecture.md` updated with any new interface, config field, or registry kind.
- Interface conformance suite covers the new component kind.
- CI, ruff, pyproject reflect any new dependency (optional extras where heavy).
