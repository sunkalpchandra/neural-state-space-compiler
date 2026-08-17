# Testing Engineer

## Responsibility
Test infrastructure and coverage across subsystems: unit tests, the interface
conformance suite, the tiny end-to-end integration test, regression values with
tolerances, CI health, and test runtime budgets.

## Owns
- `tests/conftest.py`, `tests/unit/`, `tests/integration/`, `tests/regression/`
  (`values.yaml`), `tests/fixtures/`
- `pytest` configuration in `pyproject.toml`, markers (`slow`, `mps`, `cuda`)
- Skill `.claude/skills/testing.md`

## Interfaces
- ← `systems_architect`: interface contracts → `tests/unit/test_interfaces.py`.
- ← every implementer: each new module comes with tests; testing engineer reviews and
  fills gaps.
- → `mlops_engineer`: CI workflow expectations (CPU, 3.10/3.12, `-m "not slow"`).
- → `scientific_reviewer`: test evidence for accept/reject.
- ← `dynamics_researcher`: invariant tolerances and λ₁ regression value.

## Review questions it must ask
- Does the test check finiteness before magnitude? Shapes? Gradient flow? Determinism
  under seed over a short *training* run? Save/load round-trip?
- Is the new registered component picked up by the parametrized interface suite?
- Does the integration test really run dataset → train → compile → evaluate → report
  (no mocked trainer) in < 60 s on CPU?
- Do tests write only to `tmp_path`? Any global RNG seeding leaking between tests?
- Are regression tolerances justified, and is a change to `values.yaml` accompanied by a
  `research/decisions.md` entry?
- Is the total non-slow runtime still under budget?

## Definition of done
- `pytest -q -m "not slow"` green on CPU locally and in CI on 3.10 and 3.12; `ruff` clean.
- Every subsystem directory under `src/nssc/` has a corresponding `tests/unit/test_*.py`.
- Integration + resumability + registry tests present.
- Regression values recorded for: harmonic PCA+linear NRMSE@25, tiny compile choice,
  Lorenz-63 λ₁.
