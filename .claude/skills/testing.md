# Skill: testing

## Purpose
Test strategy for `nssc`: unit tests per subsystem, an integration test that runs the
whole pipeline on a tiny system, and regression tests that pin known metric values.
`pytest -q -m "not slow"` must pass before every push (CI runs it on 3.10 and 3.12,
CPU only).

## Relevant theory
- Numerical code fails silently: NaNs, wrong shapes broadcast, gradients that are
  zero. Tests must check finiteness, shape, gradient flow, determinism, and round-trip,
  not just "no exception".
- Regression tests protect research results: a refactor that shifts Lorenz-63 rollout
  NRMSE by 20% is a bug (or a protocol change that must be declared).
- Tests are the executable form of the interfaces in `docs/architecture.md`; a component
  that passes the interface test-suite is compiler-compatible.

## Project-specific conventions
- Layout: `tests/unit/test_<package>_<module>.py`, `tests/integration/test_pipeline.py`,
  `tests/regression/test_known_values.py` (+ `tests/regression/values.yaml`).
- Markers: `@pytest.mark.slow` (> 20 s), `@pytest.mark.mps`, `@pytest.mark.cuda`
  (skipped when device unavailable). Default run excludes `slow`.
- Shared fixtures in `tests/conftest.py`: `tiny_dataset` (harmonic oscillator, D=4 via
  linear lift, 12 trajectories, T=200), `tiny_config`, `tmp_results_dir` (patches
  registry path), `all_encoders`/`all_dynamics` (parametrized over the registry).
- Interface conformance suite (`tests/unit/test_interfaces.py`) is parametrized over
  every registered encoder/decoder/dynamics: shapes `(B,T,D)→(B,T,d)→(B,T,D)`,
  `step`, `rollout` shape `(B,H,d)`, `jacobian` shape `(B,d,d)` and finite, forward and
  backward finite, `n_params()` equals sum of parameter numel, save/load round-trip
  identical outputs, CPU determinism under `seed_everything`.
- Data tests: generator invariants (energy, V, order parameter, attractor bounds), RK4 vs
  scipy, split disjointness/coverage, no leakage between seed streams, observation-map
  determinism.
- Compiler tests: profiler outputs on tiny dataset; candidate generator respects config
  bounds; scorer monotone in each term; staged search resumes from a partial state file
  and produces identical final choice; report file is valid markdown/JSON.
- Registry tests: append-only, id monotonic, `failed` status on exception, config hash
  stable under key reordering.
- Integration: `nssc compile --config tests/fixtures/tiny_experiment.yaml --dry-run`
  → registry rows, checkpoints, metrics.json, report, at least one figure — in < 60 s
  on CPU.
- Regression values (`values.yaml`): e.g. `harmonic_pca_linear_nrmse@25: {value: ...,
  tol: 0.02}`, `lorenz63_lyapunov: {value: 0.9, tol: 0.15}`. Updating a value requires
  a `research/decisions.md` entry explaining why.

## Implementation requirements
- Tests are deterministic (seeded), CPU-only by default, and use `tmp_path` — never write
  into `results/` or `data/`.
- Runtime budget: unit suite < 90 s total; integration < 60 s; slow suite unbounded but
  run before tagging a release.
- New registered component ⇒ automatically covered by the interface suite (no manual
  test list); a component may opt out of a specific check only with a documented
  `pytest.skip` reason.
- Every bug fix adds a test reproducing the bug first.
- Coverage is reported (`pytest --cov=nssc`) but no numeric threshold gates CI; missing
  tests for a subsystem block merge by review, not by percent.

## Common failure modes
- Tests that pass because the metric is NaN and `assert loss < 1` is `False`… (`assert
  torch.isfinite(loss)` first).
- Determinism test that compares one forward pass (trivially deterministic) instead of a
  short training run.
- Fixture that generates data with `np.random.seed` globally (leaks into other tests).
- Integration test that mocks the trainer (then it does not test the pipeline).
- Regression tolerances widened to make a failing refactor pass.
- Test discovering MPS and failing on CI: guard with markers.

## Validation checklist
- [ ] Unit tests exist for the changed subsystem; interface suite passes for new
      components.
- [ ] Finite / shape / grad / determinism / round-trip covered.
- [ ] `pytest -q -m "not slow"` green locally on CPU; `ruff check` clean.
- [ ] Integration test still < 60 s.
- [ ] Regression values untouched, or change justified in `research/decisions.md`.
- [ ] No test writes outside `tmp_path`.
