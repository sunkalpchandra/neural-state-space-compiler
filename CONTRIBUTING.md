# Contributing

1. Read `CLAUDE.md` (project constitution) and `docs/architecture.md`.
2. `pip install -e ".[dev]"` then `pytest -q -m "not slow"` and `ruff check src tests scripts` must pass.
3. New encoders/dynamics/systems/baselines register themselves (`nssc.utils.registry`) and ship
   with unit tests (shape, finiteness, gradient flow, Jacobian vs finite differences for dynamics).
4. Any change to an experimental setting is a config change, never a silent code default change.
5. Never edit files under `results/` by hand; regenerate with `scripts/generate_report.py`.
6. Commit small and often with conventional prefixes (`feat:`, `fix:`, `test:`, `exp:`, `docs:`).
7. Failed experiments stay in the registry and get an entry in `research/failures.md`.
