# Compiler Engineer

## Responsibility
The compiler itself: dataset profiling, candidate generation, staged resumable search,
evaluation hooks, stability analysis integration, multi-objective scoring, compile
report, `StateSpaceCompiler`, and the `nssc` CLI. Also the registry/config utilities.

## Owns
- `src/nssc/compiler/` (`profiler.py`, `candidates.py`, `scorer.py`, `report.py`,
  `compiler.py`, `model.py`)
- `src/nssc/search/` (`staged.py`, `state.py`)
- `src/nssc/utils/` (`registry.py`, `config.py`, `seeding.py`, `hashing.py`, `gitinfo.py`,
  `hardware.py`) — schema decisions with `systems_architect`
- `src/nssc/cli/`
- `configs/compiler/*.yaml`, `docs/compiler.md`

## Interfaces
- ← `systems_architect`: interface contracts and registry mechanism.
- ← `dynamics_researcher` / `representation_researcher`: registered components; the
  compiler enumerates them without knowing their names.
- ← `benchmark_engineer`: metric functions used in scoring (`nssc.metrics`).
- → `mlops_engineer`: checkpoint dirs, metadata, resumable state file.
- → `documentation_engineer`: report template wording.
- → `testing_engineer`: stage tests, resumability test, integration test.

## Review questions it must ask
- Can I add a new dynamics family without touching `nssc/compiler`? (Test: register a
  dummy family in a test module and see it appear as a candidate.)
- Does any stage read the test split? (`grep`.)
- Are score terms normalized on validation statistics and logged individually?
- Does the coarse screen discard candidates that would have won at long horizon?
  (Cell L measures this; the screen thresholds are config.)
- If the process is killed mid-search, does `--resume` reproduce the same final choice?
- Does the report list every candidate and the reason for elimination?
- Are score weights, stage budgets, thresholds all in `configs/compiler/*.yaml`?

## Definition of done
- `nssc compile --config X --dry-run` runs the tiny pipeline in < 60 s on CPU (integration
  test) and produces registry rows, checkpoints, `compile_report.{md,json}`, metrics.
- Unit tests per stage; resumability test; registry append-only + id monotonic tests.
- `docs/compiler.md` matches the code (stage names, score formula, report fields).
- No per-family special-casing in the compiler (reviewer greps for it).
