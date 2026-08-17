# Skill: research-methodology

## Purpose
How research is conducted in `nssc`. This is a hypothesis-driven codebase: code exists to
test the claims in `research/hypotheses.md` (H1–H7), not the other way round. Every
experiment, figure, and README sentence must trace back to a hypothesis, a registered run
(`results/registry.jsonl`), and a config hash.

## Relevant theory
- **Falsifiability**: each hypothesis carries a falsification criterion written *before*
  the experiment runs (see `research/hypotheses.md`). An experiment that cannot fail is not
  an experiment.
- **Pre-registration light**: the config in `configs/experiments/*.yaml` is the
  pre-registration. Horizons, seeds, splits, metrics and selection criterion are fixed
  there before training starts. Changing them afterwards is a new experiment with a new id.
- **Selection vs. reporting**: model/hyperparameter selection uses the validation split
  only; test-split numbers are computed once per (config, seed) and are never used to
  choose anything.
- **Multiple comparisons**: with ~9 baselines × ~10 systems × 8 horizons, some "wins" are
  noise. Report effect sizes with uncertainty (see `statistical-analysis.md`), and prefer
  claims aggregated across systems over cherry-picked cells.
- **Negative results are results**: a hypothesis that is falsified is logged in
  `research/experiment_log.md` and `research/failures.md`, and the README says so.

## Project-specific conventions
- Experiment ids are `EXP-0001`, `EXP-0002`, … allocated by `nssc.utils.registry`
  (monotonic, never reused, never renumbered).
- Each `experiments/**/run_*.py` driver docstring begins with
  `Hypothesis: Hn — <one line>` and `Matrix cell: <letter> (docs/experiments.md)`.
- Log the *decision* to run an experiment in `research/decisions.md` if it changes the
  plan; log the *result* in `research/experiment_log.md`.
- Wording rules: "matches", "is within X of", "we did not observe" — never "outperforms
  existing methods" unless a benchmark in this repo (`experiments/benchmarks/`) shows it
  with seeds 0–4 and CIs that do not overlap.
- Latent dimensions are *statistical* coordinates. Do not call `z_1` "the angle" or "the
  slow variable" without an explicit alignment analysis (e.g. linear regression from
  z to ground-truth state with reported R² on held-out trajectories).
- Compute-cost honesty: report GPU/MPS/CPU device, wall-clock, and parameter count next to
  every accuracy number. A "smaller model" claim needs the numbers.

## Implementation requirements
- `nssc.utils.registry.register_run(...)` is called at the start (status `running`) and
  end (`completed` / `failed`) of every driver. Crashes must still leave a `failed` row.
- Drivers accept `--config`, `--seed`, `--device`, `--dry-run`, `--resume`.
- Every driver writes `results/processed/<EXP-id>/metrics.json` with the metric schema in
  `benchmarking.md`, plus the resolved config (`config.resolved.yaml`) and `config_hash`.
- Never write a metric into a table or figure by hand. `nssc.visualization` and
  `scripts/make_tables.py` read from the registry / processed dirs only.
- Any deviation from protocol discovered mid-run (e.g. a run used a wrong split) →
  mark the run `invalid` in the registry with a `note`, keep the checkpoint, redo.

## Common failure modes
- Selecting the "best seed" and reporting it as the result. (Report all 5.)
- Tuning on test by accident: a driver that evaluates test at every epoch and an
  early-stopping callback that reads that metric. Early stopping reads validation only.
- Silent protocol drift: a code default (e.g. horizon list, normalization) changed without
  a config change, so old registry rows are no longer comparable. Every such knob lives in
  the config dataclass, and `config_hash` covers it.
- Comparing baselines trained with different budgets (epochs, early stopping patience,
  data). Budgets are part of the experiment config and equal across models unless the
  experiment is *about* budgets.
- Interpreting a lower recon loss as "better dynamics". Recon ≠ rollout; H1 is about
  rollout at horizons ≥ 50.
- Deleting failed runs to tidy the registry. Forbidden — status `failed`, keep the row.

## Validation checklist
- [ ] Which hypothesis (H1–H7) does this test? Written in the driver docstring and config.
- [ ] Falsification criterion exists and is stated before running.
- [ ] Split is trajectory-level (parameter-range or subject-level where applicable).
- [ ] Selection uses validation only; test evaluated once at the end.
- [ ] Seeds 0–4 for anything reported; mean ± std (+ CI) in tables.
- [ ] Registry row exists with git commit, config hash, hardware, status.
- [ ] Result logged in `research/experiment_log.md`; failure (if any) in `research/failures.md`.
- [ ] Claims in README/docs use hedged, evidence-linked language and cite an EXP id.
