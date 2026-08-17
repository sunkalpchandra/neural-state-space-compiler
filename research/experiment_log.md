# Experiment log

Chronological. One entry per experiment (or per meaningful partial result). Facts and
EXP ids only; interpretation is labeled as such. Never delete entries; superseded entries
get a `Superseded by:` line.

## Entry template

```
### YYYY-MM-DD — EXP-xxxx[, EXP-yyyy] — <short title>
- Hypothesis / cell: Hn / <A–L>
- Config: configs/experiments/<name>.yaml (config_hash <hash>), dataset <name> (<dataset_hash>)
- Git commit: <sha> (clean)
- Seeds: 0–4 (or: preliminary, seeds 0–1)
- Hardware / wall-clock: <device>, <h:mm> total
- Setup: what varied, what was held fixed, budgets, modes, horizons, context length
- Result (facts): key numbers as mean ± std (n=5), CI where used, mode + horizon labeled;
  diverged fraction; table/figure ids (results/tables/..., F3)
- Interpretation: supported / not supported / mixed / inconclusive — and why, in one paragraph
- Hypothesis status change: Hn: <old> → <new> (or none)
- Follow-ups / open questions added: <ids in open_questions.md>
- Reviewer: <accept | accept with changes | reject> — <one line>
- Failures during this experiment: <link to failures.md entry or none>
```

---

_No entries yet._
