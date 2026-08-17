# Failures

Failed experiments, broken runs, and process failures are scientific evidence. They are
never deleted from the registry (status `failed` or `invalid`) and each gets an entry
here. Include negative results that were *not* crashes: hypotheses that were falsified
belong in `experiment_log.md`, but any *waste* (wrong config, leakage found, wasted
compute) belongs here so it is not repeated.

## Entry template

```
### YYYY-MM-DD — <EXP-xxxx or "process"> — <short title>
- Category: crash | NaN/divergence | leakage | protocol error | wrong config | infra | tooling | integrity
- What happened: <facts, error message / metric evidence>
- Root cause: <as far as known>
- Cost: <wall-clock, seeds affected>
- Detected by: <test / reviewer / manual inspection / registry audit>
- Fix / prevention: <code change, new test, new checklist item; commit sha>
- Registry status set to: failed | invalid (with note)
- Do not: <the one-line lesson>
```

---

_No entries yet._
