# Interactive explorer (`nssc dashboard`)

FastAPI backend (`dashboard/app.py`) + a single self-contained Plotly.js page
(`dashboard/static/`). No build step. Install extras: `pip install -e ".[dashboard]"`.

    nssc dashboard --port 8050            # http://127.0.0.1:8050
    python -m dashboard.app --port 8050   # equivalent

## Sources
The sidebar lists (a) completed registry runs whose checkpoint exists on disk
(`results/registry.jsonl`, plus `results/registry_smoke.jsonl`) and (b) compile runs
(`results/compile/*/compile_report.json`). Loading a source loads the checkpoint and the
**test split** of its dataset (normalised with the run's own training statistics).

## Panels
| # | panel | backend route |
|---|-------|---------------|
| 1 | raw signal (≤ 8 dims shown) | `GET /api/trajectory` |
| 2 | learned latent state z₁..z_d | `GET /api/trajectory` |
| 3 | latent phase portrait (2-D / 3-D, coloured by time) | `GET /api/trajectory` |
| 4 | true vs predicted rollout, adjustable context/horizon, ±2σ envelope (method labelled) | `GET /api/trajectory` |
| 5 | dynamics field F(z) − z on a 2-D latent plane + trajectory overlay | `GET /api/field` |
| 6 | local Jacobian eigenvalues on the unit circle + spectral-radius histogram | `GET /api/stability` |
| 7 | compiler decision: ranking table, stacked score terms, reasons, stage funnel | `GET /api/compile_report` |
| 8 | counterfactual rollout from an edited z₀ | `POST /api/counterfactual` |

## Design
Dark instrument-style UI: monospace/system type, dense grid, no gradients, no marketing
cards. All numbers come from the loaded checkpoint and its registry record — nothing is
precomputed or hand-edited.
