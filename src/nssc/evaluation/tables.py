"""Generate benchmark tables (markdown + json) from the experiment registry."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from nssc.evaluation.aggregate import format_markdown, load_groups, paired_test, summary_table
from nssc.utils.io import save_json

DEFAULT_METRICS = ["test/recursive/nrmse@1", "test/recursive/nrmse@10", "test/recursive/nrmse@50",
                   "test/recursive/nrmse@100", "test/recursive/nrmse@250", "test/recursive/nrmse_mean",
                   "test/recursive/divergence_time", "test/teacher_forced/nrmse", "test/params/total"]
LABELS = {"test/recursive/nrmse@1": "NRMSE@1", "test/recursive/nrmse@10": "NRMSE@10",
          "test/recursive/nrmse@50": "NRMSE@50", "test/recursive/nrmse@100": "NRMSE@100",
          "test/recursive/nrmse@250": "NRMSE@250", "test/recursive/nrmse_mean": "NRMSE mean",
          "test/recursive/divergence_time": "div. time", "test/teacher_forced/nrmse": "TF NRMSE",
          "test/params/total": "params"}


def suite_tables(suite: str, registry_path: str = "results/registry.jsonl",
                 out_dir: str | Path = "results/tables", metrics: list[str] | None = None,
                 reference_model: str | None = None) -> dict[str, Any]:
    metrics = metrics or DEFAULT_METRICS
    groups = load_groups(registry_path, suite=suite)
    rows = summary_table(groups, metrics)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    md = [f"# Suite `{suite}` — mean ± std over seeds (test split)", "",
          format_markdown(rows, metrics, LABELS), ""]
    tests: dict[str, Any] = {}
    if reference_model:
        md += [f"## Paired comparisons vs `{reference_model}` (NRMSE@50, same seeds)", "",
               "| dataset | model | n | mean diff (ref − model) | paired t p | Wilcoxon p |", "|---|---|---|---|---|---|"]
        for (ds, m), recs in sorted(groups.items()):
            ref = groups.get((ds, reference_model))
            if not ref or m == reference_model:
                continue
            by_seed_ref = {r["seed"]: r["metrics"].get("test/recursive/nrmse@50") for r in ref}
            by_seed_m = {r["seed"]: r["metrics"].get("test/recursive/nrmse@50") for r in recs}
            seeds = sorted(set(by_seed_ref) & set(by_seed_m))
            if len(seeds) < 2:
                continue
            t = paired_test([by_seed_ref[s] for s in seeds], [by_seed_m[s] for s in seeds])
            tests[f"{ds}/{m}"] = t
            md.append(f"| {ds} | {m} | {t['n']} | {t['mean_diff']:+.4f} | {t.get('t_p', float('nan')):.3f} | "
                      f"{t.get('wilcoxon_p', float('nan')):.3f} |")
    (out / f"{suite}.md").write_text("\n".join(md))
    save_json({"rows": rows, "paired_tests": tests, "metrics": metrics}, out / f"{suite}.json")
    return {"rows": rows, "markdown": "\n".join(md), "paired_tests": tests}
