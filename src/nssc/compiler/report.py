"""Human- and machine-readable compile report.

The ``reasons`` list is generated from actual comparisons within the final
candidate pool (never templated numbers).
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from nssc.utils.io import load_json, save_json


@dataclass
class CompileReport:
    selected: dict[str, Any]
    selected_metrics: dict[str, Any]
    ranking: list[dict[str, Any]]
    stage_summaries: list[dict[str, Any]]
    profile: dict[str, Any]
    weights: dict[str, Any]
    reasons: list[str] = field(default_factory=list)
    n_runs: int = 0
    n_failed: int = 0
    wall_time_s: float = 0.0
    dataset: dict[str, Any] = field(default_factory=dict)
    checkpoint: str | None = None
    rollout_key: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save(self, path: str | Path) -> None:
        save_json(self.to_dict(), path)
        Path(path).with_suffix(".md").write_text(self.to_markdown())

    @classmethod
    def load(cls, path: str | Path) -> CompileReport:
        return cls(**load_json(path))

    def to_markdown(self) -> str:
        s = self.selected
        m = self.selected_metrics
        obs = (self.dataset.get("observation") or {}).get("type", "identity")
        dsname = self.dataset.get("name") or (
            f"{self.dataset.get('system', 'dataset')}"
            + (f" [{obs} obs → D={self.dataset['observation'].get('obs_dim')}]" if obs != "identity" else ""))
        L = [f"# Compile report — {dsname}", ""]
        L += [f"**Selected latent dimension:** {s['latent_dim']}",
              f"**Selected representation:** `{s['encoder']}` (decoder `{s['decoder']}`)",
              f"**Selected dynamics:** `{s['dynamics']}`",
              f"**Parameters:** {int(m.get('val/params/total', float('nan')))} trainable, "
              f"{int(m.get('val/params/total_stored', m.get('val/params/total', float('nan'))))} stored "
              f"(the complexity term uses *stored*: parameters + buffers)",
              f"**Runs:** {self.n_runs} ({self.n_failed} failed) in {self.wall_time_s / 60:.1f} min", ""]
        L += ["## Reason", ""] + [f"- {r}" for r in self.reasons] + [""]
        L += ["## Selection weights", "", "```", str(self.weights), "```", ""]
        L += ["## Final ranking", "",
              "All metrics below are **validation** (selection never sees the test split); they are means "
              "over the final stage's seeds with the spread across seeds in parentheses.", "",
              "| rank | candidate | J | val rollout (mean ± std) | val 1-step | val recon | params (stored) | ρ_max | verdict (worst seed) |",
              "|---|---|---|---|---|---|---|---|---|"]
        for r in self.ranking:
            a = r["agg"]
            nun = int(a.get("val/stability/n_unstable_seeds", 0) or 0)
            L.append(f"| {r['rank']} | {r['name']} | {r['score']:.3f} | "
                     f"{a.get(self.rollout_key, float('nan')):.4f} ± "
                     f"{a.get(self.rollout_key + '__std', 0.0):.4f} | "
                     f"{a.get('val/teacher_forced/nrmse', float('nan')):.4f} | "
                     f"{a.get('val/recon/nrmse', float('nan')):.4f} | "
                     f"{int(a.get('val/params/total', 0))} "
                     f"({int(a.get('val/params/total_stored', a.get('val/params/total', 0)))}) | "
                     f"{a.get('val/stability/rho_max', float('nan')):.3f} | "
                     f"{a.get('val/stability/verdict', '?')}"
                     + (f" ({nun}/{int(a.get('n_seeds', 0))} unstable)" if nun else "") + " |")
        L.append("")
        for st in self.stage_summaries:
            L.append(f"- stage `{st['stage']}`: {st['n_candidates']} candidates → "
                     f"{st['n_survivors']} survivors")
        if self.profile:
            rec = self.profile.get("recommendations", {})
            L += ["", "## Dataset profile (excerpt)", "",
                  f"- obs_dim={self.profile.get('obs_dim')}, n_traj={self.profile.get('n_traj')}, "
                  f"n_steps={self.profile.get('n_steps')}",
                  f"- suggested latent dims: {self.profile.get('suggested_latent_dims')}",
                  f"- hints: {rec}"]
        return "\n".join(L)


def build_reasons(rows: list[dict[str, Any]], selected: dict[str, Any], rollout_key: str
                  ) -> list[str]:
    """Derive plain-language justifications from the final ranking."""
    if not rows:
        return ["no completed candidates"]
    top = rows[0]
    a = top["agg"]
    reasons: list[str] = []
    horizon = rollout_key.split("@")[-1] if "@" in rollout_key else "mean"
    sd = rollout_key + "__std"
    r_top = a.get(rollout_key, float("nan"))
    # vs best linear candidate
    linear = [r for r in rows if r["name"].split("+")[1].split("@")[0] in ("linear", "affine")
              and r is not top]
    if linear:
        best_lin = min(linear, key=lambda r: r["agg"].get(rollout_key, float("inf")))
        v = best_lin["agg"].get(rollout_key, float("nan"))
        if math.isfinite(v) and math.isfinite(r_top) and r_top > 0:
            ratio = v / r_top
            word = "lower" if ratio >= 1 else "higher"
            ratio = ratio if ratio >= 1 else 1 / ratio
            reasons.append(f"{ratio:.2f}× {word} {horizon}-step rollout NRMSE than the best "
                           f"linear-dynamics candidate ({best_lin['name']}: {v:.4f} vs selected {r_top:.4f})")
    # vs runner-up
    if len(rows) > 1:
        ru = rows[1]
        v = ru["agg"].get(rollout_key, float("nan"))
        p_top, p_ru = a.get("val/params/total", float("nan")), ru["agg"].get("val/params/total", float("nan"))
        if math.isfinite(v) and math.isfinite(r_top):
            rel = (r_top - v) / max(v, 1e-12) * 100
            n = int(a.get("n_seeds", 0))
            reasons.append(f"selected has {rel:+.1f}% validation {horizon}-step rollout NRMSE vs "
                           f"runner-up {ru['name']} ({r_top:.4f} ± {a.get(sd, 0.0):.4f} vs "
                           f"{v:.4f} ± {ru['agg'].get(sd, 0.0):.4f}, n={n} seeds — "
                           f"spreads overlap when the ± ranges do; no paired test is run here)")
        if math.isfinite(p_top) and math.isfinite(p_ru) and p_ru > 0:
            reasons.append(f"{(p_top / p_ru - 1) * 100:+.0f}% parameter count vs runner-up "
                           f"({int(p_top)} vs {int(p_ru)})")
    # smallest model comparison
    smallest = min(rows, key=lambda r: r["agg"].get("val/params/total", float("inf")))
    if smallest is not top:
        v = smallest["agg"].get(rollout_key, float("nan"))
        p = smallest["agg"].get("val/params/total", float("nan"))
        if math.isfinite(v) and math.isfinite(r_top):
            reasons.append(f"smallest candidate {smallest['name']} ({int(p)} params) has "
                           f"{v / max(r_top, 1e-12):.2f}× the selected model's rollout NRMSE")
    # stability
    rho = a.get("val/stability/rho_max", float("nan"))
    verdict = a.get("val/stability/verdict", "?")
    lam = a.get("val/stability/lyapunov_max", float("nan"))
    nun = int(a.get("val/stability/n_unstable_seeds", 0) or 0)
    reasons.append(f"stability (worst seed): verdict={verdict}, max local spectral radius {rho:.3f}, "
                   f"λ_max≈{lam:.3f}/step, blow-up fraction {a.get('val/stability/frac_blowup', float('nan')):.2f} "
                   f"(max over seeds {a.get('val/stability/frac_blowup_max', float('nan')):.2f}); "
                   f"{nun}/{int(a.get('n_seeds', 0))} seeds not stable")
    # one-step / recon
    reasons.append(f"validation recon NRMSE {a.get('val/recon/nrmse', float('nan')):.4f}, "
                   f"one-step NRMSE {a.get('val/teacher_forced/nrmse', float('nan')):.4f} "
                   f"(position-matched {a.get('val/teacher_forced_ctx/nrmse', float('nan')):.4f})")
    if a.get("n_seeds", 1) > 1:
        reasons.append(f"aggregated over {a['n_seeds']} seeds")
    return reasons


def report_for(experiment: str | None = None, compile_dir: str | None = None) -> str:
    if compile_dir:
        return CompileReport.load(Path(compile_dir) / "compile_report.json").to_markdown()
    from nssc.utils.experiment_registry import ExperimentRegistry

    rec = ExperimentRegistry().get(experiment or "")
    if rec is None:
        return f"unknown experiment {experiment}"
    lines = [f"# {rec['experiment_id']} — {rec['model']} on {rec['dataset']} (seed {rec['seed']})",
             f"status: {rec['status']}  commit: {rec['git_commit'][:10]}  params: {rec.get('param_count')}",
             f"checkpoint: {rec.get('checkpoint')}", "", "| metric | value |", "|---|---|"]
    for k, v in rec.get("metrics", {}).items():
        lines.append(f"| {k} | {v:.5g} |" if isinstance(v, float) else f"| {k} | {v} |")
    return "\n".join(lines)
