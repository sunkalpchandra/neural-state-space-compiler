"""Entry point used by ``nssc visualize``."""

from __future__ import annotations

from pathlib import Path

from nssc.visualization.figures import figures_for_compile, figures_for_experiment


def visualize_experiment(experiment: str | None = None, compile_dir: str | None = None,
                         output: str = "results/figures", registry_path: str | None = None
                         ) -> list[Path]:
    """Dispatch: ``--compile-dir`` → compiler figure set (into ``output/<compile name>/``);
    ``--experiment`` (id or run dir) → per-run figure set (into ``output/<experiment>/``)."""
    if compile_dir:
        cd = Path(compile_dir)
        return figures_for_compile(cd, Path(output) / f"compile_{cd.name}")
    if experiment:
        name = Path(experiment).name if Path(experiment).exists() else experiment
        return figures_for_experiment(experiment, Path(output) / name, registry_path=registry_path)
    raise ValueError("visualize: give --experiment EXP-xxxx (or a run directory) or --compile-dir")
