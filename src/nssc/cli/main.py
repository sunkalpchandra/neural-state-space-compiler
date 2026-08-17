"""nssc CLI.

    nssc profile   --config configs/datasets/lorenz63.yaml
    nssc train     --config configs/experiments/lorenz63_mlp.yaml [--seed 0] [--set a.b=c]
    nssc evaluate  --experiment EXP-0001
    nssc compile   --config configs/compiler/lorenz63.yaml --output results/compile/lorenz63
    nssc benchmark --suite synthetic
    nssc visualize --experiment EXP-0001
    nssc report    --experiment EXP-0001
    nssc registry  [--status completed] [--tag smoke]
    nssc smoke
    nssc dashboard
"""

from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(no_args_is_help=True, add_completion=False, help=__doc__)
data_app = typer.Typer(no_args_is_help=True, help="Dataset generation utilities.")
app.add_typer(data_app, name="data")
console = Console()

ROOT = Path(__file__).resolve().parents[3]


def _cfg(config: str | None, sets: list[str] | None, base: dict | None = None):
    from nssc.utils.config import load_config

    return load_config(config, overrides=sets or [], base=base)


@app.command()
def profile(config: str = typer.Option(..., "--config", "-c", help="dataset yaml"),
            output: str | None = typer.Option(None, "--output", "-o")):
    """Profile a dataset (dimensionality, intrinsic dim, autocorrelation, noise, stationarity)."""
    from nssc.compiler.profiler import profile_dataset
    from nssc.data.builder import build_dataset
    from nssc.utils.io import save_json

    ds = build_dataset(_cfg(config, None).to_dict())
    prof = profile_dataset(ds)
    console.print_json(json.dumps(prof.to_dict(), default=str))
    if output:
        save_json(prof.to_dict(), output)


@app.command()
def train(config: str = typer.Option(..., "--config", "-c"),
          seed: int | None = typer.Option(None, "--seed"),
          set_: list[str] = typer.Option(None, "--set", "-s", help="dotted overrides a.b=c"),
          output: str | None = typer.Option(None, "--output", "-o"),
          device: str | None = typer.Option(None, "--device")):
    """Train + evaluate one run config; registers it in results/registry.jsonl."""
    import torch

    from nssc.experiment import run_experiment

    cfg = _cfg(config, set_)
    if seed is not None:
        cfg["seed"] = seed
    if output:
        cfg["output_dir"] = output
    res = run_experiment(cfg, device=torch.device(device) if device else None,
                         log=lambda s: console.print(s))
    console.rule(f"{res['experiment_id']} {res['status']}")
    if res["status"] == "completed":
        _print_kv(res["summary"])
    else:
        console.print(res.get("error"))
        raise typer.Exit(1)


@app.command()
def evaluate(experiment: str = typer.Option(..., "--experiment", "-e"),
             split: str = typer.Option("test"), context: int | None = typer.Option(None),
             device: str | None = typer.Option(None, "--device")):
    """Re-evaluate a registered experiment from its checkpoint."""
    import numpy as np
    import torch

    from nssc.evaluation import EvalConfig, evaluate_model
    from nssc.experiment import prepare_data
    from nssc.training import load_checkpoint
    from nssc.utils.experiment_registry import ExperimentRegistry

    rec = ExperimentRegistry().get(experiment)
    if rec is None or not rec.get("checkpoint"):
        console.print(f"[red]no checkpoint for {experiment}")
        raise typer.Exit(1)
    model, _ = load_checkpoint(rec["checkpoint"])
    splits, _, raw = prepare_data(rec["config"]["dataset"])
    ecfg_d = dict(rec["config"].get("eval", {}))
    if context is not None:
        ecfg_d["context"] = context
    ecfg = EvalConfig(**{k: v for k, v in ecfg_d.items() if k in EvalConfig.__dataclass_fields__})
    m = evaluate_model(model, torch.from_numpy(splits[split].x), ecfg, sigma=np.ones(raw.obs_dim),
                       device=torch.device(device) if device else None)
    _print_kv({k: v for k, v in m.items() if not isinstance(v, dict)})


@app.command()
def registry(status: str | None = typer.Option(None), tag: str | None = typer.Option(None),
             limit: int = typer.Option(50)):
    """List registered experiments."""
    from nssc.utils.experiment_registry import ExperimentRegistry

    recs = ExperimentRegistry().records()
    if status:
        recs = [r for r in recs if r["status"] == status]
    if tag:
        recs = [r for r in recs if tag in r.get("tags", [])]
    t = Table("id", "status", "dataset", "model", "seed", "params", "test nrmse@50", "commit")
    for r in recs[-limit:]:
        m = r.get("metrics", {})
        v = m.get("test/recursive/nrmse@50", m.get("test/recursive/nrmse_mean"))
        t.add_row(r["experiment_id"], r["status"], r["dataset"], r["model"], str(r["seed"]),
                  str(r.get("param_count")), f"{v:.4f}" if isinstance(v, (int, float)) else "-",
                  str(r["git_commit"])[:8])
    console.print(t)


@app.command()
def compile(config: str = typer.Option(..., "--config", "-c"),  # noqa: A001
            output: str | None = typer.Option(None, "--output", "-o"),
            set_: list[str] = typer.Option(None, "--set", "-s"),
            resume: bool = typer.Option(True, help="resume a partially completed search"),
            device: str | None = typer.Option(None, "--device")):
    """Run the state-space compiler: profile → candidates → staged search → select → report."""
    import torch

    from nssc.compiler import StateSpaceCompiler

    cfg = _cfg(config, set_)
    if output:
        cfg["output_dir"] = output
    comp = StateSpaceCompiler(cfg, device=torch.device(device) if device else None,
                              log=lambda s: console.print(s))
    compiled = comp.run(resume=resume)
    console.rule("compiled")
    console.print(compiled.report.to_markdown())


@app.command()
def benchmark(suite: str = typer.Option("synthetic", "--suite"),
              set_: list[str] = typer.Option(None, "--set", "-s"),
              device: str | None = typer.Option(None, "--device")):
    """Run a benchmark suite defined in configs/experiments/benchmarks/<suite>.yaml."""
    from nssc.search.runner import run_suite

    run_suite(ROOT / f"configs/experiments/benchmarks/{suite}.yaml", overrides=set_ or [],
              device=device, log=lambda s: console.print(s))


@app.command()
def visualize(experiment: str | None = typer.Option(None, "--experiment", "-e"),
              compile_dir: str | None = typer.Option(None, "--compile-dir"),
              output: str = typer.Option("results/figures", "--output", "-o")):
    """Generate figures for an experiment or a compile run."""
    from nssc.visualization.cli_hooks import visualize_experiment

    paths = visualize_experiment(experiment=experiment, compile_dir=compile_dir, output=output)
    for p in paths:
        console.print(f"wrote {p}")


@app.command()
def report(experiment: str | None = typer.Option(None, "--experiment", "-e"),
           compile_dir: str | None = typer.Option(None, "--compile-dir")):
    """Print a human-readable report for an experiment or a compile run."""
    from nssc.compiler.report import report_for

    console.print(report_for(experiment=experiment, compile_dir=compile_dir))


@app.command()
def smoke():
    """Run the tiny end-to-end smoke experiment (seconds)."""
    import torch

    from nssc.experiment import run_experiment
    from nssc.utils.experiment_registry import ExperimentRegistry

    cfg = _cfg(str(ROOT / "configs/experiments/smoke.yaml"), None)
    cfg["dataset"]["_file"] = str(ROOT / cfg["dataset"]["_file"])
    cfg["output_dir"] = "results/raw/smoke"
    res = run_experiment(cfg, registry=ExperimentRegistry("results/registry_smoke.jsonl"),
                         device=torch.device("cpu"), log=lambda s: console.print(s))
    console.print(res["status"])


@app.command()
def dashboard(port: int = typer.Option(8050), host: str = typer.Option("127.0.0.1")):
    """Launch the interactive dynamical-system explorer."""
    from dashboard.app import serve  # type: ignore

    serve(host=host, port=port)


@data_app.command("generate")
def data_generate(system: str = typer.Option(..., "--system"),
                  config: str | None = typer.Option(None, "--config", "-c"),
                  output: str = typer.Option("data/cache", "--output", "-o"),
                  set_: list[str] = typer.Option(None, "--set", "-s")):
    """Generate and cache a synthetic dataset as .npz."""
    from nssc.data.builder import build_dataset

    path = config or str(ROOT / f"configs/datasets/{system}.yaml")
    ds = build_dataset(_cfg(path, set_).to_dict())
    out = ds.save(Path(output) / f"{system}_{ds.metadata['version']}.npz")
    console.print(f"{ds.n_traj}×{ds.n_steps}×{ds.obs_dim} → {out}")


@data_app.command("list")
def data_list():
    import nssc.data  # noqa: F401
    from nssc.utils.registry import SYSTEMS

    for k in SYSTEMS.keys():
        console.print(k)


def _print_kv(d: dict) -> None:
    t = Table("metric", "value")
    for k, v in d.items():
        t.add_row(k, f"{v:.5g}" if isinstance(v, float) else str(v))
    console.print(t)


if __name__ == "__main__":
    app()
