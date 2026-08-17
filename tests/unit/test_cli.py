from pathlib import Path

from typer.testing import CliRunner

from nssc.cli.main import app

ROOT = Path(__file__).resolve().parents[2]
runner = CliRunner()


def test_help_lists_commands():
    r = runner.invoke(app, ["--help"])
    assert r.exit_code == 0
    for cmd in ("profile", "train", "compile", "benchmark", "evaluate", "visualize", "report", "registry",
                "tables", "smoke", "dashboard", "data"):
        assert cmd in r.output


def test_data_list_and_generate(tmp_path):
    r = runner.invoke(app, ["data", "list"])
    assert r.exit_code == 0 and "lorenz63" in r.output
    r = runner.invoke(app, ["data", "generate", "--system", "harmonic", "--output", str(tmp_path),
                            "--set", "n_traj=2", "--set", "n_steps=16"])
    assert r.exit_code == 0, r.output
    assert list(tmp_path.glob("harmonic_*.npz"))


def test_profile_command(tmp_path):
    r = runner.invoke(app, ["profile", "--config", str(ROOT / "configs/datasets/tiny_smoke.yaml"),
                            "--output", str(tmp_path / "p.json")])
    assert r.exit_code == 0, r.output
    assert (tmp_path / "p.json").exists()


def test_train_command_registers(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # registry written under cwd/results
    r = runner.invoke(app, ["train", "--config", str(ROOT / "configs/experiments/smoke.yaml"),
                            "--set", f"dataset._file={ROOT / 'configs/datasets/tiny_smoke.yaml'}",
                            "--device", "cpu", "--output", str(tmp_path / "run")])
    assert r.exit_code == 0, r.output
    assert (tmp_path / "results/registry.jsonl").exists()
    r = runner.invoke(app, ["registry"])
    assert r.exit_code == 0 and "EXP-0001" in r.output
    r = runner.invoke(app, ["report", "--experiment", "EXP-0001"])
    assert r.exit_code == 0 and "EXP-0001" in r.output
