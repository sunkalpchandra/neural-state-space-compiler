def test_experiment_registry_lifecycle(tmp_registry):
    r = tmp_registry
    assert r.next_id() == "EXP-0001"
    rec = r.register(config={"a": 1}, config_hash="h1", dataset="d", model="m", seed=0)
    assert rec.experiment_id == "EXP-0001" and rec.status == "running"
    r.complete(rec, metrics={"mse": 0.1}, param_count=10)
    got = r.get("EXP-0001")
    assert got["status"] == "completed" and got["metrics"]["mse"] == 0.1
    rec2 = r.register(config={}, config_hash="h2", dataset="d", model="m", seed=1)
    r.fail(rec2, "boom")
    assert rec2.experiment_id == "EXP-0002"
    assert r.get("EXP-0002")["status"] == "failed"
    assert len(r.records()) == 2  # failed runs are never deleted
    assert len(r.find_by_hash("h1")) == 1


def test_concurrent_register_never_collides(tmp_path):
    """Regression (F-005): parallel processes must not be handed the same EXP id."""
    import subprocess
    import sys

    path = tmp_path / "reg.jsonl"
    code = (
        "import sys;from nssc.utils.experiment_registry import ExperimentRegistry;"
        "r=ExperimentRegistry(sys.argv[1]);"
        "[r.register(config={},config_hash='h',dataset='d',model='m',seed=i) for i in range(8)]"
    )
    procs = [subprocess.Popen([sys.executable, "-c", code, str(path)]) for _ in range(4)]
    for p in procs:
        assert p.wait() == 0
    import json

    ids = [json.loads(x)["experiment_id"] for x in path.read_text().splitlines() if x.strip()]
    assert len(ids) == 32 and len(set(ids)) == 32


def test_register_records_device(tmp_path):
    from nssc.utils.experiment_registry import ExperimentRegistry

    r = ExperimentRegistry(tmp_path / "r.jsonl")
    rec = r.register(config={}, config_hash="h", dataset="d", model="m", seed=0, device="cpu")
    assert rec.hardware["device"] == "cpu"
