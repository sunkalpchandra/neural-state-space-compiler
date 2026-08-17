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
