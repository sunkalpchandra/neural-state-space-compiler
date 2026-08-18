from nssc.evaluation.pareto import dominated_area, pareto_markdown, pareto_points


def _rec(ds, m, seed, v, params, baseline=False):
    return {"status": "completed", "tags": ["suite:s", f"ds:{ds}", f"m:{m}"] + (["baseline"] if baseline else []),
            "seed": seed, "metrics": {"test/recursive/nrmse@50": v}, "param_count": params}


def test_pareto_points_and_area():
    recs = [_rec("a", "persistence", 0, 1.0, 0, True), _rec("a", "small", 0, 0.5, 10),
            _rec("a", "big_bad", 0, 0.6, 1000), _rec("a", "big_good", 0, 0.1, 1000)]
    per = pareto_points(recs, "s")
    pts = {p["model"]: p for p in per["a"]}
    assert pts["persistence"]["pareto"] and pts["small"]["pareto"] and pts["big_good"]["pareto"]
    assert not pts["big_bad"]["pareto"]
    area = dominated_area(per["a"])
    assert area > 0
    md = pareto_markdown(per, "test/recursive/nrmse@50")
    assert "big_good" in md and "**yes**" in md
