from nssc.evaluation.aggregate import (
    bootstrap_ci,
    format_markdown,
    group_runs,
    mean_std,
    paired_test,
    pareto_front,
    summary_table,
)


def _rec(ds, m, seed, v, status="completed"):
    return {"status": status, "tags": ["suite:s", f"ds:{ds}", f"m:{m}"], "seed": seed,
            "metrics": {"test/recursive/nrmse@50": v}, "param_count": 10}


def test_group_and_summary():
    recs = [_rec("a", "x", 0, 1.0), _rec("a", "x", 1, 3.0), _rec("a", "x", 1, 2.0),  # latest seed wins
            _rec("a", "y", 0, 5.0, "failed"), _rec("b", "x", 0, 1.0)]
    g = group_runs(recs, suite="s")
    assert set(g) == {("a", "x"), ("b", "x")} and len(g[("a", "x")]) == 2
    rows = summary_table(g, ["test/recursive/nrmse@50"])
    r = [r for r in rows if r["dataset"] == "a"][0]
    assert r["test/recursive/nrmse@50"] == 1.5 and r["n_seeds"] == 2
    md = format_markdown(rows, ["test/recursive/nrmse@50"])
    assert "1.5000 ±" in md


def test_stats_helpers():
    assert mean_std([1, 2, 3])[:2] == (2.0, 1.0)
    lo, hi = bootstrap_ci([1, 2, 3, 4, 5])
    assert lo < 3 < hi
    p = paired_test([1, 2, 3, 4, 5], [2, 3, 4, 5, 6])
    assert p["n"] == 5 and p["mean_diff"] == -1.0 and p["t_p"] < 0.01
    assert pareto_front([(1, 5), (2, 2), (5, 1), (3, 3), (6, 6)]) == [True, True, True, False, False]
