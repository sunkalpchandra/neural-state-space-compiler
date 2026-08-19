import math

from nssc.compiler.scorer import (
    MultiObjectiveScorer,
    ScoreWeights,
    aggregate_seeds,
    pick_rollout_key,
)


def _run(recon, one, roll, params, blow=0.0, inst=0.0):
    return {"status": "completed", "summary": {"val/recon/nrmse": recon, "val/teacher_forced/nrmse": one,
                                               "val/recursive/nrmse@50": roll, "val/params/total": params,
                                               "val/stability/instability_score": inst,
                                               "val/stability/frac_blowup": blow,
                                               "val/stability/verdict": "stable"}}


def test_exact_zero_reconstruction_does_not_dominate():
    """Regression: PCA d=D reconstructs exactly; a useless-rollout model must not win."""
    per = {"pca_linear": [_run(1e-9, 0.05, 1.0, 9)], "gru_res": [_run(0.02, 0.01, 0.02, 22000)]}
    rows = MultiObjectiveScorer(ScoreWeights()).rank(per)
    assert rows[0]["candidate_id"] == "gru_res"
    assert all(math.isfinite(r["score"]) for r in rows)


def test_val_mse_criterion_ignores_rollout_and_blowup_penalty():
    per = {"a": [_run(0.01, 0.01, 5.0, 100, blow=1.0)], "b": [_run(0.02, 0.02, 0.1, 100)]}
    assert MultiObjectiveScorer(ScoreWeights(criterion="val_mse")).rank(per)[0]["candidate_id"] == "a"
    assert MultiObjectiveScorer(ScoreWeights()).rank(per)[0]["candidate_id"] == "b"


def test_seed_aggregation_and_failed_candidates():
    runs = [_run(0.1, 0.1, 0.1, 10), _run(0.3, 0.3, 0.3, 10), {"status": "failed"}]
    agg = aggregate_seeds(runs)
    assert agg["n_seeds"] == 2 and agg["n_failed"] == 1 and abs(agg["val/recon/nrmse"] - 0.2) < 1e-9
    rows = MultiObjectiveScorer(ScoreWeights()).rank({"ok": runs, "dead": [{"status": "failed"}]})
    assert rows[-1]["candidate_id"] == "dead" and rows[-1]["score"] == float("inf")


def test_pick_rollout_key_common_longest_horizon():
    ms = [{"val/recursive/nrmse@10": 1, "val/recursive/nrmse@50": 1}, {"val/recursive/nrmse@10": 1}]
    assert pick_rollout_key(ms) == "val/recursive/nrmse@10"


def test_stability_verdict_is_worst_case_over_seeds():
    """R-20: one exploding seed must not be outvoted by two stable ones."""
    from nssc.compiler.scorer import aggregate_seeds

    def run(verdict, blow):
        return {"status": "completed", "summary": {"val/stability/verdict": verdict,
                                                   "val/stability/frac_blowup": blow}}

    agg = aggregate_seeds([run("stable", 0.0), run("stable", 0.0), run("explodes", 1.0)])
    assert agg["val/stability/verdict"] == "explodes"
    assert agg["val/stability/n_unstable_seeds"] == 1
    assert agg["val/stability/frac_blowup_max"] == 1.0
    assert agg["val/stability/verdict_by_seed"].count("stable") == 2
