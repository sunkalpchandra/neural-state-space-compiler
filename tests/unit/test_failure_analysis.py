from nssc.evaluation.failure_analysis import categorize


def _m(**kw):
    base = {"recon/nrmse": 0.05, "teacher_forced/nrmse": 0.05, "recursive/nrmse_mean": 0.2,
            "stability/rho_max": 1.05, "stability/frac_blowup": 0.0, "stability/lyapunov_max": 0.0}
    base.update(kw)
    return {"val": base}


def test_ok():
    assert categorize(_m()).verdict == "ok"


def test_instability_and_recon():
    r = categorize(_m(**{"stability/frac_blowup": 0.5, "recon/nrmse": 0.9}))
    assert {"latent_instability", "poor_reconstruction"} <= set(r.categories)


def test_chaotic_divergence_vs_poor_long_horizon():
    a = categorize(_m(**{"recursive/nrmse_mean": 1.2, "stability/lyapunov_max": 0.05}))
    b = categorize(_m(**{"recursive/nrmse_mean": 1.2, "stability/lyapunov_max": -0.1}))
    assert "chaotic_divergence" in a.categories and "poor_long_horizon" in b.categories


def test_overfit_underfit_collapse():
    hist = [{"train/total": 0.01, "val/total": 0.5}] * 6
    r = categorize(_m(), history=hist, latent_profile={"var_ratio": [0.99, 0.005, 0.005], "n_dead": 2,
                                                       "effective_dim": 1.05})
    assert "overfitting" in r.categories and "representation_collapse" in r.categories
    hist2 = [{"train/total": 2.0 - 0.1 * i, "val/total": 2.0} for i in range(6)]
    assert "underfitting" in categorize(_m(), history=hist2).categories
