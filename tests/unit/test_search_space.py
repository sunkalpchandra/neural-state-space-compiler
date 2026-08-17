from nssc.search.space import CandidateSpec, generate_candidates, resolve_latent_dims


def test_generate_basic_and_pca_rule():
    c = generate_candidates({"latent_dims": [2, 4], "encoders": ["pca", "mlp"],
                             "dynamics": ["linear", "residual_mlp"]}, None, 8)
    names = {x.name for x in c}
    assert "pca+linear@d2" in names and "pca+residual_mlp@d2" not in names
    assert all(x.decoder == ("pca" if x.encoder == "pca" else "mlp") for x in c)
    assert len({x.id for x in c}) == len(c)


def test_multiscale_slow_dim_consistency():
    c = generate_candidates({"latent_dims": [2, 4],
                             "encoders": [{"name": "multiscale", "kwargs": {"slow_dim": 1}},
                                          {"name": "multiscale", "kwargs": {"slow_dim": 2}}, "mlp"],
                             "dynamics": [{"name": "multiscale", "kwargs": {"slow_dim": 1}},
                                          {"name": "multiscale", "kwargs": {"slow_dim": 2}}, "residual_mlp"]},
                            None, 8)
    for x in c:
        se, sd = x.encoder_kwargs.get("slow_dim"), x.dynamics_kwargs.get("slow_dim")
        assert not (se is not None and sd is not None and se != sd)
        assert all(v is None or v < x.latent_dim for v in (se, sd))
    assert any(x.encoder == "multiscale" and x.dynamics == "residual_mlp" for x in c)


def test_auto_dims_from_profile_and_roundtrip():
    dims = resolve_latent_dims("auto", {"suggested_latent_dims": [2, 3, 64]}, obs_dim=8)
    assert dims == [2, 3]
    spec = CandidateSpec(3, "mlp", "koopman", "mlp", {"hidden_dims": [8]}, {}, {"residual": True})
    assert CandidateSpec.from_dict(spec.to_dict()) == spec
    assert spec.model_config()["dynamics"]["kwargs"] == {"residual": True}
