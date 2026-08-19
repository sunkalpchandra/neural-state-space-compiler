import torch

from nssc.models.builder import build_latent_model, model_name


def test_latent_model_modes_and_shapes():
    m = build_latent_model({"latent_dim": 3, "encoder": {"name": "gru", "kwargs": {"hidden": 8}},
                            "decoder": {"name": "mlp", "kwargs": {"hidden_dims": [8]}},
                            "dynamics": {"name": "linear"}}, obs_dim=5)
    x = torch.randn(4, 12, 5)
    assert m.reconstruct(x).shape == (4, 12, 5)
    xn, z, zn = m.predict_teacher_forced(x)
    assert xn.shape == (4, 11, 5) and z.shape == (4, 12, 3) and zn.shape == (4, 11, 3)
    xr, zr = m.rollout(x[:, :6], 7)
    assert xr.shape == (4, 7, 5) and zr.shape == (4, 7, 3)
    xr2, _ = m.rollout_from_latent(zr[:, 0], 3)
    assert xr2.shape == (4, 3, 5)
    counts = m.num_parameters()
    assert counts["total"] == counts["encoder"] + counts["dynamics"] + counts["decoder"] > 0
    assert model_name({"latent_dim": 3, "encoder": "gru", "dynamics": "linear"}) == "gru+linear@d3"


def test_builder_pairs_pca_decoder_and_ties():
    m = build_latent_model({"latent_dim": 2, "encoder": "pca", "decoder": "pca", "dynamics": "linear"}, 6)
    x = torch.randn(3, 20, 6)
    m.encoder.fit(x)
    m.decoder.tie(m.encoder)
    assert torch.allclose(m.decoder.components, m.encoder.components)


def test_stored_size_counts_buffers_so_pca_is_not_free():
    """The compiler's complexity term must see PCA's component matrix (review finding)."""
    import torch

    from nssc.models.builder import build_latent_model

    m = build_latent_model({"latent_dim": 4, "encoder": "pca", "decoder": "pca",
                            "dynamics": "linear"}, obs_dim=32)
    m.encoder.fit(torch.randn(2, 40, 32))
    m.decoder.tie(m.encoder)
    c = m.num_parameters()
    assert c["encoder"] == 0 and c["encoder_stored"] > 32 * 4
    assert c["total_stored"] > c["total"]
    mlp = build_latent_model({"latent_dim": 4, "encoder": {"name": "mlp", "kwargs": {"hidden_dims": [8]}},
                              "decoder": {"name": "mlp", "kwargs": {"hidden_dims": [8]}},
                              "dynamics": "linear"}, obs_dim=32)
    assert mlp.num_parameters()["total_stored"] == mlp.num_parameters()["total"]
