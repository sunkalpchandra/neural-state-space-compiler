import torch
from torch.utils.data import DataLoader, TensorDataset

from nssc.models.builder import build_latent_model
from nssc.training import (
    LatentDynamicsLoss,
    LossWeights,
    Trainer,
    TrainerConfig,
    load_checkpoint,
    save_checkpoint,
)


def _model(dyn="residual_mlp", enc="mlp", d=2, D=4):
    return build_latent_model({"latent_dim": d, "encoder": {"name": enc, "kwargs": {"hidden_dims": [16]}},
                               "decoder": {"name": "pca" if enc == "pca" else "mlp",
                                           "kwargs": {} if enc == "pca" else {"hidden_dims": [16]}},
                               "dynamics": {"name": dyn, "kwargs": {"hidden_dims": [16]} if "mlp" in dyn else {}}}, D)


def _loader(n=32, T=24, D=4):
    t = torch.linspace(0, 6, T)
    x = torch.stack([torch.stack([torch.sin(t + p), torch.cos(t + p), torch.sin(2 * t + p), t * 0], -1)
                     for p in torch.rand(n) * 6])
    return DataLoader(TensorDataset(x), batch_size=8, shuffle=False)


def test_loss_components_and_stability_penalty_grad():
    m = _model()
    x = next(iter(_loader()))[0]
    lf = LatentDynamicsLoss(LossWeights(stability=1.0), rollout_horizon=5)
    total, comps = lf(m, x)
    for k in ("recon", "latent_1step", "obs_1step", "rollout", "stability"):
        assert k in comps and comps[k] >= 0
    total.backward()
    assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())


def test_curriculum_and_early_stopping_and_history():
    m = _model()
    cfg = TrainerConfig(epochs=6, rollout_horizon=8, rollout_curriculum=True, curriculum_epochs=4,
                        early_stopping_patience=100, log_every=100)
    tr = Trainer(m, cfg, device=torch.device("cpu"))
    assert [tr._horizon_at(e) for e in range(6)] == [2, 4, 6, 8, 8, 8]
    out = tr.fit(_loader(), _loader(n=8))
    assert out["epochs_run"] == 6 and len(out["history"]) == 6
    assert out["history"][-1]["train/total"] < out["history"][0]["train/total"]
    assert tr.best_state is not None


def test_pca_dmd_closed_form_needs_no_sgd(tmp_path):
    m = _model(dyn="linear", enc="pca")
    tr = Trainer(m, TrainerConfig(epochs=5, log_every=100), device=torch.device("cpu"))
    out = tr.fit(_loader(), _loader(n=8))
    assert out["epochs_run"] == 5  # PCA frozen (0 params) but linear A is refined by SGD after DMD init
    assert m.encoder.num_parameters() == 0
    p = save_checkpoint(m, tmp_path / "ck", {"a": 1})
    m2, meta = load_checkpoint(p)
    x = torch.randn(2, 5, 4)
    assert torch.allclose(m.reconstruct(x), m2.reconstruct(x)) and meta == {"a": 1}
