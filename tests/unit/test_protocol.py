"""Protocol-v2 guarantees: stationary validation criterion and hash separation from v1."""

import torch
from torch.utils.data import DataLoader, TensorDataset

from nssc.experiment import PROTOCOL_VERSION, _dc, run_config_hash
from nssc.models.builder import build_latent_model
from nssc.training import Trainer, TrainerConfig


def _sine_loader(n=16, T=40, D=3):
    t = torch.linspace(0, 12, T)
    x = torch.stack([torch.stack([torch.sin(t + p), torch.cos(t + p), torch.sin(0.5 * t + p)], -1)
                     for p in torch.rand(n) * 6])
    return DataLoader(TensorDataset(x), batch_size=8)


def _model():
    return build_latent_model({"latent_dim": 3, "encoder": {"name": "mlp", "kwargs": {"hidden_dims": [16]}},
                               "decoder": {"name": "mlp", "kwargs": {"hidden_dims": [16]}},
                               "dynamics": {"name": "residual_mlp", "kwargs": {"hidden_dims": [16]}}}, 3)


def test_validation_horizon_is_fixed_under_curriculum():
    """Regression F-007: the monitored val loss must not change meaning while the curriculum ramps."""
    cfg = TrainerConfig(epochs=4, rollout_horizon=12, rollout_curriculum=True, curriculum_epochs=4,
                        log_every=100)
    tr = Trainer(_model(), cfg, device=torch.device("cpu"))
    ld = _sine_loader()
    tr.fit(ld, ld)
    assert [h["horizon"] for h in tr.history] == [3, 6, 9, 12]      # training curriculum still ramps
    assert tr.loss_fn.rollout_horizon == 12                          # restored after validation
    # with the fix off, the val loss is computed at the (shorter) curriculum horizon
    tr2 = Trainer(_model(), TrainerConfig(epochs=1, rollout_horizon=12, rollout_curriculum=True,
                                          curriculum_epochs=8, val_fixed_horizon=False, log_every=100),
                  device=torch.device("cpu"))
    tr2.fit(ld, ld)
    tr3 = Trainer(_model(), TrainerConfig(epochs=1, rollout_horizon=12, rollout_curriculum=True,
                                          curriculum_epochs=8, val_fixed_horizon=True, log_every=100),
                  device=torch.device("cpu"))
    tr3.fit(ld, ld)
    assert tr3.history[0]["val/rollout"] != tr2.history[0]["val/rollout"]


def test_protocol_version_separates_hashes():
    cfg = {"dataset": {"system": "harmonic"}, "model": {"latent_dim": 2}, "seed": 0}
    import nssc.experiment as ex

    h2 = run_config_hash(cfg)
    old = ex.PROTOCOL_VERSION
    try:
        ex.PROTOCOL_VERSION = 1
        h1 = run_config_hash(cfg)
    finally:
        ex.PROTOCOL_VERSION = old
    assert h1 != h2 and PROTOCOL_VERSION == 2


def test_dc_reports_ignored_keys():
    cfg, ignored = _dc(TrainerConfig, {"epochs": 3, "context": 20, "typo_key": 1})
    assert cfg.epochs == 3 and ignored == ["context", "typo_key"]
