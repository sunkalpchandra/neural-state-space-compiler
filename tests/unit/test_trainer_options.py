import torch
from torch.utils.data import DataLoader, TensorDataset

from nssc.models.builder import build_latent_model
from nssc.training import Trainer, TrainerConfig


def test_compile_and_amp_flags_do_not_break_cpu_training():
    m = build_latent_model({"latent_dim": 2, "encoder": {"name": "mlp", "kwargs": {"hidden_dims": [8]}},
                            "decoder": {"name": "mlp", "kwargs": {"hidden_dims": [8]}},
                            "dynamics": {"name": "residual_mlp", "kwargs": {"hidden_dims": [8]}}}, 3)
    x = torch.randn(8, 16, 3)
    tr = Trainer(m, TrainerConfig(epochs=1, compile=True, amp=True, log_every=100), device=torch.device("cpu"))
    assert tr.use_amp is False  # amp only on CUDA
    out = tr.fit(DataLoader(TensorDataset(x), batch_size=4))
    assert out["epochs_run"] == 1 and torch.isfinite(torch.tensor(out["history"][0]["train/total"]))
