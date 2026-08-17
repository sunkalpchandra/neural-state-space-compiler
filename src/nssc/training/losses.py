"""Composite loss for latent state-space models.

    L = w_recon · ||x − D(E(x))||²
      + w_latent_1step · ||E(x)_{t+1} − F(E(x)_t)||²          (latent consistency)
      + w_obs_1step   · ||x_{t+1} − D(F(E(x)_t))||²           (one-step prediction)
      + w_rollout     · mean_k ||x_{t+k} − D(F^k(E(x)_t))||²  (multi-step, k ≤ H_train)
      + w_stability   · stability regulariser (Jacobian spectral radius surrogate)
      + Σ dynamics.extra_losses()  (e.g. Koopman consistency)

Rollout horizon can follow a curriculum (``rollout_horizon`` grows over epochs).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import Tensor

from nssc.models.latent_model import LatentModel


@dataclass
class LossWeights:
    recon: float = 1.0
    latent_1step: float = 1.0
    obs_1step: float = 1.0
    rollout: float = 1.0
    stability: float = 0.0
    extra: dict[str, float] = field(default_factory=dict)  # per-key weights for extra_losses


class LatentDynamicsLoss:
    def __init__(self, weights: LossWeights, rollout_horizon: int = 10, rollout_stride: int = 1,
                 stability_target: float = 1.0, stability_samples: int = 16,
                 detach_rollout_targets: bool = True) -> None:
        self.w = weights
        self.rollout_horizon = rollout_horizon
        self.rollout_stride = rollout_stride
        self.stability_target = stability_target
        self.stability_samples = stability_samples
        self.detach_rollout_targets = detach_rollout_targets

    def __call__(self, model: LatentModel, x: Tensor) -> tuple[Tensor, dict[str, float]]:
        """``x``: (B, T, D). Returns (total_loss, dict of unweighted component values)."""
        w = self.w
        z = model.encode(x)
        x_hat = model.decode(z)
        comps: dict[str, Tensor] = {}
        comps["recon"] = torch.mean((x_hat - x) ** 2)

        z_prev, z_next = z[:, :-1], z[:, 1:]
        z_next_hat = model.dynamics.step_sequence(z_prev)
        target_z = z_next.detach() if self.detach_rollout_targets else z_next
        comps["latent_1step"] = torch.mean((z_next_hat - target_z) ** 2)
        comps["obs_1step"] = torch.mean((model.decode(z_next_hat) - x[:, 1:]) ** 2)

        H = min(self.rollout_horizon, x.shape[1] - 1)
        if w.rollout > 0 and H > 1:
            starts = list(range(0, x.shape[1] - H, self.rollout_stride)) or [0]
            z0 = z[:, starts].reshape(-1, z.shape[-1])
            z_roll = model.dynamics.rollout(z0, H)  # (B*S, H, d)
            x_roll = model.decode(z_roll)
            tgt = torch.stack([x[:, s + 1 : s + 1 + H] for s in starts], dim=1)
            tgt = tgt.reshape(-1, H, x.shape[-1])
            comps["rollout"] = torch.mean((x_roll - tgt) ** 2)
        else:
            comps["rollout"] = torch.zeros((), device=x.device)

        if w.stability > 0:
            comps["stability"] = self.stability_penalty(model, z.detach())
        else:
            comps["stability"] = torch.zeros((), device=x.device)

        total = (w.recon * comps["recon"] + w.latent_1step * comps["latent_1step"]
                 + w.obs_1step * comps["obs_1step"] + w.rollout * comps["rollout"]
                 + w.stability * comps["stability"])
        for k, v in model.dynamics.extra_losses().items():
            comps[k] = v
            total = total + w.extra.get(k, 1.0) * v
        if hasattr(model.dynamics, "consistency_loss"):
            c = model.dynamics.consistency_loss(z_prev.reshape(-1, z.shape[-1]),
                                                z_next.reshape(-1, z.shape[-1]))
            comps["koopman_consistency"] = c
            total = total + w.extra.get("koopman_consistency", 1.0) * c
        return total, {k: float(v.detach()) for k, v in comps.items()}

    def stability_penalty(self, model: LatentModel, z: Tensor) -> Tensor:
        """Penalise Jacobian spectral radius (via power-iteration-free surrogate: ||J v||/||v||
        for random unit v) above ``stability_target``. Cheap, differentiable, batched."""
        d = z.shape[-1]
        flat = z.reshape(-1, d)
        idx = torch.randperm(flat.shape[0], device=z.device)[: self.stability_samples]
        zs = flat[idx].requires_grad_(True)
        v = torch.randn_like(zs)
        v = v / (v.norm(dim=-1, keepdim=True) + 1e-8)
        out = model.dynamics.step(zs)
        jv = torch.autograd.grad(out, zs, grad_outputs=v, create_graph=True)[0]  # vᵀJ
        gain = jv.norm(dim=-1)
        return torch.relu(gain - self.stability_target).pow(2).mean()
