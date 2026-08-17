"""Minimal, deterministic trainer for LatentModel.

Features: AdamW, cosine/plateau LR schedule, grad clipping, rollout-horizon
curriculum, early stopping on validation loss, best-checkpoint restore, per-epoch
history, wall-clock accounting, optional closed-form PCA fit before SGD.
"""

from __future__ import annotations

import copy
import math
import time
from dataclasses import asdict, dataclass, field
from typing import Any

import torch
from torch import Tensor
from torch.utils.data import DataLoader

from nssc.models.latent_model import LatentModel
from nssc.training.losses import LatentDynamicsLoss, LossWeights


@dataclass
class TrainerConfig:
    epochs: int = 50
    lr: float = 1e-3
    weight_decay: float = 1e-5
    grad_clip: float = 1.0
    scheduler: str = "cosine"  # cosine | plateau | none
    warmup_epochs: int = 0
    early_stopping_patience: int = 20
    rollout_horizon: int = 10
    rollout_curriculum: bool = True  # linearly grow horizon from 1 → rollout_horizon
    curriculum_epochs: int | None = None  # default: half the epochs
    rollout_stride: int = 4
    loss: dict[str, Any] = field(default_factory=dict)  # LossWeights fields
    log_every: int = 10
    max_batches_per_epoch: int | None = None  # for smoke tests / screening
    device: str | None = None
    amp: bool = False


class Trainer:
    def __init__(self, model: LatentModel, cfg: TrainerConfig, device: torch.device | None = None
                 ) -> None:
        self.model = model
        self.cfg = cfg
        self.device = device or torch.device(cfg.device or "cpu")
        self.model.to(self.device)
        weights = LossWeights(**{k: v for k, v in cfg.loss.items() if k != "extra"},
                              extra=dict(cfg.loss.get("extra", {})))
        self.loss_fn = LatentDynamicsLoss(weights, rollout_horizon=cfg.rollout_horizon,
                                          rollout_stride=cfg.rollout_stride)
        params = [p for p in model.parameters() if p.requires_grad]
        self.opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay) if params else None
        self.history: list[dict[str, Any]] = []
        self.best_state: dict[str, Tensor] | None = None
        self.best_val = math.inf
        self.train_time_s = 0.0

    # ------------------------------------------------------------ helpers
    def _lr_at(self, epoch: int) -> float:
        cfg = self.cfg
        if epoch < cfg.warmup_epochs:
            return cfg.lr * (epoch + 1) / cfg.warmup_epochs
        if cfg.scheduler == "cosine":
            t = (epoch - cfg.warmup_epochs) / max(1, cfg.epochs - cfg.warmup_epochs)
            return cfg.lr * 0.5 * (1 + math.cos(math.pi * min(1.0, t)))
        return cfg.lr

    def _horizon_at(self, epoch: int) -> int:
        cfg = self.cfg
        if not cfg.rollout_curriculum:
            return cfg.rollout_horizon
        ce = cfg.curriculum_epochs or max(1, cfg.epochs // 2)
        frac = min(1.0, (epoch + 1) / ce)
        return max(1, int(round(frac * cfg.rollout_horizon)))

    def _batch_x(self, batch: Any) -> Tensor:
        x = batch["x"] if isinstance(batch, dict) else batch[0] if isinstance(batch, (list, tuple)) else batch
        return x.to(self.device, non_blocking=True).float()

    @torch.no_grad()
    def maybe_closed_form_fit(self, loader: DataLoader, max_batches: int = 50) -> None:
        """PCA-style encoders: fit on a sample of training windows before SGD."""
        if not getattr(self.model.encoder, "requires_fit", False):
            return
        xs = []
        for i, b in enumerate(loader):
            xs.append(self._batch_x(b))
            if i + 1 >= max_batches:
                break
        x = torch.cat(xs, 0)
        self.model.encoder.fit(x)
        if hasattr(self.model.decoder, "tie"):
            self.model.decoder.tie(self.model.encoder)
        # closed-form dynamics init when available (e.g. least squares for linear)
        dyn = self.model.dynamics
        if hasattr(dyn, "least_squares_fit"):
            z = self.model.encode(x)
            dyn.least_squares_fit(z[:, :-1].reshape(-1, z.shape[-1]), z[:, 1:].reshape(-1, z.shape[-1]))

    # --------------------------------------------------------------- loops
    def train_epoch(self, loader: DataLoader, epoch: int) -> dict[str, float]:
        self.model.train()
        self.loss_fn.rollout_horizon = self._horizon_at(epoch)
        if self.opt is not None:
            for g in self.opt.param_groups:
                g["lr"] = self._lr_at(epoch)
        agg: dict[str, float] = {}
        n = 0
        for i, batch in enumerate(loader):
            if self.cfg.max_batches_per_epoch and i >= self.cfg.max_batches_per_epoch:
                break
            x = self._batch_x(batch)
            total, comps = self.loss_fn(self.model, x)
            if self.opt is not None:
                self.opt.zero_grad(set_to_none=True)
                total.backward()
                if self.cfg.grad_clip:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
                self.opt.step()
            comps["total"] = float(total.detach())
            for k, v in comps.items():
                agg[k] = agg.get(k, 0.0) + v
            n += 1
        return {k: v / max(n, 1) for k, v in agg.items()}

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> dict[str, float]:
        self.model.eval()
        agg: dict[str, float] = {}
        n = 0
        with torch.enable_grad():  # stability penalty needs autograd even in eval
            for batch in loader:
                x = self._batch_x(batch)
                total, comps = self.loss_fn(self.model, x)
                comps["total"] = float(total.detach())
                for k, v in comps.items():
                    agg[k] = agg.get(k, 0.0) + v
                n += 1
        return {k: v / max(n, 1) for k, v in agg.items()}

    def fit(self, train_loader: DataLoader, val_loader: DataLoader | None = None,
            log=None) -> dict[str, Any]:
        cfg = self.cfg
        t0 = time.perf_counter()
        self.maybe_closed_form_fit(train_loader)
        bad_epochs = 0
        for epoch in range(cfg.epochs):
            tr = self.train_epoch(train_loader, epoch)
            va = self.evaluate(val_loader) if val_loader is not None else {}
            rec = {"epoch": epoch, "lr": self._lr_at(epoch), "horizon": self.loss_fn.rollout_horizon,
                   **{f"train/{k}": v for k, v in tr.items()}, **{f"val/{k}": v for k, v in va.items()},
                   "time_s": time.perf_counter() - t0}
            self.history.append(rec)
            if log and (epoch % cfg.log_every == 0 or epoch == cfg.epochs - 1):
                log(rec)
            monitor = va.get("total", tr["total"])
            if not math.isfinite(monitor):
                break
            if monitor < self.best_val - 1e-8:
                self.best_val = monitor
                self.best_state = copy.deepcopy(self.model.state_dict())
                bad_epochs = 0
            else:
                bad_epochs += 1
                if bad_epochs >= cfg.early_stopping_patience:
                    break
            if self.opt is None:  # nothing to train (PCA + closed-form) → single pass
                break
        if self.best_state is not None:
            self.model.load_state_dict(self.best_state)
        self.train_time_s = time.perf_counter() - t0
        return {"best_val": self.best_val, "epochs_run": len(self.history),
                "train_time_s": self.train_time_s, "history": self.history,
                "config": asdict(cfg)}
