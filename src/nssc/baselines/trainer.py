"""Trainer for :class:`SequenceForecaster` baselines on ``WindowDataset`` windows.

Loss per window ``x (B, L, D)`` with ``L = context + horizon``:

* teacher-forced next-step MSE over all positions ``t >= context - 1``
  (predict ``x_{t+1}`` from ``x_{≤t}``; ``predict_next_sequence``);
* optional multi-step recursive MSE: ``forecast(x[:, :context], h)`` vs
  ``x[:, context:context+h]`` with a linear horizon curriculum ``1 → rollout_horizon``
  (weight ``rollout_weight``; ``0`` disables);
* direct mode: MSE of ``predict_direct(x[:, :context])`` vs the next ``direct_horizon`` steps
  (weight ``direct_weight``) in addition to the teacher-forced term.

Optimisation mirrors :class:`nssc.training.trainer.Trainer`: AdamW, cosine
schedule with warm-up, gradient clipping, early stopping on validation loss,
best-state restore, per-epoch history.
"""

from __future__ import annotations

import copy
import math
import time
from dataclasses import asdict, dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader

from nssc.baselines.base import SequenceForecaster


@dataclass
class BaselineTrainerConfig:
    epochs: int = 50
    lr: float = 1e-3
    weight_decay: float = 1e-5
    grad_clip: float = 1.0
    scheduler: str = "cosine"  # cosine | none
    warmup_epochs: int = 0
    early_stopping_patience: int = 20
    context: int = 20  # positions t >= context-1 enter the teacher-forced loss; rollout start
    rollout_horizon: int = 10
    rollout_weight: float = 1.0  # 0 → teacher forcing only
    rollout_curriculum: bool = True  # linearly grow horizon from 1 → rollout_horizon
    curriculum_epochs: int | None = None  # default: half the epochs
    direct_weight: float = 1.0
    log_every: int = 10
    max_batches_per_epoch: int | None = None
    device: str | None = None


class BaselineTrainer:
    def __init__(self, model: SequenceForecaster, cfg: BaselineTrainerConfig,
                 device: torch.device | None = None) -> None:
        self.model = model
        self.cfg = cfg
        self.device = device or torch.device(cfg.device or "cpu")
        self.model.to(self.device)
        params = [p for p in model.parameters() if p.requires_grad]
        self.opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay) if params else None
        self.history: list[dict[str, Any]] = []
        self.best_state: dict[str, Tensor] | None = None
        self.best_val = math.inf
        self.train_time_s = 0.0
        self.current_horizon = 0

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
        if cfg.rollout_weight <= 0 or cfg.rollout_horizon <= 0:
            return 0
        if not cfg.rollout_curriculum:
            return cfg.rollout_horizon
        ce = cfg.curriculum_epochs or max(1, cfg.epochs // 2)
        frac = min(1.0, (epoch + 1) / ce)
        return max(1, int(round(frac * cfg.rollout_horizon)))

    def _batch_x(self, batch: Any) -> Tensor:
        x = batch["x"] if isinstance(batch, dict) else batch[0] if isinstance(batch, (list, tuple)) else batch
        return x.to(self.device, non_blocking=True).float()

    def compute_loss(self, x: Tensor, horizon: int) -> tuple[Tensor, dict[str, float]]:
        """``x``: (B, L, D) window. Returns (total, components)."""
        cfg = self.cfg
        c = min(cfg.context, x.shape[1] - 1)
        comps: dict[str, float] = {}
        pred = self.model.predict_next_sequence(x)
        tf = F.mse_loss(pred[:, c - 1 : -1], x[:, c:])
        total = tf
        comps["teacher_forced"] = float(tf.detach())
        h = min(horizon, x.shape[1] - c)
        if h > 0 and cfg.rollout_weight > 0:
            roll = self.model.forecast(x[:, :c], h)
            rl = F.mse_loss(roll, x[:, c : c + h])
            total = total + cfg.rollout_weight * rl
            comps["rollout"] = float(rl.detach())
        if self.model.mode == "direct" and cfg.direct_weight > 0:
            hd = self.model.direct_horizon
            if x.shape[1] - c < hd:
                raise ValueError(f"window horizon {x.shape[1] - c} < direct_horizon {hd}")
            direct = self.model.predict_direct(x[:, :c])
            dl = F.mse_loss(direct, x[:, c : c + hd])
            total = total + cfg.direct_weight * dl
            comps["direct"] = float(dl.detach())
        return total, comps

    @torch.no_grad()
    def maybe_closed_form_fit(self, loader: DataLoader, max_batches: int = 50) -> None:
        """Baselines exposing ``fit(x)`` (e.g. mean) are fitted on a sample of training windows."""
        if not hasattr(self.model, "fit"):
            return
        xs = []
        for i, b in enumerate(loader):
            xs.append(self._batch_x(b))
            if i + 1 >= max_batches:
                break
        self.model.fit(torch.cat(xs, 0))

    # --------------------------------------------------------------- loops
    def train_epoch(self, loader: DataLoader, epoch: int) -> dict[str, float]:
        self.model.train()
        self.current_horizon = self._horizon_at(epoch)
        if self.opt is not None:
            for g in self.opt.param_groups:
                g["lr"] = self._lr_at(epoch)
        agg: dict[str, float] = {}
        n = 0
        for i, batch in enumerate(loader):
            if self.cfg.max_batches_per_epoch and i >= self.cfg.max_batches_per_epoch:
                break
            x = self._batch_x(batch)
            total, comps = self.compute_loss(x, self.current_horizon)
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
        """Validation loss at the *full* rollout horizon (no curriculum) for a stable criterion."""
        self.model.eval()
        agg: dict[str, float] = {}
        n = 0
        h = self.cfg.rollout_horizon if self.cfg.rollout_weight > 0 else 0
        for batch in loader:
            x = self._batch_x(batch)
            total, comps = self.compute_loss(x, h)
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
            rec = {"epoch": epoch, "lr": self._lr_at(epoch), "horizon": self.current_horizon,
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
            if self.opt is None:  # nothing to train (persistence / mean) → single pass
                break
        if self.best_state is not None:
            self.model.load_state_dict(self.best_state)
        self.train_time_s = time.perf_counter() - t0
        return {"best_val": self.best_val, "epochs_run": len(self.history),
                "train_time_s": self.train_time_s, "history": self.history,
                "config": asdict(cfg)}
