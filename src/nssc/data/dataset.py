"""Trajectory dataset container and torch sliding-window dataset.

Arrays are ``(N, T, D)`` for observations (float32), ``(N, T, d)`` for optional
ground-truth latents (never fed to models), ``(T,)`` for time stamps and an
optional boolean ``mask`` ``(N, T, D)`` (``True`` = observed).
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from nssc.data.splits import check_no_leakage, trajectory_split


@dataclass
class TrajectoryDataset:
    x: np.ndarray                      # (N, T, D) float32
    t: np.ndarray                      # (T,)
    z_true: np.ndarray | None = None   # (N, T, d)
    mask: np.ndarray | None = None     # (N, T, D) bool
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.x = np.asarray(self.x, dtype=np.float32)
        self.t = np.asarray(self.t, dtype=np.float64)
        if self.x.ndim != 3:
            raise ValueError(f"x must be (N, T, D), got {self.x.shape}")
        if self.t.shape != (self.x.shape[1],):
            raise ValueError(f"t must be (T,)={self.x.shape[1]}, got {self.t.shape}")
        if self.z_true is not None:
            self.z_true = np.asarray(self.z_true, dtype=np.float32)
        if self.mask is not None:
            self.mask = np.asarray(self.mask, dtype=bool)

    # -------------------------------------------------------------- properties
    @property
    def n_traj(self) -> int:
        return self.x.shape[0]

    @property
    def n_steps(self) -> int:
        return self.x.shape[1]

    @property
    def obs_dim(self) -> int:
        return self.x.shape[2]

    @property
    def latent_dim(self) -> int | None:
        return None if self.z_true is None else self.z_true.shape[2]

    def __len__(self) -> int:
        return self.n_traj

    # ------------------------------------------------------------------ subset
    def subset(self, idx: np.ndarray, split_name: str | None = None) -> TrajectoryDataset:
        """Select trajectories ``idx`` (a copy). ``metadata['split']`` records the name."""
        idx = np.asarray(idx)
        meta = copy.deepcopy(self.metadata)
        if split_name is not None:
            meta["split"] = split_name
        meta["traj_idx"] = idx.tolist()
        # Real-data loaders store per-trajectory annotations (e.g. subject id per segment) and
        # list their keys under ``per_traj_keys`` so subsets keep them aligned with ``x``.
        for key in meta.get("per_traj_keys", []):
            if key in meta and len(meta[key]) == self.n_traj:
                meta[key] = [meta[key][i] for i in idx.tolist()]
        meta.pop("split_indices", None)  # a subset is not re-splittable by the parent's indices
        return TrajectoryDataset(
            x=self.x[idx].copy(), t=self.t.copy(),
            z_true=None if self.z_true is None else self.z_true[idx].copy(),
            mask=None if self.mask is None else self.mask[idx].copy(), metadata=meta,
        )

    def split(self, seed: int | None = None, fractions: tuple[float, float, float] | None = None
              ) -> dict[str, TrajectoryDataset]:
        """Trajectory-level train/val/test split (see :func:`trajectory_split`).

        ``seed``/``fractions`` default to ``metadata['config']['split']`` when present,
        else ``0`` / ``(0.7, 0.15, 0.15)``.

        If ``metadata['split_indices'] = {'train': [...], 'val': [...], 'test': [...]}`` is
        present (real-data loaders with e.g. subject-level splits) those explicit
        trajectory indices are used verbatim and ``seed``/``fractions`` are ignored.
        """
        fixed = self.metadata.get("split_indices")
        if fixed:
            parts = {k: np.sort(np.asarray(v, dtype=int)) for k, v in fixed.items()}
            check_no_leakage(parts.get("train", []), parts.get("val", []), parts.get("test", []))
            return {k: self.subset(v, k) for k, v in parts.items()}
        sc = (self.metadata.get("config") or {}).get("split") or {}
        seed = int(sc.get("seed", 0)) if seed is None else seed
        fractions = tuple(sc.get("fractions", (0.7, 0.15, 0.15))) if fractions is None else fractions
        parts = trajectory_split(self.n_traj, fractions=fractions, seed=seed)
        return {k: self.subset(v, k) for k, v in parts.items()}

    # --------------------------------------------------------------- normalize
    def compute_stats(self) -> dict[str, np.ndarray]:
        """Per-dim ``mean``/``std`` ``(D,)`` over all trajectories and steps (NaN-aware)."""
        flat = self.x.reshape(-1, self.obs_dim).astype(np.float64)
        mean = np.nanmean(flat, axis=0)
        std = np.nanstd(flat, axis=0)
        std = np.where(std < 1e-8, 1.0, std)
        return {"mean": mean.astype(np.float32), "std": std.astype(np.float32)}

    def normalize(self, stats: dict[str, np.ndarray] | None = None
                  ) -> tuple[TrajectoryDataset, dict[str, np.ndarray]]:
        """Return ``((x - mean) / std`` copy, stats). Pass train stats for val/test."""
        stats = self.compute_stats() if stats is None else stats
        xn = (self.x - stats["mean"]) / stats["std"]
        meta = copy.deepcopy(self.metadata)
        meta["normalized"] = True
        return TrajectoryDataset(xn.astype(np.float32), self.t, self.z_true, self.mask, meta), stats

    # ---------------------------------------------------------------- save/load
    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays: dict[str, Any] = {"x": self.x, "t": self.t,
                                  "metadata": np.array(json.dumps(self.metadata, default=str))}
        if self.z_true is not None:
            arrays["z_true"] = self.z_true
        if self.mask is not None:
            arrays["mask"] = self.mask
        np.savez_compressed(path, **arrays)
        return path

    @classmethod
    def load(cls, path: str | Path) -> TrajectoryDataset:
        with np.load(path, allow_pickle=False) as d:
            return cls(
                x=d["x"], t=d["t"],
                z_true=d["z_true"] if "z_true" in d else None,
                mask=d["mask"] if "mask" in d else None,
                metadata=json.loads(str(d["metadata"])),
            )

    def to_torch(self, device: str | torch.device = "cpu") -> dict[str, torch.Tensor]:
        """Tensors ``x (N,T,D)``, ``t (T,)``, optional ``z_true``, ``mask``."""
        out = {"x": torch.as_tensor(self.x, device=device),
               "t": torch.as_tensor(self.t, dtype=torch.float32, device=device)}
        if self.z_true is not None:
            out["z_true"] = torch.as_tensor(self.z_true, device=device)
        if self.mask is not None:
            out["mask"] = torch.as_tensor(self.mask, device=device)
        return out


def n_windows(n_traj: int, n_steps: int, length: int, stride: int) -> int:
    """Number of sliding windows: ``N * (floor((T - L) / stride) + 1)`` (0 if ``T < L``)."""
    if n_steps < length:
        return 0
    return n_traj * ((n_steps - length) // stride + 1)


class WindowDataset(Dataset):
    """Sliding windows of length ``context + horizon`` over every trajectory.

    Item: ``{'x': (L, D), 'traj': int, 'start': int}`` plus ``'z_true': (L, d)`` and
    ``'mask': (L, D)`` when present in the source dataset.
    """

    def __init__(self, ds: TrajectoryDataset, context: int, horizon: int, stride: int = 1) -> None:
        if stride < 1:
            raise ValueError("stride must be >= 1")
        self.ds, self.context, self.horizon, self.stride = ds, int(context), int(horizon), int(stride)
        self.length = self.context + self.horizon
        self.per_traj = max(0, (ds.n_steps - self.length) // self.stride + 1) if ds.n_steps >= self.length else 0
        self.x = torch.as_tensor(ds.x)
        self.z = None if ds.z_true is None else torch.as_tensor(ds.z_true)
        self.mask = None if ds.mask is None else torch.as_tensor(ds.mask)

    def __len__(self) -> int:
        return self.ds.n_traj * self.per_traj

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        if i < 0 or i >= len(self):
            raise IndexError(i)
        traj, w = divmod(i, self.per_traj)
        s = w * self.stride
        sl = slice(s, s + self.length)
        item = {"x": self.x[traj, sl], "traj": torch.tensor(traj), "start": torch.tensor(s)}
        if self.z is not None:
            item["z_true"] = self.z[traj, sl]
        if self.mask is not None:
            item["mask"] = self.mask[traj, sl]
        return item


def make_loaders(ds_splits: dict[str, TrajectoryDataset], context: int, horizon: int,
                 batch_size: int = 64, stride: int = 1, num_workers: int = 0,
                 shuffle_train: bool = True, seed: int | None = None) -> dict[str, DataLoader]:
    """Build a ``DataLoader`` per split; only ``train`` is shuffled.

    ``seed`` seeds the shuffling generator so batch order is reproducible independently of global
    RNG consumption elsewhere in the process (review finding R-04). Note that the *trajectory
    split itself* is deliberately fixed across run seeds (it comes from the dataset config), so a
    seed sweep varies initialisation and batch order, not the data partition.
    """
    loaders = {}
    for name, ds in ds_splits.items():
        wds = WindowDataset(ds, context, horizon, stride)
        gen = None
        if seed is not None and name == "train" and shuffle_train:
            gen = torch.Generator()
            gen.manual_seed(int(seed))
        loaders[name] = DataLoader(wds, batch_size=batch_size,
                                   shuffle=(name == "train" and shuffle_train),
                                   num_workers=num_workers, drop_last=False, generator=gen)
    return loaders
