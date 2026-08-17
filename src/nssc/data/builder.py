"""Build a :class:`TrajectoryDataset` from a config dict / YAML.

Config schema (all keys optional except ``system``)::

    system: lorenz63
    params: {rho: 28.0}          # overrides system defaults
    n_traj: 100
    n_steps: 500
    dt: 0.01                     # default: system.default_dt
    substeps: 1
    seed: 0
    transient: 1000              # default: system.default_transient
    observation: {type: identity | linear | mlp | polynomial | redundant | pipeline, ...}
    noise_std: 0.0
    missing_rate: 0.0
    kuramoto_sin_cos: true       # Kuramoto only: observe (cos, sin) instead of raw phases
    split: {fractions: [0.7, 0.15, 0.15], seed: 0}   # used by TrajectoryDataset.split()
    ood: {...}                   # documentation of OOD parameter ranges (hashed, not applied)

The dataset ``metadata['version']`` is ``stable_hash`` of the resolved config so
any change in split/preprocessing shows up as a new dataset version.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import numpy as np

from nssc.data import systems as _systems  # noqa: F401  (populates SYSTEMS)
from nssc.data.dataset import TrajectoryDataset
from nssc.data.observation import ObservationMap, add_noise, mask_missing
from nssc.utils.config import load_yaml
from nssc.utils.hashing import stable_hash
from nssc.utils.registry import SYSTEMS
from nssc.utils.seeding import rng as make_rng

DEFAULTS: dict[str, Any] = {
    "params": {}, "n_traj": 32, "n_steps": 256, "dt": None, "substeps": None, "seed": 0,
    "transient": None, "observation": {"type": "identity"}, "noise_std": 0.0,
    "missing_rate": 0.0, "kuramoto_sin_cos": True, "ic_scale": 1.0,
}


def resolve_config(cfg: dict[str, Any]) -> dict[str, Any]:
    """Fill defaults (including system ``dt``/``transient``) so the hash is stable."""
    if "system" not in cfg:
        raise KeyError("dataset config needs 'system'")
    out = copy.deepcopy(DEFAULTS)
    out.update(copy.deepcopy(dict(cfg)))  # unknown keys (split, ood, name) are kept & hashed
    sys_cls = SYSTEMS.get(out["system"])
    if out["dt"] is None:
        out["dt"] = float(sys_cls.default_dt)
    if out["transient"] is None:
        out["transient"] = int(sys_cls.default_transient)
    if out["substeps"] is None:
        out["substeps"] = int(sys_cls.default_substeps)
    obs = out["observation"] or {"type": "identity"}
    if isinstance(obs, str):
        obs = {"type": obs}
    out["observation"] = dict(obs)
    return out


def build_dataset(cfg: dict[str, Any] | str | Path) -> TrajectoryDataset:
    """Simulate, observe, corrupt and package a dataset. See module docstring."""
    if isinstance(cfg, (str, Path)):
        cfg = load_yaml(cfg)
    cfg = resolve_config(cfg)
    system = SYSTEMS.build(cfg["system"], params=cfg["params"], dt=cfg["dt"])
    z = system.simulate(cfg["n_traj"], cfg["n_steps"], dt=cfg["dt"], seed=cfg["seed"],
                        transient=cfg["transient"], substeps=cfg["substeps"],
                        ic_scale=cfg["ic_scale"])
    if not np.all(np.isfinite(z)):
        raise FloatingPointError(f"{cfg['system']}: non-finite trajectory")

    latent = z
    if cfg["system"] == "kuramoto" and cfg["kuramoto_sin_cos"]:
        from nssc.data.systems.kuramoto import observe_sin_cos

        latent = observe_sin_cos(z)
    obs_map = ObservationMap.from_config(cfg["observation"])
    x = obs_map(latent).astype(np.float64)

    g = make_rng(cfg["seed"] + 10_007)  # noise stream distinct from initial conditions
    x = add_noise(x, cfg["noise_std"], g)
    mask = None
    if cfg["missing_rate"] > 0:
        x, mask = mask_missing(x, cfg["missing_rate"], g)

    t = np.arange(cfg["n_steps"], dtype=np.float64) * cfg["dt"]
    metadata = {
        "system": cfg["system"], "params": system.params, "dt": cfg["dt"],
        "substeps": cfg["substeps"], "transient": cfg["transient"], "seed": cfg["seed"],
        "n_traj": cfg["n_traj"], "n_steps": cfg["n_steps"], "state_dim": system.state_dim,
        "observation": obs_map.to_config(), "noise_std": cfg["noise_std"],
        "missing_rate": cfg["missing_rate"], "config": cfg, "version": stable_hash(cfg),
    }
    return TrajectoryDataset(x=x.astype(np.float32), t=t, z_true=z.astype(np.float32),
                             mask=mask, metadata=metadata)
