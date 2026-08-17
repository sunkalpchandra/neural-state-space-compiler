"""YAML configuration loading with `_base_` inheritance and dotted overrides.

Configs are plain nested dicts wrapped in :class:`Config` (attribute access +
dict semantics). Typed dataclasses in individual subsystems consume sub-trees.

Supported features
------------------
* ``_base_: path/to/other.yaml`` (relative to the file) — deep-merged underneath.
* CLI overrides ``a.b.c=value`` parsed with YAML semantics.
* :func:`stable_hash` of the resolved config for experiment identity.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

from nssc.utils.hashing import stable_hash


def deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


class Config(dict):
    """dict with attribute access; nested dicts are wrapped lazily."""

    def __getattr__(self, item: str) -> Any:
        try:
            v = self[item]
        except KeyError as e:
            raise AttributeError(item) from e
        return Config(v) if isinstance(v, dict) and not isinstance(v, Config) else v

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value

    def get_path(self, dotted: str, default: Any = None) -> Any:
        node: Any = self
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def set_path(self, dotted: str, value: Any) -> None:
        parts = dotted.split(".")
        node: dict = self
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = value

    def to_dict(self) -> dict:
        return _plain(self)

    def hash(self) -> str:
        return stable_hash(self.to_dict())


def _plain(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _plain(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_plain(v) for v in obj]
    return obj


def load_yaml(path: str | Path) -> dict:
    path = Path(path)
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    base = data.pop("_base_", None)
    if base:
        bases = base if isinstance(base, list) else [base]
        merged: dict = {}
        for b in bases:
            merged = deep_merge(merged, load_yaml((path.parent / b).resolve()))
        data = deep_merge(merged, data)
    return data


def parse_override(s: str) -> tuple[str, Any]:
    if "=" not in s:
        raise ValueError(f"override must be key=value, got {s!r}")
    k, v = s.split("=", 1)
    return k.strip(), yaml.safe_load(v)


def load_config(path: str | Path | None = None, overrides: list[str] | None = None,
                base: dict | None = None) -> Config:
    data = copy.deepcopy(base or {})
    if path is not None:
        data = deep_merge(data, load_yaml(path))
    cfg = Config(data)
    for o in overrides or []:
        k, v = parse_override(o)
        cfg.set_path(k, v)
    return cfg


def save_yaml(cfg: dict, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(_plain(cfg), f, sort_keys=False)
