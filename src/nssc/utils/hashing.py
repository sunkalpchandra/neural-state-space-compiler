"""Stable hashing of configurations and arrays."""

from __future__ import annotations

import hashlib
import json
from typing import Any

import numpy as np


def _canon(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _canon(v) for k, v in sorted(obj.items())}
    if isinstance(obj, (list, tuple)):
        return [_canon(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return {"__ndarray__": hashlib.sha1(obj.tobytes()).hexdigest(), "shape": list(obj.shape)}
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if hasattr(obj, "__dataclass_fields__"):
        from dataclasses import asdict

        return _canon(asdict(obj))
    return obj


def stable_hash(obj: Any, length: int = 12) -> str:
    """SHA-256 of the canonical JSON serialisation of ``obj``, truncated."""
    payload = json.dumps(_canon(obj), sort_keys=True, default=str).encode()
    return hashlib.sha256(payload).hexdigest()[:length]
