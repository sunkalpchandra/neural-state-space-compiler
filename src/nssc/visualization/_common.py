"""Shared array-handling helpers: torch/numpy coercion and non-finite sanitising."""

from __future__ import annotations

from typing import Any

import numpy as np

NONFINITE_NOTE = "[non-finite values replaced]"


def to_numpy(x: Any) -> np.ndarray:
    """Accept numpy arrays, torch tensors (any device, with grad), lists, scalars."""
    if x is None:
        raise ValueError("expected an array, got None")
    if hasattr(x, "detach"):
        x = x.detach()
        if hasattr(x, "cpu"):
            x = x.cpu()
        if getattr(x, "is_complex", None) and x.is_complex():
            return x.numpy()
        x = x.float().numpy() if hasattr(x, "float") else np.asarray(x)
        return np.asarray(x)
    return np.asarray(x)


def clean(x: Any, fill: float = 0.0, clip: float | None = 1e6) -> tuple[np.ndarray, bool]:
    """``(array, had_nonfinite)``: NaN → ``fill``, ±inf → ±``clip`` (or ``fill`` if clip None)."""
    a = to_numpy(x)
    if a.dtype.kind in "biu":
        a = a.astype(float)
    if a.dtype.kind == "c":
        bad = ~np.isfinite(a)
        if bad.any():
            a = np.where(bad, complex(fill, 0.0), a)
        return a, bool(bad.any())
    if a.dtype.kind != "f":
        return a, False
    bad = ~np.isfinite(a)
    if not bad.any():
        return a, False
    big = clip if clip is not None else fill
    return np.nan_to_num(a, nan=fill, posinf=big, neginf=-big), True


def note_title(title: str, had_bad: bool) -> str:
    return f"{title} {NONFINITE_NOTE}" if had_bad else title


def finite_range(a: np.ndarray, pad: float = 0.05) -> tuple[float, float]:
    a = np.asarray(a, float)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return -1.0, 1.0
    lo, hi = float(a.min()), float(a.max())
    if hi <= lo:
        return lo - 1.0, hi + 1.0
    d = (hi - lo) * pad
    return lo - d, hi + d
