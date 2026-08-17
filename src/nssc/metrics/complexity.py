"""Model complexity and cost metrics: parameters, FLOPs, latency, memory."""

from __future__ import annotations

import time
from collections.abc import Callable

import torch
from torch import nn


def count_parameters(module: nn.Module, trainable_only: bool = True) -> int:
    return sum(p.numel() for p in module.parameters() if (p.requires_grad or not trainable_only))


def estimate_flops_per_step(module: nn.Module, example: torch.Tensor) -> int | None:
    """Rough multiply-add count for one forward pass via torch.utils.flop_counter.

    Returns None if the counter is unavailable or the module uses unsupported ops.
    """
    try:
        from torch.utils.flop_counter import FlopCounterMode

        with FlopCounterMode(display=False) as fc:
            module(example)
        return int(fc.get_total_flops())
    except Exception:
        return None


@torch.no_grad()
def measure_inference_latency(fn: Callable[[], object], n_warmup: int = 5, n_iters: int = 20,
                              device: torch.device | None = None) -> dict[str, float]:
    """Wall-clock latency (ms) statistics for calling ``fn`` repeatedly."""
    for _ in range(n_warmup):
        fn()
    if device is not None and device.type == "cuda":
        torch.cuda.synchronize()
    times = []
    for _ in range(n_iters):
        t0 = time.perf_counter()
        fn()
        if device is not None and device.type == "cuda":
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1e3)
    t = torch.tensor(times)
    return {"latency_ms_mean": float(t.mean()), "latency_ms_std": float(t.std(unbiased=False)),
            "latency_ms_min": float(t.min())}


def peak_memory_mb(device: torch.device | None = None) -> float | None:
    if device is not None and device.type == "cuda":
        return float(torch.cuda.max_memory_allocated(device) / 2**20)
    try:
        import resource

        # ru_maxrss is bytes on macOS, kilobytes on Linux
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        import sys

        return float(rss / 2**20) if sys.platform == "darwin" else float(rss / 2**10)
    except Exception:
        return None
