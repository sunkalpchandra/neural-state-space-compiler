"""Small MLP builder shared by dynamics families."""

from __future__ import annotations

from collections.abc import Sequence

from torch import nn

_ACTS: dict[str, type[nn.Module]] = {
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "tanh": nn.Tanh,
    "silu": nn.SiLU,
    "elu": nn.ELU,
    "softplus": nn.Softplus,
}


def get_activation(name: str) -> nn.Module:
    if name not in _ACTS:
        raise KeyError(f"unknown activation '{name}'. Available: {sorted(_ACTS)}")
    return _ACTS[name]()


def make_mlp(
    in_dim: int,
    out_dim: int,
    hidden_dims: Sequence[int] = (128, 128),
    act: str = "gelu",
    zero_init_last: bool = False,
) -> nn.Sequential:
    """MLP ``in_dim → hidden... → out_dim`` with activation between layers.

    ``zero_init_last`` zeroes the final linear layer (useful for residual updates
    so the model starts as the identity map).
    """
    layers: list[nn.Module] = []
    d = in_dim
    for h in hidden_dims:
        layers.append(nn.Linear(d, h))
        layers.append(get_activation(act))
        d = h
    last = nn.Linear(d, out_dim)
    if zero_init_last:
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)
    layers.append(last)
    return nn.Sequential(*layers)
