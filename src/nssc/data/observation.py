"""Tier-2 observation maps: latent ``(N, T, d)`` -> observed ``(N, T, D)``.

Every map is deterministic given its ``seed`` and round-trips through
``to_config()`` / ``from_config()``. Maps operate on the trailing dimension so
they accept any leading shape ``(..., d)``. Noise / masking helpers are plain
functions taking an explicit ``numpy.random.Generator``.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from nssc.utils.seeding import rng as make_rng


class ObservationMap:
    """Base class. Subclasses implement ``__call__``/``to_config``; register in ``OBS_MAPS``."""

    kind: str = "identity"

    def __call__(self, z: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def to_config(self) -> dict[str, Any]:
        return {"type": self.kind}

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> ObservationMap:
        cfg = dict(cfg)
        kind = cfg.pop("type")
        return OBS_MAPS[kind](**cfg)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, ObservationMap) and self.to_config() == other.to_config()

    def __hash__(self) -> int:  # pragma: no cover
        return hash(repr(self.to_config()))


class IdentityObservation(ObservationMap):
    kind = "identity"

    def __call__(self, z: np.ndarray) -> np.ndarray:
        return np.asarray(z)


class LinearObservation(ObservationMap):
    """``x = z @ W`` with ``W ~ N(0, 1/d)`` of shape ``(d, D)`` (optionally orthonormal cols).

    ``d`` is inferred lazily from the first input if not given; the matrix is a
    deterministic function of ``(seed, d, D)``.
    """

    kind = "linear"

    def __init__(self, obs_dim: int, seed: int = 0, orthogonal: bool = False,
                 in_dim: int | None = None) -> None:
        self.obs_dim, self.seed, self.orthogonal, self.in_dim = int(obs_dim), int(seed), bool(orthogonal), in_dim
        self._W: np.ndarray | None = None

    def matrix(self, d: int) -> np.ndarray:
        if self._W is None or self._W.shape[0] != d:
            g = make_rng(self.seed)
            W = g.normal(size=(d, self.obs_dim)) / np.sqrt(d)
            if self.orthogonal:
                q, _ = np.linalg.qr(W if d >= self.obs_dim else W.T)
                W = q if d >= self.obs_dim else q.T
            self._W, self.in_dim = W, d
        return self._W

    def __call__(self, z: np.ndarray) -> np.ndarray:
        z = np.asarray(z)
        return z @ self.matrix(z.shape[-1])

    def to_config(self) -> dict[str, Any]:
        return {"type": self.kind, "obs_dim": self.obs_dim, "seed": self.seed,
                "orthogonal": self.orthogonal, "in_dim": self.in_dim}


class RandomMLPObservation(ObservationMap):
    """Fixed random tanh MLP ``d -> hidden -> ... -> D`` (numpy, no training)."""

    kind = "mlp"

    def __init__(self, obs_dim: int, hidden: int | list[int] = 64, seed: int = 0,
                 n_layers: int = 1, in_dim: int | None = None, gain: float = 1.0) -> None:
        self.obs_dim, self.seed, self.in_dim, self.gain = int(obs_dim), int(seed), in_dim, float(gain)
        self.hidden = list(hidden) if isinstance(hidden, (list, tuple)) else [int(hidden)] * n_layers
        self.n_layers = len(self.hidden)
        self._weights: list[tuple[np.ndarray, np.ndarray]] | None = None

    def weights(self, d: int) -> list[tuple[np.ndarray, np.ndarray]]:
        if self._weights is None or self._weights[0][0].shape[0] != d:
            g = make_rng(self.seed)
            dims = [d, *self.hidden, self.obs_dim]
            ws = []
            for a, b in zip(dims[:-1], dims[1:]):
                W = g.normal(size=(a, b)) * self.gain / np.sqrt(a)
                bvec = 0.1 * g.normal(size=(b,))
                ws.append((W, bvec))
            self._weights, self.in_dim = ws, d
        return self._weights

    def __call__(self, z: np.ndarray) -> np.ndarray:
        h = np.asarray(z, dtype=np.float64)
        ws = self.weights(h.shape[-1])
        for i, (W, b) in enumerate(ws):
            h = h @ W + b
            if i < len(ws) - 1:
                h = np.tanh(h)
        return h

    def to_config(self) -> dict[str, Any]:
        return {"type": self.kind, "obs_dim": self.obs_dim, "hidden": self.hidden,
                "seed": self.seed, "in_dim": self.in_dim, "gain": self.gain}


class PolynomialObservation(ObservationMap):
    """Monomials up to ``degree`` (no constant): ``[z, z_i z_j (i<=j), ...]``, then optional
    random linear projection to ``obs_dim`` (``None`` keeps all monomials)."""

    kind = "polynomial"

    def __init__(self, degree: int = 2, obs_dim: int | None = None, seed: int = 0) -> None:
        self.degree, self.obs_dim, self.seed = int(degree), obs_dim, int(seed)
        self._proj: LinearObservation | None = None

    @staticmethod
    def features(z: np.ndarray, degree: int) -> np.ndarray:
        from itertools import combinations_with_replacement

        d = z.shape[-1]
        feats = [z]
        for k in range(2, degree + 1):
            for idx in combinations_with_replacement(range(d), k):
                feats.append(np.prod(z[..., list(idx)], axis=-1, keepdims=True))
        return np.concatenate(feats, axis=-1)

    def __call__(self, z: np.ndarray) -> np.ndarray:
        phi = self.features(np.asarray(z, dtype=np.float64), self.degree)
        if self.obs_dim is None:
            return phi
        if self._proj is None:
            self._proj = LinearObservation(self.obs_dim, seed=self.seed)
        return self._proj(phi)

    def to_config(self) -> dict[str, Any]:
        return {"type": self.kind, "degree": self.degree, "obs_dim": self.obs_dim, "seed": self.seed}


class RedundantObservation(ObservationMap):
    """Tile ``z`` ``repeats`` times then mix with a random near-identity linear map:
    ``x = tile(z) @ (I + alpha * G)`` -> ``D = repeats * d``."""

    kind = "redundant"

    def __init__(self, repeats: int = 4, alpha: float = 0.1, seed: int = 0) -> None:
        self.repeats, self.alpha, self.seed = int(repeats), float(alpha), int(seed)
        self._M: np.ndarray | None = None

    def __call__(self, z: np.ndarray) -> np.ndarray:
        z = np.asarray(z, dtype=np.float64)
        D = self.repeats * z.shape[-1]
        if self._M is None or self._M.shape[0] != D:
            g = make_rng(self.seed)
            self._M = np.eye(D) + self.alpha * g.normal(size=(D, D)) / np.sqrt(D)
        return np.tile(z, self.repeats) @ self._M

    def to_config(self) -> dict[str, Any]:
        return {"type": self.kind, "repeats": self.repeats, "alpha": self.alpha, "seed": self.seed}


class ObservationPipeline(ObservationMap):
    """Sequential composition of maps; ``to_config`` nests member configs."""

    kind = "pipeline"

    def __init__(self, maps: list[ObservationMap | dict[str, Any]] | None = None) -> None:
        self.maps = [m if isinstance(m, ObservationMap) else ObservationMap.from_config(m)
                     for m in (maps or [])]

    def __call__(self, z: np.ndarray) -> np.ndarray:
        for m in self.maps:
            z = m(z)
        return z

    def to_config(self) -> dict[str, Any]:
        return {"type": self.kind, "maps": [m.to_config() for m in self.maps]}


OBS_MAPS: dict[str, type[ObservationMap]] = {
    c.kind: c for c in (IdentityObservation, LinearObservation, RandomMLPObservation,
                        PolynomialObservation, RedundantObservation, ObservationPipeline)
}


def add_noise(x: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
    """Additive iid Gaussian noise with std ``sigma`` (``sigma <= 0`` returns a copy)."""
    x = np.asarray(x)
    if sigma <= 0:
        return x.copy()
    return x + sigma * rng.normal(size=x.shape)


def mask_missing(x: np.ndarray, rate: float, rng: np.random.Generator
                 ) -> tuple[np.ndarray, np.ndarray]:
    """Drop entries iid with prob ``rate``. Returns ``(x_with_nan, mask)`` where
    ``mask`` is ``True`` for observed entries (same shape as ``x``)."""
    x = np.asarray(x, dtype=np.float64).copy()
    mask = rng.random(x.shape) >= rate if rate > 0 else np.ones(x.shape, dtype=bool)
    x[~mask] = np.nan
    return x, mask


def irregular_subsample(t: np.ndarray, x: np.ndarray, keep_rate: float,
                        rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Keep a random subset of time indices (shared across the batch), always keeping ``t[0]``.

    ``t``: ``(T,)``, ``x``: ``(..., T, D)`` -> ``(t_kept, x_kept)`` with ``T' <= T``.
    """
    T = len(t)
    keep = rng.random(T) < keep_rate
    keep[0] = True
    idx = np.nonzero(keep)[0]
    return np.asarray(t)[idx], np.asarray(x)[..., idx, :]
