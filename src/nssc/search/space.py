"""Candidate specification and generation.

A candidate is a fully specified model config (latent dim, encoder, decoder,
dynamics + kwargs). Candidates are generated from the compiler config's
lists; ``latent_dims: auto`` defers to the dataset profile's suggestions.
"""

from __future__ import annotations

import itertools
from dataclasses import asdict, dataclass, field
from typing import Any

from nssc.utils.hashing import stable_hash

# encoder → default decoder pairing when the config does not name one
_DEFAULT_DECODER = {"pca": "pca", "linear": "linear"}


@dataclass(frozen=True)
class CandidateSpec:
    latent_dim: int
    encoder: str
    dynamics: str
    decoder: str = "mlp"
    encoder_kwargs: dict[str, Any] = field(default_factory=dict, hash=False, compare=False)
    decoder_kwargs: dict[str, Any] = field(default_factory=dict, hash=False, compare=False)
    dynamics_kwargs: dict[str, Any] = field(default_factory=dict, hash=False, compare=False)
    training_overrides: dict[str, Any] = field(default_factory=dict, hash=False, compare=False)
    tags: tuple[str, ...] = ()

    @property
    def id(self) -> str:
        return f"{self.encoder}+{self.dynamics}@d{self.latent_dim}-{stable_hash(self.to_dict(), 6)}"

    @property
    def name(self) -> str:
        return f"{self.encoder}+{self.dynamics}@d{self.latent_dim}"

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["tags"] = list(self.tags)
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> CandidateSpec:
        d = dict(d)
        d["tags"] = tuple(d.get("tags", ()))
        return cls(**d)

    def model_config(self) -> dict[str, Any]:
        return {
            "latent_dim": self.latent_dim,
            "encoder": {"name": self.encoder, "kwargs": dict(self.encoder_kwargs)},
            "decoder": {"name": self.decoder, "kwargs": dict(self.decoder_kwargs)},
            "dynamics": {"name": self.dynamics, "kwargs": dict(self.dynamics_kwargs)},
        }


def _as_specs(items: list[Any]) -> list[tuple[str, dict[str, Any]]]:
    out = []
    for it in items:
        if isinstance(it, str):
            out.append((it, {}))
        else:
            out.append((it["name"], dict(it.get("kwargs", {}) or {})))
    return out


def resolve_latent_dims(spec: Any, profile: dict[str, Any] | None, obs_dim: int) -> list[int]:
    if spec == "auto" or spec is None:
        dims = None
        if profile:
            dims = (profile.get("recommendations", {}) or {}).get("candidate_latent_dims") \
                or profile.get("suggested_latent_dims")
        dims = dims or [2, 4, 8, 16, 32]
        return sorted({int(d) for d in dims if 1 <= int(d) <= max(1, obs_dim)})
    # explicit lists are honoured as-is (overcomplete latents are legitimate, e.g. lifting)
    return sorted({int(d) for d in spec if int(d) >= 1})


def generate_candidates(cfg: dict[str, Any], profile: dict[str, Any] | None = None,
                        obs_dim: int = 1) -> list[CandidateSpec]:
    """Cartesian product of latent_dims × encoders × dynamics (× hidden sizes optional).

    ``cfg`` keys: ``latent_dims`` (list|auto), ``encoders`` (list of str|{name,kwargs}),
    ``dynamics`` (same), ``decoders`` (optional mapping encoder→decoder spec),
    ``hidden_dims`` (optional list of hidden sizes applied to mlp-like components),
    ``exclude`` (list of {encoder,dynamics,latent_dim} partial matches),
    ``max_candidates`` (int).
    """
    dims = resolve_latent_dims(cfg.get("latent_dims", "auto"), profile, obs_dim)
    encoders = _as_specs(cfg.get("encoders", ["mlp"]))
    dynamics = _as_specs(cfg.get("dynamics", ["residual_mlp"]))
    decoders_cfg = cfg.get("decoders", {}) or {}
    hidden = cfg.get("hidden_dims") or [None]
    exclude = cfg.get("exclude", []) or []
    out: list[CandidateSpec] = []
    for d, (e, ekw), (dy, dkw), h in itertools.product(dims, encoders, dynamics, hidden):
        dec_spec = decoders_cfg.get(e, _DEFAULT_DECODER.get(e, "mlp"))
        dec_name, dec_kw = _as_specs([dec_spec])[0]
        ekw, dkw, dec_kw = dict(ekw), dict(dkw), dict(dec_kw)
        tags: list[str] = []
        if h is not None:
            for kw, comp in ((ekw, e), (dkw, dy), (dec_kw, dec_name)):
                if comp in ("mlp", "residual_mlp", "koopman", "neural_ode", "gaussian", "multiscale"):
                    kw.setdefault("hidden_dims", [h, h])
            tags.append(f"h{h}")
        # multi-scale consistency: encoder and dynamics slow_dim must agree when both define it
        sd_e, sd_d = ekw.get("slow_dim"), dkw.get("slow_dim")
        if sd_e is not None and sd_d is not None and sd_e != sd_d:
            continue
        if any(sd is not None and int(sd) >= d for sd in (sd_e, sd_d)):
            continue
        if e == "pca" and dy not in ("linear", "affine") and cfg.get("pca_only_linear", True):
            continue  # PCA is frozen; pairing it with SGD dynamics is allowed but off by default
        if any(all(str(m.get(k)) == str(v) for k, v in {"encoder": e, "dynamics": dy,
                                                          "latent_dim": d}.items() if k in m)
               for m in exclude):
            continue
        out.append(CandidateSpec(d, e, dy, dec_name, ekw, dec_kw, dkw, {}, tuple(tags)))
    mx = cfg.get("max_candidates")
    if mx and len(out) > mx:
        out = out[: int(mx)]
    return out
