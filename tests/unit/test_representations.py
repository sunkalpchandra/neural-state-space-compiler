"""Unit tests for nssc.representations (all registered encoders / decoders)."""

from __future__ import annotations

import io

import pytest
import torch

from nssc.representations import (
    MultiScaleEncoder,
    PCADecoder,
    PCAEncoder,
    build_decoder,
    build_encoder,
    explained_variance_curve,
)
from nssc.representations.multiscale import causal_moving_average
from nssc.representations.ssm import chunked_diagonal_scan
from nssc.utils.registry import DECODERS, ENCODERS

B, T, D, d = 2, 40, 12, 4
SMALL = {  # keep CPU tests fast
    "mlp": {"hidden_dims": (32, 32)},
    "tcn": {"channels": 16, "n_layers": 3},
    "gru": {"hidden": 16},
    "lstm": {"hidden": 16},
    "ssm": {"d_model": 16, "d_state": 8, "n_layers": 2, "chunk": 8},
    "multiscale": {"base_kwargs": {"channels": 16, "n_layers": 2},
                   "slow_kwargs": {"channels": 8, "n_layers": 2}, "slow_window": 4},
}
ENC_KEYS = ENCODERS.keys()
DEC_KEYS = DECODERS.keys()


def make_encoder(key: str, obs_dim: int = D, latent_dim: int = d):
    torch.manual_seed(0)
    return build_encoder(key, obs_dim, latent_dim, **SMALL.get(key, {}))


@pytest.fixture
def x() -> torch.Tensor:
    torch.manual_seed(1)
    return torch.randn(B, T, D)


def test_registries_populated():
    assert set(ENC_KEYS) >= {"pca", "linear", "mlp", "tcn", "gru", "lstm", "ssm", "multiscale"}
    assert set(DEC_KEYS) >= {"pca", "linear", "mlp"}


@pytest.mark.parametrize("key", ENC_KEYS)
def test_encoder_shape_finite(key, x):
    enc = make_encoder(key)
    enc.fit(x)
    z = enc(x)
    assert z.shape == (B, T, d)
    assert torch.isfinite(z).all()
    assert isinstance(enc.is_causal, bool)
    assert isinstance(enc.is_pointwise, bool)
    assert isinstance(enc.requires_fit, bool)


@pytest.mark.parametrize("key", DEC_KEYS)
def test_decoder_shape_finite(key):
    torch.manual_seed(0)
    dec = build_decoder(key, d, D, **({"hidden_dims": (32,)} if key == "mlp" else {}))
    z = torch.randn(B, T, d)
    xhat = dec(z)
    assert xhat.shape == (B, T, D)
    assert torch.isfinite(xhat).all()


@pytest.mark.parametrize("key", ENC_KEYS)
def test_encoder_gradients_and_num_parameters(key, x):
    enc = make_encoder(key)
    if key == "pca":
        assert enc.num_parameters() == 0
        assert enc.requires_fit
        return
    assert enc.num_parameters() > 0
    enc(x).pow(2).mean().backward()
    for name, p in enc.named_parameters():
        assert p.grad is not None, name
        assert torch.isfinite(p.grad).all(), name
        assert p.grad.abs().sum() > 0, name


@pytest.mark.parametrize("key", DEC_KEYS)
def test_decoder_gradients(key):
    torch.manual_seed(0)
    dec = build_decoder(key, d, D)
    if key == "pca":
        assert dec.num_parameters() == 0
        return
    assert dec.num_parameters() > 0
    dec(torch.randn(B, T, d)).pow(2).mean().backward()
    assert all(p.grad is not None and p.grad.abs().sum() > 0 for p in dec.parameters())


@pytest.mark.parametrize("key", [k for k in ENC_KEYS if ENCODERS.get(k).is_causal])
def test_causality(key, x):
    """Perturbing x_t at t >= k must leave z_{<k} unchanged (exactly, up to fp noise)."""
    enc = make_encoder(key).eval()
    enc.fit(x)
    k = T // 2
    x2 = x.clone()
    x2[:, k:] += torch.randn_like(x2[:, k:]) * 3.0
    with torch.no_grad():
        z1, z2 = enc(x), enc(x2)
    assert torch.allclose(z1[:, :k], z2[:, :k], atol=1e-6, rtol=0), key
    # sanity: the perturbation is visible at/after k (not a constant map)
    assert not torch.allclose(z1[:, k:], z2[:, k:], atol=1e-4)


@pytest.mark.parametrize("key", [k for k in ENC_KEYS if ENCODERS.get(k).is_pointwise])
def test_pointwise_encoders_are_timestep_independent(key, x):
    enc = make_encoder(key).eval()
    enc.fit(x)
    with torch.no_grad():
        z_seq = enc(x)
        z_flat = enc(x.reshape(-1, 1, D)).reshape(B, T, d)
    assert torch.allclose(z_seq, z_flat, atol=1e-6)


def test_pca_recovers_rank_d_subspace():
    torch.manual_seed(0)
    z_true = torch.randn(B, T, d)
    W = torch.randn(d, D)
    x = z_true @ W + 0.5  # exact rank-d data with an offset
    enc = PCAEncoder(D, d)
    enc.fit(x)
    dec = PCADecoder(d, D).tie(enc)
    xhat = dec(enc(x))
    assert torch.allclose(xhat, x, atol=1e-4)
    assert torch.allclose(enc.explained_variance_ratio.sum(), torch.tensor(1.0), atol=1e-5)
    gram = enc.components @ enc.components.T  # orthonormal components
    assert torch.allclose(gram, torch.eye(d), atol=1e-5)
    curve = explained_variance_curve(x)
    assert curve.shape == (D,)
    assert curve[d - 1] > 1 - 1e-5 and torch.all(curve[1:] >= curve[:-1] - 1e-6)
    # fewer samples than latent dims still works
    small = PCAEncoder(D, d)
    small.fit(torch.randn(3, D))
    assert torch.isfinite(small(torch.randn(1, 5, D))).all()


def test_chunked_scan_matches_sequential():
    torch.manual_seed(0)
    a = torch.rand(3, 5)
    bu = torch.randn(2, 37, 3, 5)
    fast = chunked_diagonal_scan(a, bu, chunk=8)
    h = torch.zeros(2, 3, 5)
    ref = []
    for t in range(37):
        h = a * h + bu[:, t]
        ref.append(h)
    assert torch.allclose(fast, torch.stack(ref, 1), atol=1e-5)


def _roughness(z: torch.Tensor) -> torch.Tensor:
    """Mean squared temporal difference normalised by variance (scale-free)."""
    dz = z[:, 1:] - z[:, :-1]
    return dz.pow(2).mean() / z.var(dim=1).mean().clamp_min(1e-12)


def test_multiscale_slow_branch_varies_slowly():
    torch.manual_seed(0)
    t = torch.linspace(0, 8 * torch.pi, 256)
    slow = torch.sin(0.05 * t)[None, :, None] * torch.randn(1, 1, D)
    fast = torch.sin(3.0 * t)[None, :, None] * torch.randn(1, 1, D)
    x = slow + fast + 0.1 * torch.randn(1, 256, D)
    enc = MultiScaleEncoder(D, d, slow_dim=2, slow_window=16,
                            base_kwargs={"channels": 16, "n_layers": 2},
                            slow_kwargs={"channels": 8, "n_layers": 2}).eval()
    assert enc.slow_dim == 2 and enc.fast_dim == 2
    with torch.no_grad():
        z = enc(x)
    z_slow, z_fast = enc.split(z)
    assert z_slow.shape == (1, 256, 2) and z_fast.shape == (1, 256, 2)
    assert _roughness(z_slow) < _roughness(z_fast)
    s = causal_moving_average(x, 16)
    assert s.shape == x.shape
    with pytest.raises(ValueError):
        MultiScaleEncoder(D, d, slow_dim=d)


@pytest.mark.parametrize("key", ENC_KEYS)
def test_encoder_state_dict_round_trip(key, x):
    enc = make_encoder(key)
    enc.fit(x)
    buf = io.BytesIO()
    torch.save(enc.state_dict(), buf)
    buf.seek(0)
    enc2 = build_encoder(key, D, d, **SMALL.get(key, {}))
    enc2.load_state_dict(torch.load(buf))
    enc.eval()
    enc2.eval()
    with torch.no_grad():
        assert torch.equal(enc(x), enc2(x))


@pytest.mark.parametrize("key", DEC_KEYS)
def test_decoder_state_dict_round_trip(key):
    torch.manual_seed(0)
    dec = build_decoder(key, d, D)
    if key == "pca":
        dec.components.normal_()
    buf = io.BytesIO()
    torch.save(dec.state_dict(), buf)
    buf.seek(0)
    dec2 = build_decoder(key, d, D)
    dec2.load_state_dict(torch.load(buf))
    z = torch.randn(B, T, d)
    with torch.no_grad():
        assert torch.equal(dec(z), dec2(z))


def test_tcn_receptive_field_and_pointwise_flags():
    enc = make_encoder("tcn")
    assert enc.receptive_field == 1 + 2 * (1 + 2 + 4)
    assert not enc.is_pointwise and enc.is_causal
    for k in ("pca", "linear", "mlp"):
        assert ENCODERS.get(k).is_pointwise
    for k in ("tcn", "gru", "lstm", "ssm", "multiscale"):
        assert not ENCODERS.get(k).is_pointwise
