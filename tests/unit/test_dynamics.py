"""Unit tests for all registered latent dynamics families."""

from __future__ import annotations

import io

import pytest
import torch

from nssc.dynamics import DYNAMICS, GaussianDynamics, KoopmanDynamics, build_dynamics
from nssc.dynamics.linear import AffineDynamics, LinearDynamics
from nssc.dynamics.multiscale import MultiScaleDynamics
from nssc.dynamics.neural_ode import NeuralODEDynamics

KEYS = DYNAMICS.keys()
D, B, H = 4, 3, 6

# small kwargs so tests stay fast; kept per-key so defaults are also exercised elsewhere
SMALL = {
    "mlp": dict(hidden_dims=(16, 16)),
    "residual_mlp": dict(hidden_dims=(16, 16), stability_reg=1e-3),
    "koopman": dict(hidden_dims=(16,)),
    "neural_ode": dict(hidden_dims=(16,)),
    "multiscale": dict(hidden_dims=(16,)),
    "gaussian": dict(base_kwargs=dict(hidden_dims=(16,))),
}


def make(key: str, control_dim: int = 0, **kw):
    torch.manual_seed(0)
    return build_dynamics(key, D, control_dim=control_dim, **SMALL.get(key, {}), **kw)


def _perturb(model):
    # make nonlinear families non-trivial (many start as identity)
    with torch.no_grad():
        for p in model.parameters():
            p.add_(0.1 * torch.randn_like(p))


@pytest.mark.parametrize("key", KEYS)
def test_step_and_rollout_shapes(key):
    m = make(key)
    z = torch.randn(B, D)
    out = m.step(z)
    assert out.shape == (B, D)
    roll = m.rollout(z, H)
    assert roll.shape == (B, H, D)
    assert torch.isfinite(roll).all()
    seq = m.step_sequence(torch.randn(B, H, D))
    assert seq.shape == (B, H, D)


@pytest.mark.parametrize("key", KEYS)
def test_control_input(key):
    m = make(key, control_dim=2)
    z = torch.randn(B, D)
    u = torch.randn(B, H, 2)
    assert m.step(z, u[:, 0]).shape == (B, D)
    assert m.rollout(z, H, u).shape == (B, H, D)


@pytest.mark.parametrize("key", KEYS)
def test_gradient_flows(key):
    m = make(key)
    _perturb(m)
    z = torch.randn(B, D, requires_grad=True)
    loss = m.rollout(z, H).pow(2).mean()
    for _, v in m.extra_losses().items():
        loss = loss + v
    loss.backward()
    assert z.grad is not None and torch.isfinite(z.grad).all()
    grads = [p.grad for p in m.parameters() if p.grad is not None]
    assert grads, "no parameter received a gradient"
    assert all(torch.isfinite(g).all() for g in grads)


@pytest.mark.parametrize("key", KEYS)
def test_jacobian_matches_finite_differences(key):
    m = make(key)
    _perturb(m)
    m.eval()
    z = torch.randn(B, D)
    jac = m.jacobian(z)
    assert jac.shape == (B, D, D)
    eps = 1e-3
    fd = torch.zeros(B, D, D)
    with torch.no_grad():
        for j in range(D):
            e = torch.zeros(D)
            e[j] = eps
            fd[:, :, j] = (m.step(z + e) - m.step(z - e)) / (2 * eps)
    assert torch.allclose(jac, fd, rtol=1e-3, atol=1e-3), (jac - fd).abs().max()


@pytest.mark.parametrize("key", KEYS)
def test_state_dict_round_trip(key):
    m = make(key)
    _perturb(m)
    buf = io.BytesIO()
    torch.save(m.state_dict(), buf)
    buf.seek(0)
    m2 = make(key)
    m2.load_state_dict(torch.load(buf))
    z = torch.randn(B, D)
    assert torch.allclose(m.step(z), m2.step(z))


@pytest.mark.parametrize("key", KEYS)
def test_is_linear_flag_and_num_parameters(key):
    m = make(key)
    assert m.is_linear == (key in ("linear", "affine"))
    assert m.num_parameters() > 0


# ---------------------------------------------------------------- linear
def test_linear_eigenvalues_and_least_squares():
    m = make("linear")
    ev = m.eigenvalues()
    assert ev.shape == (D,)
    assert m.spectral_radius() > 0
    A_true = torch.eye(D) * 0.9 + 0.05 * torch.randn(D, D)
    z = torch.randn(200, D)
    m.least_squares_fit(z, z @ A_true.T)
    assert torch.allclose(m.A, A_true, atol=1e-4)
    m2 = LinearDynamics.from_least_squares(z, z @ A_true.T)
    assert torch.allclose(m2.A, A_true, atol=1e-4)
    # affine recovers bias too
    b_true = torch.randn(D)
    aff = AffineDynamics(D)
    aff.least_squares_fit(z, z @ A_true.T + b_true)
    assert torch.allclose(aff.A, A_true, atol=1e-4)
    assert torch.allclose(aff.b, b_true, atol=1e-4)


def test_linear_spectral_penalty():
    m = LinearDynamics(D, spectral_norm_max=0.5)
    with torch.no_grad():
        m.A.copy_(torch.eye(D) * 2.0)
    losses = m.extra_losses()
    assert "spectral_norm" in losses and losses["spectral_norm"].item() > 0
    losses["spectral_norm"].backward()
    assert m.A.grad is not None
    assert not LinearDynamics(D).extra_losses()


# ---------------------------------------------------------------- koopman
def test_koopman_consistency_loss():
    m = make("koopman")
    z = torch.randn(B, D)
    z_next = torch.randn(B, D)
    loss = m.consistency_loss(z, z_next)
    assert loss.ndim == 0 and loss.item() >= 0
    extra = m.extra_losses()
    assert "koopman_consistency" in extra
    extra["koopman_consistency"].backward()
    assert m.K_full.grad is not None and torch.isfinite(m.K_full.grad).all()
    assert m.eigenvalues().shape == (m.obs_dim_lift,)
    # sequence input works and lifted rollout has right shape
    assert m.consistency_loss(torch.randn(B, H, D), torch.randn(B, H, D)).item() >= 0
    assert m.rollout_lifted(z, H).shape == (B, H, D)


def test_koopman_residual_and_learned_readout():
    m = KoopmanDynamics(D, residual=True, exact_readout=False, hidden_dims=(8,))
    assert m.K.shape == (4 * D, 4 * D)
    z = torch.randn(B, D)
    assert m.step(z).shape == (B, D)
    fd = m.jacobian(z)
    assert fd.shape == (B, D, D)


# ---------------------------------------------------------------- neural ODE
def test_neural_ode_matches_exp_decay():
    m = NeuralODEDynamics(D, hidden_dims=(8,), dt=0.5, n_substeps=4)
    m.vector_field = lambda z, u=None: -z  # f(z) = -z  → z(t) = z0 exp(-t)
    z0 = torch.randn(B, D)
    roll = m.rollout(z0, 6)
    t = 0.5 * torch.arange(1, 7)
    expected = z0.unsqueeze(1) * torch.exp(-t).view(1, -1, 1)
    assert torch.allclose(roll, expected, atol=1e-3, rtol=1e-3)
    assert m.trajectory(z0, 3.0, 12).shape == (B, 12, D)


def test_neural_ode_vector_field_shape_and_solvers():
    m = make("neural_ode")
    assert m.vector_field(torch.randn(B, D)).shape == (B, D)
    e = NeuralODEDynamics(D, hidden_dims=(8,), solver="euler")
    assert e.step(torch.randn(B, D)).shape == (B, D)
    with pytest.raises(KeyError):
        NeuralODEDynamics(D, solver="dopri")


# ---------------------------------------------------------------- SSM
@pytest.mark.parametrize("param", ["tanh", "exp"])
def test_ssm_stability_and_eigenvalues(param):
    m = build_dynamics("ssm", D, param=param, rank=1)
    assert (m.a.abs() < 1).all()
    assert m.eigenvalues().shape == (D,)
    lin = build_dynamics("ssm", D, param=param, feature_dim=0)
    z = torch.randn(B, D)
    roll = lin.rollout(z, 200)
    assert roll[:, -1].abs().max() < z.abs().max()  # contractive linear part


# ---------------------------------------------------------------- multiscale
@pytest.mark.parametrize("mode", ["rate", "strided"])
def test_multiscale_slow_changes_less_than_fast(mode):
    torch.manual_seed(0)
    m = MultiScaleDynamics(D, slow_dim=2, slow_rate=0.05, slow_every=3, mode=mode, hidden_dims=(16,))
    _perturb(m)
    z0 = torch.randn(B, D)
    roll = m.rollout(z0, 12)
    zs, zf = m.split(roll)
    slow_change = (zs[:, 1:] - zs[:, :-1]).abs().mean()
    fast_change = (zf[:, 1:] - zf[:, :-1]).abs().mean()
    assert slow_change < fast_change
    if mode == "strided":
        # slow block frozen on non-multiple steps
        assert torch.equal(roll[:, 1, :2], roll[:, 2, :2])
        assert not torch.equal(roll[:, 2, :2], roll[:, 3, :2])


# ---------------------------------------------------------------- gaussian
def test_gaussian_interface():
    m = make("gaussian")
    assert m.is_stochastic
    z = torch.randn(B, D)
    z_next = torch.randn(B, D)
    nll = m.nll(z, z_next)
    assert nll.ndim == 0 and torch.isfinite(nll)
    nll.backward()
    assert m.logvar_net[-1].bias.grad is not None
    assert m.sample_step(z).shape == (B, D)
    s = m.rollout_samples(z, H, n_samples=5)
    assert s.shape == (5, B, H, D) and torch.isfinite(s).all()
    mean, std = m.rollout_moments(z, H, n_samples=5)
    assert mean.shape == (B, H, D) and std.shape == (B, H, D)
    assert torch.allclose(m.step(z), m.base.step(z))
    assert m.nll(torch.randn(B, H, D), torch.randn(B, H, D)).ndim == 0
    with pytest.raises(ValueError):
        GaussianDynamics(D, base="gaussian")


def test_gaussian_wraps_linear():
    m = GaussianDynamics(D, base="linear")
    assert m.is_linear
    assert m.jacobian(torch.randn(B, D)).shape == (B, D, D)


def test_registry_and_builder():
    for k in ("linear", "affine", "mlp", "residual_mlp", "koopman", "neural_ode", "ssm", "multiscale", "gaussian"):
        assert k in DYNAMICS
    with pytest.raises(KeyError):
        build_dynamics("nope", D)
