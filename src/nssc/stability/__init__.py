"""Stability diagnostics for latent dynamics: spectra, Lyapunov, norm growth, rollout blow-up."""

from nssc.stability.analysis import StabilityReport, analyze_stability  # noqa: F401
from nssc.stability.lyapunov import largest_lyapunov_exponent  # noqa: F401
from nssc.stability.spectral import jacobian_spectrum, spectral_radius_stats  # noqa: F401
