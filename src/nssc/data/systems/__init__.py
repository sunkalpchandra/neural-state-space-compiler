"""Synthetic dynamical systems; importing this package populates ``SYSTEMS``."""

from nssc.data.systems.base import DynamicalSystem
from nssc.data.systems.coupled_oscillators import CoupledOscillators
from nssc.data.systems.fitzhugh_nagumo import FitzHughNagumo
from nssc.data.systems.gray_scott import GrayScott1D
from nssc.data.systems.harmonic import DampedOscillator, HarmonicOscillator
from nssc.data.systems.kuramoto import Kuramoto, observe_sin_cos, order_parameter
from nssc.data.systems.lorenz63 import Lorenz63
from nssc.data.systems.lorenz96 import Lorenz96
from nssc.data.systems.lotka_volterra import LotkaVolterra
from nssc.data.systems.pendulum import Pendulum
from nssc.data.systems.vanderpol import VanDerPol

__all__ = [
    "CoupledOscillators",
    "DampedOscillator",
    "DynamicalSystem",
    "FitzHughNagumo",
    "GrayScott1D",
    "HarmonicOscillator",
    "Kuramoto",
    "Lorenz63",
    "Lorenz96",
    "LotkaVolterra",
    "Pendulum",
    "VanDerPol",
    "observe_sin_cos",
    "order_parameter",
]
