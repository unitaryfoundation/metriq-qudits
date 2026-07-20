"""Unit conversions for the numerical model.

The simulation uses nanoseconds for time and radians for phase.  Consequently,
Hamiltonian frequencies are expressed in rad/ns and decay rates in 1/ns.
"""

from __future__ import annotations

import numpy as np

NS_PER_US = 1_000.0


def angular_frequency_from_hz(value_hz: float) -> float:
    """Convert cyclic frequency in Hz to angular frequency in rad/ns."""
    return 2.0 * np.pi * value_hz * 1e-9


def angular_frequency_from_khz(value_khz: float) -> float:
    """Convert cyclic frequency in kHz to angular frequency in rad/ns."""
    return angular_frequency_from_hz(value_khz * 1e3)


def angular_frequency_from_mhz(value_mhz: float) -> float:
    """Convert cyclic frequency in MHz to angular frequency in rad/ns."""
    return angular_frequency_from_hz(value_mhz * 1e6)


def inverse_time_from_khz(value_khz: float) -> float:
    """Convert a decay rate in kHz to 1/ns (no 2π factor)."""
    return value_khz * 1e-6


def inverse_time_from_us(value_us: float) -> float:
    """Convert a lifetime in µs to a decay rate in 1/ns."""
    if value_us <= 0:
        raise ValueError("lifetime must be positive")
    return 1.0 / (value_us * NS_PER_US)
