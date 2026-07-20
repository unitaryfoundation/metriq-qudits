"""
Backend-independent coefficients for the displaced-frame Hamiltonian.

All frequencies are in rad/ns.
"""

from __future__ import annotations

import numpy as np

from ecd_qv.pulses.pulse_primitives import Storage
from ecd_qv.physics.units import angular_frequency_from_hz, angular_frequency_from_khz


def storage_parameters(storage: Storage) -> tuple[float, float, float, float]:
    """Return ``(chi, chi_prime, self_kerr, kappa)`` in rad/ns.

    ``self_kerr`` is the full Kerr ``K`` whose Hamiltonian coefficient is
    ``-K/2``. ``Storage.kappa_kHz`` is the conventional quoted κ/2π linewidth.
    """
    return (
        angular_frequency_from_khz(storage.chi_kHz),
        angular_frequency_from_hz(storage.chi_prime_Hz),
        angular_frequency_from_hz(storage.Ks_Hz),
        angular_frequency_from_khz(storage.kappa_kHz),
    )


def mode_static_coefficients(storage: Storage) -> tuple[float, float, float, float]:
    """Coefficients of (n, n*nq, a†²a²*nq, a†²a²)."""
    chi, chi_prime, self_kerr, _ = storage_parameters(storage)
    return chi / 2.0, -chi, -chi_prime / 2.0, -self_kerr / 2.0


def mode_diagonal_coefficients(
    alpha: np.ndarray, storage: Storage,
) -> tuple[np.ndarray, ...]:
    """Coefficients of (n, n*nq, nq) after displacement."""
    chi, chi_prime, self_kerr, _ = storage_parameters(storage)
    abs2 = np.abs(alpha) ** 2
    return (
        -2.0 * self_kerr * abs2,
        -2.0 * chi_prime * abs2,
        -chi * abs2 - (chi_prime / 2.0) * abs2 ** 2,
    )


def mode_conditional_coefficients(
    alpha: np.ndarray, storage: Storage,
) -> tuple[np.ndarray, np.ndarray]:
    """Coefficients of nq*(a+a†) and nq*i(a†-a)."""
    chi, _, _, _ = storage_parameters(storage)
    return -chi * np.real(alpha), -chi * np.imag(alpha)


def ancilla_drive_coefficients(drive: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return I/Q coefficients for H_drive = I X + Q Y.

    The canonical complex envelope follows You et al. 2024, Sec. III.A.1:
    ``drive = I + 1j*Q``, equivalently
    ``H_drive = drive*q† + conj(drive)*q``.
    """
    return np.real(drive), np.imag(drive)
