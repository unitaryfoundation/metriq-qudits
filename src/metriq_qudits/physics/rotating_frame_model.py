"""
Backend-independent coefficients for the undisplaced rotating-frame Hamiltonian.

This is an *independent* verification counterpart to ``displaced_frame_model``.
It is written self-contained on purpose — it shares no coefficient code with the
displaced frame — so that agreement between the two simulators is a genuine
cross-check rather than a tautology.

Physical picture: the same cavity–transmon system in the same g/e-midpoint
rotating frame, but with no classical displacement α(t). The cavity is therefore
driven directly by ε(t) instead of the drive being folded into a trajectory.

All frequencies are in rad/ns (time in ns).
"""

from __future__ import annotations

import numpy as np

from metriq_qudits.pulses.pulse_primitives import Storage

_TWO_PI = 2.0 * np.pi


# Unit conversions to rad/ns, inlined so this module depends on no shared
# physics helpers (they mirror the rad/ns convention used everywhere else).
def _rad_per_ns_from_hz(value_hz: float) -> float:
    return _TWO_PI * value_hz * 1e-9


def _rad_per_ns_from_khz(value_khz: float) -> float:
    return _TWO_PI * value_khz * 1e-6


def storage_parameters(storage: Storage) -> tuple[float, float, float, float]:
    """Return ``(chi, chi_prime, self_kerr, kappa)`` in rad/ns.

    ``self_kerr`` is the full Kerr ``K`` (its Hamiltonian coefficient is ``-K/2``);
    ``kappa`` is the cavity linewidth ``κ/2π`` converted to a rate.
    """
    return (
        _rad_per_ns_from_khz(storage.chi_kHz),
        _rad_per_ns_from_hz(storage.chi_prime_Hz),
        _rad_per_ns_from_hz(storage.Ks_Hz),
        _rad_per_ns_from_khz(storage.kappa_kHz),
    )


def static_coefficients(storage: Storage) -> tuple[float, float, float, float]:
    """Coefficients of ``(n, n*nq, a†²a²*nq, a†²a²)`` in the undisplaced frame.

    For a single qudit (one cavity mode + transmon ancilla). The g/e-midpoint
    rotating frame evaluated at α = 0 gives the drive-independent terms

        (χ/2) n − χ n·nq − (χ'/2) a†²a²·nq − (K/2) a†²a².

    These are the dispersive, second-order dispersive, and self-Kerr terms of the
    single-mode system Hamiltonian in You et al. 2024 (arXiv:2403.00275), Eq. B2.
    The transmon anharmonicity term K_q/2·nq(nq−1) is applied separately in the
    simulator's static Hamiltonian (it is an operator term, not a scalar prefactor).
    """
    chi, chi_prime, self_kerr, _ = storage_parameters(storage)
    return chi / 2.0, -chi, -chi_prime / 2.0, -self_kerr / 2.0


def cavity_drive_coefficients(epsilon: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """I/Q coefficients for ``H_cav = Re(ε)(a+a†) + Im(ε)·i(a†−a)``.

    This is the lab cavity drive generating the semiclassical trajectory
    ``dα/dt = −iε``; the rotating frame keeps it as an explicit term rather than
    displacing it away.
    """
    epsilon = np.asarray(epsilon, dtype=complex)
    return np.real(epsilon), np.imag(epsilon)


def ancilla_drive_coefficients(omega: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """I/Q coefficients for ``H_drive = Re(ω)(q+q†) + Im(ω)·i(q†−q)``."""
    omega = np.asarray(omega, dtype=complex)
    return np.real(omega), np.imag(omega)
