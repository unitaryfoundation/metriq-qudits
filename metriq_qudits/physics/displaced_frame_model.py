"""Backend-independent coefficients for the displaced-frame Hamiltonian.

Both simulators build the same time-dependent Hamiltonian as a sum of fixed
operators multiplied by scalar or time-dependent coefficients,

    H(t) = Σ_k  c_k(t) · Ô_k,

and this module supplies the coefficients c_k so the QuTiP and dynamiqs backends
share one source of physics (You et al., arXiv:2403.00275, Eqs. B4-B5). Working
in the frame that follows the classical trajectory α(t) turns the large cavity
drive into the comparatively small residual couplings computed here.

The functions are grouped by the operators they multiply:
    * static      -- time-independent residual couplings
    * diagonal    -- extra frequency shifts induced by the displacement α(t)
    * conditional -- the qubit-conditioned cavity displacement (the ECD term)
    * drive       -- the direct ancilla drive

Operators Ô (the cavity/ancilla objects the coefficients multiply):
    n̂   = â†â        cavity photon-number
    n̂_q = q̂†q̂        ancilla excitation-number (0,1 qubit; 0,1,2 for a 3-level transmon)
    â, â†            cavity lowering / raising operators
    â†²â² = n̂(n̂-1)    "quartic" operator carrying self-Kerr and higher dispersion
    (â+â†), i(â†-â)  cavity position / momentum quadratures

Sign conventions and factors of one half are applied here so the backends only
have to multiply operator × coefficient. Angular frequencies are in rad/ns.
"""

from __future__ import annotations

import numpy as np

from metriq_qudits.pulses.drive_envelopes import CavityMode


def mode_static_coefficients(mode: CavityMode) -> tuple[float, float, float, float]:
    """Time-independent coefficients of the static Hamiltonian operators.

    In the displaced frame at the g/e midpoint only a small residual coupling
    remains; these are its (constant) strengths.

    Parameters
    ----------
    mode : CavityMode
        Cavity mode parameters.

    Returns
    -------
    (c_n, c_nnq, c_qnq, c_q) : tuple of float
        Coefficients [rad/ns] of, respectively, n̂ (residual cavity frequency,
        χ/2), n̂·n̂_q (dispersive coupling, -χ), â†²â²·n̂_q (second-order dispersive,
        -χ'/2), and â†²â² (self-Kerr, -K/2).
    """
    chi, chi_prime, self_kerr, _ = mode.angular_rates()
    return chi / 2.0, -chi, -chi_prime / 2.0, -self_kerr / 2.0


def mode_diagonal_coefficients(
    alpha: np.ndarray, mode: CavityMode,
) -> tuple[np.ndarray, ...]:
    """Frequency shifts induced by the classical displacement, per time sample.

    Displacing a Kerr / dispersive cavity by the classical field α(t) adds
    amplitude-dependent shifts to the number-diagonal operators; these vanish
    when |α| = 0.

    Parameters
    ----------
    alpha : numpy.ndarray of complex
        Classical cavity trajectory α(t) [dimensionless amplitude], one sample per ns.
    mode : CavityMode
        Cavity mode parameters.

    Returns
    -------
    (c_n, c_nnq, c_nq) : tuple of numpy.ndarray
        Time-dependent coefficients [rad/ns] of n̂ (self-Kerr shift, -2K|α|²),
        n̂·n̂_q (second-order dispersive shift, -2χ'|α|²), and n̂_q (the dispersive
        plus higher-order shift, -χ|α|² - (χ'/2)|α|⁴).
    """
    chi, chi_prime, self_kerr, _ = mode.angular_rates()
    abs2 = np.abs(alpha) ** 2
    return (
        -2.0 * self_kerr * abs2,
        -2.0 * chi_prime * abs2,
        -chi * abs2 - (chi_prime / 2.0) * abs2 ** 2,
    )


def mode_conditional_coefficients(
    alpha: np.ndarray, mode: CavityMode,
) -> tuple[np.ndarray, np.ndarray]:
    """Coefficients of the qubit-conditioned cavity displacement (the ECD term).

    When the ancilla is excited, the dispersive coupling displaces the cavity; in
    the displaced frame this appears as a drive on the cavity quadratures
    proportional to the classical field α(t). This is the term that makes an ECD
    gate's displacement conditional on the ancilla state.

    Parameters
    ----------
    alpha : numpy.ndarray of complex
        Classical cavity trajectory α(t) [dimensionless amplitude], one sample per ns.
    mode : CavityMode
        Cavity mode parameters.

    Returns
    -------
    (c_x, c_y) : tuple of numpy.ndarray
        Time-dependent coefficients [rad/ns] of n̂_q·(â+â†) and n̂_q·i(â†-â), equal
        to -χ·Re α and -χ·Im α respectively.
    """
    chi, _, _, _ = mode.angular_rates()
    return -chi * np.real(alpha), -chi * np.imag(alpha)


def ancilla_drive_coefficients(drive: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """I/Q coefficients for the direct ancilla drive H_drive = I·X + Q·Y.

    The canonical complex envelope follows You et al. 2024, Sec. III.A.1:
    ``drive = I + 1j*Q``, equivalently ``H_drive = drive·q† + conj(drive)·q``.

    Parameters
    ----------
    drive : numpy.ndarray of complex
        Ancilla drive envelope Ω(t) [rad/ns], one sample per ns.

    Returns
    -------
    (I, Q) : tuple of numpy.ndarray
        In-phase Re(drive) and quadrature Im(drive) components [rad/ns], the
        coefficients of the transmon X and Y operators.
    """
    return np.real(drive), np.imag(drive)
