from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np


def computational_indices(d: int, num_modes: int, N_cav: int) -> np.ndarray:
    """Cavity Fock indices of the d**num_modes computational basis states.

    Parameters
    ----------
    d : int
        Qudit dimension (number of logical levels encoded in the cavity), d ≥ 2.
    num_modes : int
        Cavity mode count.
    N_cav : int
        Fock truncation of the cavity, N_cav ≥ d.

    Returns
    -------
    numpy.ndarray of int
        Flat indices into the cavity basis, length d**num_modes.
    """
    return np.array([
        sum(x * N_cav ** (num_modes - 1 - i) for i, x in enumerate(xs))
        for xs in itertools.product(range(d), repeat=num_modes)
    ])


def ecd_probs(rho_cav: np.ndarray, d: int, num_modes: int, N_cav: int) -> np.ndarray:
    """Computational-subspace populations from the cavity density matrix.

    The raw diagonal populations, with negative numerical noise clipped to zero.
    Leakage outside the subspace is not added back, so these need not sum to 1.

    Parameters
    ----------
    rho_cav : numpy.ndarray of complex
        Cavity density matrix, shape (N_cav**num_modes, N_cav**num_modes).
    d, num_modes, N_cav : int
        See :func:`computational_indices`.

    Returns
    -------
    numpy.ndarray of float
        Populations on the d**num_modes computational outcomes.
    """
    idx = computational_indices(d, num_modes, N_cav)
    return np.maximum(np.real(np.diag(rho_cav))[idx], 0.0)


def state_fidelity(
    rho_cav: np.ndarray, psi: np.ndarray, d: int, num_modes: int, N_cav: int,
) -> float:
    """F = ⟨ψ|ρ|ψ⟩ on the computational subspace.

    Leakage outside the subspace counts against F (no renormalization).

    Parameters
    ----------
    rho_cav : numpy.ndarray of complex
        Cavity density matrix, shape (N_cav**num_modes, N_cav**num_modes).
    psi : numpy.ndarray of complex
        Normalized target-state amplitudes, length d**num_modes.
    d, num_modes, N_cav : int
        See :func:`computational_indices`.

    Returns
    -------
    float
        State fidelity in [0, 1].
    """
    idx = computational_indices(d, num_modes, N_cav)
    rho_comp = np.asarray(rho_cav)[np.ix_(idx, idx)]
    return max(0.0, float(np.real(np.conj(psi) @ rho_comp @ psi)))


def compute_hog(p: np.ndarray, q: np.ndarray) -> float:
    """Fraction of measured probability on heavy (above-median ideal) outputs.

    Parameters
    ----------
    p : numpy.ndarray of float
        Measured/noisy output probabilities (sum to 1), length D.
    q : numpy.ndarray of float
        Ideal output probabilities (sum to 1), same length as ``p``.

    Returns
    -------
    float
        Total ``p`` on the outcomes where ``q`` exceeds its median, in [0, 1].
    """
    return float(np.sum(p[q > np.median(q)]))


def xeb_terms(p: np.ndarray, q: np.ndarray, D: int) -> tuple[float, float]:
    """XEB numerator and denominator for one circuit.

    Sum each across the circuit ensemble and divide to get the reported XEB.
    """
    num = D * float(np.dot(p, q)) - 1.0
    den = D * float(np.dot(q, q)) - 1.0
    return num, den


def compute_xeb(p: np.ndarray, q: np.ndarray, D: int) -> float:
    """Normalized linear XEB for one circuit: 0 = fully mixed, 1 = ideal.

    The single-circuit ratio. For an ensemble, sum :func:`xeb_terms` and divide.
    """
    num, den = xeb_terms(p, q, D)
    return 0.0 if den < 1e-10 else float(num / den)


@dataclass
class CircuitMetrics:
    """HOG, XEB, and fidelity for one circuit, with the XEB terms to sum."""

    hog: float
    xeb: float
    fid: float
    xeb_num: float
    xeb_den: float


def eval_circuit(
    rho: np.ndarray, psi: np.ndarray, d: int, num_modes: int, N_cav: int,
) -> CircuitMetrics:
    """HOG, XEB, and fidelity for one simulated circuit."""
    D = d ** num_modes
    q = np.abs(psi) ** 2
    p = ecd_probs(np.array(rho), d, num_modes, N_cav)
    num, den = xeb_terms(p, q, D)
    xeb = 0.0 if den < 1e-10 else float(num / den)
    return CircuitMetrics(
        hog=compute_hog(p, q),
        xeb=xeb,
        fid=state_fidelity(rho, psi, d, num_modes, N_cav),
        xeb_num=num,
        xeb_den=den,
    )
