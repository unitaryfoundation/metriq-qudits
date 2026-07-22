from __future__ import annotations

import itertools

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
    """Computational-subspace probabilities from the cavity density matrix.

    Parameters
    ----------
    rho_cav : numpy.ndarray of complex
        Cavity density matrix, shape (N_cav**num_modes, N_cav**num_modes).
    d, num_modes, N_cav : int
        See :func:`computational_indices`.

    Returns
    -------
    numpy.ndarray of float
        Probabilities on the d**num_modes computational outcomes: non-negative
        and summing to 1 (negative diagonal noise is clipped, then renormalized).
    """
    idx = computational_indices(d, num_modes, N_cav)
    p = np.maximum(np.real(np.diag(rho_cav))[idx], 0.0)
    return p / p.sum()


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


def harmonic(n: int) -> float:
    """The n-th harmonic number H_n = Σ_{k=1}^{n} 1/k.

    Parameters
    ----------
    n : int
        Upper index, n ≥ 0 (n = 0 gives an empty sum, 0.0).

    Returns
    -------
    float
        H_n.
    """
    return float(np.sum(1.0 / np.arange(1, n + 1)))


def hog_ideal(D: int) -> float:
    """Exact Haar-ensemble mean HOG: 0.5*(1 + H_D - H_{D/2}).

    The theoretical reference a measured HOG is compared against.

    Parameters
    ----------
    D : int
        Computational-subspace dimension, D = d**num_modes.

    Returns
    -------
    float
        Haar-ensemble mean heavy-output-generation probability.
    """
    return 0.5 * (1 + harmonic(D) - harmonic(D // 2))


def compute_xeb(p: np.ndarray, q: np.ndarray, D: int) -> float:
    """Normalized linear cross-entropy benchmark: 0 = fully mixed, 1 = ideal.

    Parameters
    ----------
    p : numpy.ndarray of float
        Measured/noisy output probabilities (sum to 1), length D.
    q : numpy.ndarray of float
        Ideal output probabilities (sum to 1), same length as ``p``.
    D : int
        Computational-subspace dimension, D = d**num_modes.

    Returns
    -------
    float
        Normalized XEB (0 for the fully mixed distribution, 1 for the ideal one);
        returns 0.0 when the ideal distribution is (near) uniform.
    """
    num = D * np.dot(p, q) - 1
    den = D * np.dot(q, q) - 1
    return 0.0 if den < 1e-10 else float(num / den)


def eval_circuit(
    rho: np.ndarray, psi: np.ndarray, d: int, num_modes: int, N_cav: int,
) -> tuple[float, float, float]:
    """Compute (HOG, XEB, fidelity) for one simulated circuit.

    Parameters
    ----------
    rho : numpy.ndarray of complex
        Cavity density matrix, shape (N_cav**num_modes, N_cav**num_modes).
    psi : numpy.ndarray of complex
        Normalized target-state amplitudes, length d**num_modes.
    d, num_modes, N_cav : int
        See :func:`computational_indices`.

    Returns
    -------
    (float, float, float)
        HOG, XEB, and state fidelity for this circuit.
    """
    D = d ** num_modes
    q = np.abs(psi) ** 2
    p = ecd_probs(np.array(rho), d, num_modes, N_cav)
    fid = state_fidelity(rho, psi, d, num_modes, N_cav)
    return compute_hog(p, q), compute_xeb(p, q, D), fid
