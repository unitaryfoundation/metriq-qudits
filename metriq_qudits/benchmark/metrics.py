import itertools

import numpy as np


def computational_indices(d, num_modes, N_cav):
    """Joint-cavity Fock indices of the d^num_modes computational basis states."""
    return np.array([
        sum(x * N_cav ** (num_modes - 1 - i) for i, x in enumerate(xs))
        for xs in itertools.product(range(d), repeat=num_modes)
    ])


def ecd_probs(rho_cav, d, num_modes, N_cav):
    """Extract d^num_modes computational-subspace probabilities from the joint cavity
    density matrix.
    """
    idx = computational_indices(d, num_modes, N_cav)
    p = np.maximum(np.real(np.diag(rho_cav))[idx], 0.0)
    return p / p.sum()


def state_fidelity(rho_cav, psi, d, num_modes, N_cav):
    """F = ⟨ψ|ρ|ψ⟩ on the computational subspace. Leakage outside the subspace
    counts against F (no renormalization)."""
    idx = computational_indices(d, num_modes, N_cav)
    rho_comp = np.asarray(rho_cav)[np.ix_(idx, idx)]
    return max(0.0, float(np.real(np.conj(psi) @ rho_comp @ psi)))


def compute_hog(p, q):
    """Fraction of noisy probability on heavy (above-median ideal) outputs."""
    return float(np.sum(p[q > np.median(q)]))


def harmonic(n):
    return float(np.sum(1.0 / np.arange(1, n + 1)))


def hog_ideal(D):
    """Exact Haar-ensemble mean HOG: 0.5*(1 + H_D - H_{D/2}).

    The theoretical reference a measured HOG is compared against."""
    return 0.5 * (1 + harmonic(D) - harmonic(D // 2))


def compute_xeb(p, q, D):
    """Normalized XEB: 0 = fully mixed, 1 = ideal."""
    num = D * np.dot(p, q) - 1
    den = D * np.dot(q, q) - 1
    return 0.0 if den < 1e-10 else float(num / den)


def eval_circuit(rho, psi, d, num_modes, N_cav):
    D = d ** num_modes
    q = np.abs(psi) ** 2
    p = ecd_probs(np.array(rho), d, num_modes, N_cav)
    fid = state_fidelity(rho, psi, d, num_modes, N_cav)
    return compute_hog(p, q), compute_xeb(p, q, D), fid
