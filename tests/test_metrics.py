"""Unit tests for metriq_qudits.metrics."""

import numpy as np
import pytest

from metriq_qudits.metrics import (
    compute_hog,
    compute_xeb,
    computational_indices,
    ecd_probs,
    eval_circuit,
    state_fidelity,
)


def _embed_pure(psi, n_cav):
    """Density matrix of a computational state embedded in an n_cav cavity."""
    full = np.zeros(n_cav, dtype=complex)
    full[: len(psi)] = psi
    return np.outer(full, full.conj())


# ── index bookkeeping ──────────────────────────────────────────────────────

def test_computational_indices_single_mode():
    np.testing.assert_array_equal(computational_indices(4, 1, 20), [0, 1, 2, 3])


def test_computational_indices_two_modes():
    # index = x0*N_cav + x1 over the d^2 pairs
    np.testing.assert_array_equal(computational_indices(2, 2, 10), [0, 1, 10, 11])


# ── ecd_probs ──────────────────────────────────────────────────────────────

def test_ecd_probs_selects_computational_populations():
    rho = np.diag([0.5, 0.3, 0.1, 0.05, 0.05]).astype(complex)
    p = ecd_probs(rho, d=3, num_modes=1, N_cav=5)
    # first 3 Fock levels, raw populations (leakage not added back)
    np.testing.assert_allclose(p, [0.5, 0.3, 0.1])
    assert p.sum() == pytest.approx(0.9)


def test_ecd_probs_clips_negative_diagonal():
    rho = np.diag([0.6, -0.01, 0.41]).astype(complex)
    p = ecd_probs(rho, d=3, num_modes=1, N_cav=3)
    np.testing.assert_allclose(p, [0.6, 0.0, 0.41])


# ── state_fidelity ─────────────────────────────────────────────────────────

def test_state_fidelity_pure_state_is_one():
    psi = np.array([np.sqrt(0.5), np.sqrt(0.3), np.sqrt(0.2)])
    rho = _embed_pure(psi, n_cav=6)
    assert state_fidelity(rho, psi, 3, 1, 6) == pytest.approx(1.0)


def test_state_fidelity_penalizes_leakage():
    # half the population sits at Fock level 4, outside the d=3 subspace
    rho = np.diag([0.5, 0.0, 0.0, 0.0, 0.5]).astype(complex)
    psi = np.array([1.0, 0.0, 0.0])
    assert state_fidelity(rho, psi, 3, 1, 5) == pytest.approx(0.5)


# ── HOG ────────────────────────────────────────────────────────────────────

def test_compute_hog_sums_heavy_outputs():
    q = np.array([0.4, 0.3, 0.2, 0.1])   # median 0.25 -> heavy = indices 0,1
    p = np.array([0.25, 0.25, 0.25, 0.25])
    assert compute_hog(p, q) == pytest.approx(0.5)


# ── XEB ────────────────────────────────────────────────────────────────────

def test_xeb_is_one_when_p_equals_q():
    q = np.array([0.5, 0.3, 0.2])
    assert compute_xeb(q, q, D=3) == pytest.approx(1.0)


def test_xeb_is_zero_for_uniform_output():
    q = np.array([0.5, 0.3, 0.2])
    p = np.array([1 / 3, 1 / 3, 1 / 3])
    assert compute_xeb(p, q, D=3) == pytest.approx(0.0)


def test_xeb_returns_zero_when_denominator_degenerate():
    # uniform ideal distribution -> D*<q,q> - 1 == 0
    q = np.array([0.5, 0.5])
    assert compute_xeb(np.array([0.9, 0.1]), q, D=2) == 0.0


# ── eval_circuit integration ───────────────────────────────────────────────

def test_eval_circuit_on_ideal_state():
    psi = np.array([np.sqrt(0.5), np.sqrt(0.3), np.sqrt(0.2)])
    rho = _embed_pure(psi, n_cav=6)
    m = eval_circuit(rho, psi, d=3, num_modes=1, N_cav=6)
    q = np.abs(psi) ** 2
    assert m.fid == pytest.approx(1.0)
    assert m.xeb == pytest.approx(1.0)
    assert m.hog == pytest.approx(compute_hog(q, q))
