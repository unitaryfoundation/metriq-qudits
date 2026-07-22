"""Unit tests for metriq_qudits.benchmark.gates (ideal-gate operator primitives)."""

import math

import jax

jax.config.update("jax_enable_x64", True)

import numpy as np

from metriq_qudits.benchmark.gates import (
    _displacement,
    _ecd,
    _embed_cavity,
    _ladder_ops,
    _rotation_matrix,
    _snap,
)


def _np(x):
    return np.asarray(x)


# ── ladder operators ───────────────────────────────────────────────────────

def test_ladder_number_operator():
    N = 6
    a, adag = _ladder_ops(N)
    assert _np(a).shape == (N, N)
    np.testing.assert_allclose(_np(adag), _np(a).conj().T)
    # a†a = diag(0, 1, ..., N-1)
    np.testing.assert_allclose(_np(adag @ a), np.diag(np.arange(N)), atol=1e-10)


def test_annihilation_lowers_fock():
    N = 5
    a, _ = _ladder_ops(N)
    ket2 = np.eye(N)[2]
    np.testing.assert_allclose(_np(a) @ ket2, np.sqrt(2) * np.eye(N)[1], atol=1e-10)


# ── displacement ───────────────────────────────────────────────────────────

def test_displacement_unitary_and_identity_at_zero():
    N = 25
    a, adag = _ladder_ops(N)
    D = _np(_displacement(0.5 + 0.3j, a, adag))
    np.testing.assert_allclose(D.conj().T @ D, np.eye(N), atol=1e-9)
    np.testing.assert_allclose(_np(_displacement(0.0, a, adag)), np.eye(N), atol=1e-12)


def test_displacement_dagger_is_negative_alpha():
    N = 20
    a, adag = _ladder_ops(N)
    alpha = 0.4 - 0.2j
    D = _np(_displacement(alpha, a, adag))
    np.testing.assert_allclose(D.conj().T, _np(_displacement(-alpha, a, adag)), atol=1e-9)


def test_displacement_creates_coherent_state():
    N = 24
    a, adag = _ladder_ops(N)
    alpha = 0.7
    psi = _np(_displacement(alpha, a, adag)) @ np.eye(N)[0]
    poisson = np.array([
        np.exp(-abs(alpha) ** 2) * abs(alpha) ** (2 * n) / math.factorial(n)
        for n in range(8)
    ])
    np.testing.assert_allclose(np.abs(psi[:8]) ** 2, poisson, atol=1e-6)


# ── ECD ────────────────────────────────────────────────────────────────────

def test_ecd_returns_conjugate_half_displacements():
    N = 20
    a, adag = _ladder_ops(N)
    beta = 0.6 + 0.75j
    D_neg, D_pos = _ecd(beta, a, adag)
    np.testing.assert_allclose(_np(D_pos), _np(_displacement(beta / 2, a, adag)), atol=1e-10)
    np.testing.assert_allclose(_np(D_neg), _np(D_pos).conj().T, atol=1e-12)
    np.testing.assert_allclose(_np(D_neg), _np(_displacement(-beta / 2, a, adag)), atol=1e-9)


# ── rotation matrix ────────────────────────────────────────────────────────

def test_rotation_identity_at_zero():
    np.testing.assert_allclose(_np(_rotation_matrix(0.0, 0.0)), np.eye(2), atol=1e-12)


def test_rotation_is_unitary():
    R = _np(_rotation_matrix(0.9, 1.3))
    np.testing.assert_allclose(R.conj().T @ R, np.eye(2), atol=1e-10)


def test_rotation_pi_pulse():
    R = _np(_rotation_matrix(np.pi, 0.0))
    np.testing.assert_allclose(R, np.array([[0, -1j], [-1j, 0]]), atol=1e-10)


# ── cavity embedding ───────────────────────────────────────────────────────

def test_embed_single_mode_is_the_operator():
    N = 3
    a, _ = _ladder_ops(N)
    np.testing.assert_allclose(_np(_embed_cavity(a, 0, 1, N)), _np(a))


def test_embed_two_modes():
    N = 2
    a, _ = _ladder_ops(N)
    eye = np.eye(N)
    np.testing.assert_allclose(_np(_embed_cavity(a, 0, 2, N)), np.kron(_np(a), eye), atol=1e-12)
    np.testing.assert_allclose(_np(_embed_cavity(a, 1, 2, N)), np.kron(eye, _np(a)), atol=1e-12)


# ── SNAP ───────────────────────────────────────────────────────────────────

def test_snap_applies_per_level_phase():
    phases = np.array([0.0, np.pi / 2, np.pi])
    psi = np.array([1.0 + 0j, 1.0, 1.0])
    np.testing.assert_allclose(_np(_snap(phases, psi)), np.exp(1j * phases) * psi, atol=1e-12)
