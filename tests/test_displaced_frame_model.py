"""Unit tests for metriq_qudits.physics.displaced_frame_model."""

import math

import numpy as np
import pytest

from metriq_qudits.physics.displaced_frame_model import (
    ancilla_drive_coefficients,
    mode_conditional_coefficients,
    mode_diagonal_coefficients,
    mode_static_coefficients,
    storage_parameters,
)
from metriq_qudits.pulses.pulse_primitives import Storage


def _storage():
    return Storage(chi_kHz=30.0, chi_prime_Hz=1.0, Ks_Hz=2.0, kappa_kHz=5.0)


def test_storage_parameters_conversions():
    chi, chi_prime, self_kerr, kappa = storage_parameters(_storage())
    assert chi == pytest.approx(2 * math.pi * 30.0 * 1e-6)
    assert chi_prime == pytest.approx(2 * math.pi * 1.0 * 1e-9)
    assert self_kerr == pytest.approx(2 * math.pi * 2.0 * 1e-9)
    assert kappa == pytest.approx(2 * math.pi * 5.0 * 1e-6)


def test_static_coefficients_relate_to_parameters():
    s = _storage()
    chi, chi_prime, self_kerr, _ = storage_parameters(s)
    assert mode_static_coefficients(s) == pytest.approx(
        (chi / 2.0, -chi, -chi_prime / 2.0, -self_kerr / 2.0)
    )


def test_diagonal_coefficients_vanish_at_alpha_zero():
    for arr in mode_diagonal_coefficients(np.zeros(4), _storage()):
        assert np.allclose(arr, 0.0)


def test_diagonal_coefficients_scale_with_photon_number():
    s = _storage()
    chi, chi_prime, self_kerr, _ = storage_parameters(s)
    alpha = np.array([0.0, 1.0 + 0j, 2.0 + 0j])
    a2 = np.abs(alpha) ** 2
    c_n, c_nnq, c_nq = mode_diagonal_coefficients(alpha, s)
    np.testing.assert_allclose(c_n, -2.0 * self_kerr * a2)
    np.testing.assert_allclose(c_nnq, -2.0 * chi_prime * a2)
    np.testing.assert_allclose(c_nq, -chi * a2 - (chi_prime / 2.0) * a2 ** 2)


def test_conditional_coefficients_track_real_imag_alpha():
    s = _storage()
    chi, *_ = storage_parameters(s)
    alpha = np.array([1.0 + 2.0j, -0.5 + 0.3j])
    c_x, c_y = mode_conditional_coefficients(alpha, s)
    np.testing.assert_allclose(c_x, -chi * np.real(alpha))
    np.testing.assert_allclose(c_y, -chi * np.imag(alpha))


def test_conditional_coefficients_vanish_at_alpha_zero():
    c_x, c_y = mode_conditional_coefficients(np.zeros(3), _storage())
    assert np.allclose(c_x, 0.0)
    assert np.allclose(c_y, 0.0)


def test_ancilla_drive_coefficients_split_iq():
    i, q = ancilla_drive_coefficients(np.array([1.0 + 2.0j, 3.0 - 4.0j]))
    np.testing.assert_allclose(i, [1.0, 3.0])
    np.testing.assert_allclose(q, [2.0, -4.0])
