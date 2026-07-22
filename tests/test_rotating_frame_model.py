"""Unit tests for metriq_qudits.physics.rotating_frame_model."""

import math

import numpy as np
import pytest

from metriq_qudits.physics import displaced_frame_model as df
from metriq_qudits.physics import rotating_frame_model as rf
from metriq_qudits.pulses.pulse_primitives import Storage


def _storage():
    return Storage(chi_kHz=30.0, chi_prime_Hz=1.0, Ks_Hz=2.0, kappa_kHz=5.0)


def test_storage_parameters_conversions():
    chi, chi_prime, self_kerr, kappa = rf.storage_parameters(_storage())
    assert chi == pytest.approx(2 * math.pi * 30.0 * 1e-6)
    assert chi_prime == pytest.approx(2 * math.pi * 1.0 * 1e-9)
    assert self_kerr == pytest.approx(2 * math.pi * 2.0 * 1e-9)
    assert kappa == pytest.approx(2 * math.pi * 5.0 * 1e-6)


def test_static_coefficients_formula():
    s = _storage()
    chi, chi_prime, self_kerr, _ = rf.storage_parameters(s)
    assert rf.static_coefficients(s) == pytest.approx(
        (chi / 2.0, -chi, -chi_prime / 2.0, -self_kerr / 2.0)
    )


def test_matches_displaced_static_part():
    # The rotating frame is independently implemented, but its drive-independent
    # (α = 0) Hamiltonian must equal the displaced-frame static part. This guards
    # against the two conventions silently drifting apart.
    s = _storage()
    assert rf.storage_parameters(s) == pytest.approx(df.storage_parameters(s))
    assert rf.static_coefficients(s) == pytest.approx(df.mode_static_coefficients(s))


def test_cavity_drive_coefficients_split_iq():
    i, q = rf.cavity_drive_coefficients(np.array([1.0 + 2.0j, -3.0 + 0.5j]))
    np.testing.assert_allclose(i, [1.0, -3.0])
    np.testing.assert_allclose(q, [2.0, 0.5])


def test_ancilla_drive_coefficients_match_displaced():
    omega = np.array([0.7 - 0.2j, 1.1 + 0.3j])
    np.testing.assert_allclose(
        rf.ancilla_drive_coefficients(omega),
        df.ancilla_drive_coefficients(omega),
    )
