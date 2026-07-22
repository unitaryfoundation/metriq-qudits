"""Unit tests for metriq_qudits.physics.alpha_dynamics."""

import numpy as np
import pytest

from metriq_qudits.physics.alpha_dynamics import (
    alpha_from_epsilon_finite_difference,
    conditional_cavity_rates,
)


# ── conditional_cavity_rates: term-by-term ────────────────────────────────

def test_rates_zero_when_everything_zero():
    rate_g, rate_e = conditional_cavity_rates(0j, 0j, 0j)
    assert rate_g == pytest.approx(0j)
    assert rate_e == pytest.approx(0j)


def test_pure_drive_gives_minus_i_epsilon():
    # dα/dt = −iε for both branches when no other couplings are present
    eps = 0.3 - 0.2j
    rate_g, rate_e = conditional_cavity_rates(0j, 0j, eps)
    assert rate_g == pytest.approx(-1j * eps)
    assert rate_e == pytest.approx(-1j * eps)


def test_detuning_acts_on_ground_branch():
    a = 0.5 + 0.1j
    rate_g, rate_e = conditional_cavity_rates(a, 0j, 0j, delta=0.4)
    assert rate_g == pytest.approx(-1j * 0.4 * a)
    assert rate_e == pytest.approx(0j)


def test_chi_shifts_only_the_excited_branch():
    a = 0.7 - 0.3j
    rate_g, rate_e = conditional_cavity_rates(0j, a, 0j, chi=0.9)
    assert rate_g == pytest.approx(0j)
    assert rate_e == pytest.approx(1j * 0.9 * a)


def test_chi_prime_is_photon_number_dependent_on_excited():
    a = 0.6 + 0.2j
    rate_g, rate_e = conditional_cavity_rates(0j, a, 0j, chi_prime=0.5)
    assert rate_g == pytest.approx(0j)
    assert rate_e == pytest.approx(1j * 0.5 * abs(a) ** 2 * a)


def test_self_kerr_term():
    a = 0.5 + 0.4j
    rate_g, _ = conditional_cavity_rates(a, 0j, 0j, self_kerr=0.8)
    assert rate_g == pytest.approx(1j * 0.8 * abs(a) ** 2 * a)


def test_kappa_is_pure_damping():
    a = 0.5 + 0.4j
    rate_g, _ = conditional_cavity_rates(a, 0j, 0j, kappa=0.2)
    assert rate_g == pytest.approx(-(0.2 / 2.0) * a)


# ── alpha_from_epsilon_finite_difference: the integrator ──────────────────

def test_short_trajectory_raises():
    with pytest.raises(ValueError):
        alpha_from_epsilon_finite_difference([1.0])


def test_zero_drive_stays_at_origin():
    alpha_g, alpha_e = alpha_from_epsilon_finite_difference(np.zeros(10))
    assert np.allclose(alpha_g, 0)
    assert np.allclose(alpha_e, 0)


def test_output_shapes_match_input():
    n = 7
    alpha_g, alpha_e = alpha_from_epsilon_finite_difference(np.zeros(n))
    assert alpha_g.shape == (n,)
    assert alpha_e.shape == (n,)


def test_branches_identical_without_chi():
    # with chi = chi_prime = 0 the only g/e asymmetry vanishes
    eps = 0.05 * np.ones(21)
    alpha_g, alpha_e = alpha_from_epsilon_finite_difference(eps)
    np.testing.assert_allclose(alpha_g, alpha_e)


def test_constant_drive_matches_linear_solution_at_even_indices():
    # leapfrog is exact for the linear ODE dα/dt = −iε at even indices
    eps = 0.1
    alpha_g, _ = alpha_from_epsilon_finite_difference(eps * np.ones(11))
    for i in (2, 4, 6, 8, 10):
        assert alpha_g[i] == pytest.approx(-1j * eps * i)


def test_chi_splits_ground_and_excited():
    eps = 0.05 * np.ones(21)
    alpha_g, alpha_e = alpha_from_epsilon_finite_difference(eps, chi=0.3)
    assert not np.allclose(alpha_g, alpha_e)
