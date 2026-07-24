"""Unit tests for metriq_qudits.physics.alpha_dynamics."""

import numpy as np
import pytest
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d

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


# ── against known analytical / reference solutions ─────────────────────────
#
# For K = chi' = 0 the equation dα/dt = -λα - iε, λ = iδ + κ/2, is linear and
# exactly solvable. The nonlinear case has no closed form, so it is checked
# against a high-accuracy solver. With delta = chi/2 (as the code uses it) the
# ground and excited branches rotate at -chi/2 and +chi/2.

CHI_KHZ_VALUES = [10.0, 32.8, 100.0]
CHI_PRIME_RAD_NS = 2 * np.pi * 3.0 * 1e-9
SELF_KERR_RAD_NS = 2 * np.pi * 1.0 * 1e-9


def _chi_rad_ns(chi_khz):
    return -2 * np.pi * chi_khz * 1e3 * 1e-9


def _exact_linear_decay(n, delta, kappa, a0):
    """α(t) = a0·e^{-λt} for zero drive, λ = iδ + κ/2."""
    t = np.arange(n)
    return a0 * np.exp(-(1j * delta + kappa / 2) * t)


def _exact_linear_ramp(n, slope, delta):
    """α(t) for a from-rest ramp drive ε(t) = slope·t, κ = 0."""
    t = np.arange(n)
    lam = 1j * delta
    particular = -(1j * slope / lam) * t + 1j * slope / lam**2
    return particular - (1j * slope / lam**2) * np.exp(-lam * t)


def _reference_branch(epsilon, branch, **params):
    """High-accuracy α(t) for one branch from the repo's rate function."""
    epsilon = np.asarray(epsilon, dtype=complex)
    n = len(epsilon)
    t = np.arange(n)
    env = interp1d(t, epsilon, kind="cubic", fill_value="extrapolate")
    pick = 0 if branch == "g" else 1

    def rhs(time, alpha):
        rates = conditional_cavity_rates(alpha[0], alpha[0], env(time), **params)
        return [rates[pick]]

    return solve_ivp(
        rhs, (0, n - 1), [0j], method="DOP853", t_eval=t, rtol=1e-12, atol=1e-12,
    ).y[0]


@pytest.mark.parametrize("chi_khz", CHI_KHZ_VALUES)
def test_matches_exact_linear_counter_rotating_branches(chi_khz):
    # zero drive: ground and excited rotate at -χ/2 and +χ/2
    chi = _chi_rad_ns(chi_khz)
    n = 300
    a0 = 1 - 0.5j
    t = np.arange(n)
    alpha_g, alpha_e = alpha_from_epsilon_finite_difference(
        np.zeros(n), delta=chi / 2, chi=chi, alpha_g_init=a0, alpha_e_init=a0,
    )
    assert np.abs(alpha_g - a0 * np.exp(-1j * (chi / 2) * t)).max() < 1e-3
    assert np.abs(alpha_e - a0 * np.exp(+1j * (chi / 2) * t)).max() < 1e-3


def test_matches_exact_linear_ramp_drive():
    n = 300
    slope = 1e-4
    delta = 0.01
    alpha_g, _ = alpha_from_epsilon_finite_difference(
        slope * np.arange(n), delta=delta,
    )
    exact = _exact_linear_ramp(n, slope, delta)
    assert np.abs(alpha_g - exact).max() < 5e-4


@pytest.mark.parametrize("chi_khz", CHI_KHZ_VALUES)
def test_matches_reference_nonlinear_over_chi(chi_khz):
    chi = _chi_rad_ns(chi_khz)
    t = np.arange(600)
    epsilon = 0.02 * np.exp(-((t - 300) / 80) ** 2) * np.exp(0.3j)
    params = dict(
        delta=chi / 2, chi=chi,
        chi_prime=CHI_PRIME_RAD_NS, self_kerr=SELF_KERR_RAD_NS,
    )
    alpha_g, alpha_e = alpha_from_epsilon_finite_difference(epsilon, **params)
    assert np.abs(alpha_g - _reference_branch(epsilon, "g", **params)).max() < 1e-3
    assert np.abs(alpha_e - _reference_branch(epsilon, "e", **params)).max() < 1e-3


def test_seeding_is_first_order_at_nonzero_initial():
    # Seeding α[0] = α[1] = α_init leaves a first-order error |δ·α_init| at the
    # seeded sample when restarting from a nonzero amplitude.
    delta = 0.02
    a0 = 2 + 0j
    alpha_g, _ = alpha_from_epsilon_finite_difference(
        np.zeros(200), delta=delta, alpha_g_init=a0,
    )
    exact = _exact_linear_decay(200, delta=delta, kappa=0, a0=a0)
    assert alpha_g[0] == a0 and alpha_g[1] == a0
    assert abs(alpha_g[1] - exact[1]) == pytest.approx(abs(delta * a0), rel=1e-2)


def test_kappa_zero_long_trajectory_stays_bounded():
    # κ = 0 (the default) is stable: |α| is conserved with a bounded ripple.
    alpha_g, _ = alpha_from_epsilon_finite_difference(
        np.zeros(6000), delta=0.01, kappa=0.0, alpha_g_init=1 + 0j,
    )
    assert np.abs(alpha_g).max() < 1.02
    assert np.abs(alpha_g).min() > 0.98


def test_kappa_positive_long_trajectory_diverges_known_limitation():
    # Known limitation: for κ > 0 the leapfrog scheme grows like ~e^{κt/2}.
    # It stays accurate while κ·t is small and diverges once κ·t is large
    # (here κ·t ≈ 12). Never hit in the pipeline (kappa_kHz defaults to 0).
    n = 6000
    kappa = 2e-3
    alpha_g, _ = alpha_from_epsilon_finite_difference(
        np.zeros(n), delta=0.01, kappa=kappa, alpha_g_init=1 + 0j,
    )
    exact = _exact_linear_decay(n, delta=0.01, kappa=kappa, a0=1 + 0j)
    assert abs(exact[-1]) < 0.05
    assert abs(alpha_g[-1]) > 1.0
