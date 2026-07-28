"""Unit tests for metriq_qudits.pulses.conditional_dynamics.

The RK4 integrator solves, per ancilla branch,

    dα/dt = (i·(-δ + K|α|² + s·(χ + χ'|α|²)) - κ/2)·α - i·ε,

with s = 0 for the ground branch and s = 1 for the excited branch.
"""

import numpy as np
import pytest
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d

from metriq_qudits.pulses.conditional_dynamics import (
    DispersiveRates,
    evolve_pair,
    evolve_with_echoes,
)


def test_zero_drive_stays_at_origin():
    g, e = evolve_pair(np.zeros(10), DispersiveRates())
    assert np.allclose(g, 0)
    assert np.allclose(e, 0)


def test_output_length_matches_input():
    g, e = evolve_pair(np.zeros(7), DispersiveRates())
    assert g.shape == (7,) and e.shape == (7,)


def test_pure_drive_is_linear_ramp():
    eps = np.full(200, 0.01 + 0.02j)
    g, e = evolve_pair(eps, DispersiveRates())
    expected = -1j * eps[0] * np.arange(200)
    assert np.max(np.abs(g - expected)) < 1e-12
    assert np.max(np.abs(e - expected)) < 1e-12


def test_detuning_gives_free_rotation():
    g, _ = evolve_pair(np.zeros(500), DispersiveRates(delta=0.01), start=(1.0 + 0j, 0j))
    expected = np.exp(-1j * 0.01 * np.arange(500))
    assert np.max(np.abs(g - expected)) < 1e-9


def test_branches_identical_without_chi():
    eps = 0.05 * np.ones(21)
    g, e = evolve_pair(eps, DispersiveRates())
    np.testing.assert_allclose(g, e)


def test_chi_counter_rotates_the_branches():
    # In the g/e-midpoint frame (delta = chi/2) the ground and excited branches
    # rotate at -chi/2 and +chi/2.
    chi = 2 * np.pi * 32.8e-6
    n, a0 = 300, 1 - 0.5j
    t = np.arange(n)
    g, e = evolve_pair(
        np.zeros(n), DispersiveRates(delta=chi / 2, chi=chi), start=(a0, a0))
    assert np.abs(g - a0 * np.exp(-1j * (chi / 2) * t)).max() < 1e-6
    assert np.abs(e - a0 * np.exp(+1j * (chi / 2) * t)).max() < 1e-6


def test_self_kerr_is_amplitude_dependent_phase():
    kerr, a0, n = 1e-4, 1.5 + 0j, 400
    t = np.arange(n)
    g, _ = evolve_pair(np.zeros(n), DispersiveRates(kerr=kerr), start=(a0, 0j))
    expected = a0 * np.exp(1j * kerr * abs(a0) ** 2 * t)
    assert np.abs(g - expected).max() < 1e-6


def test_kappa_decays_stably_over_long_trajectory():
    # RK4 damps correctly for kappa > 0 over long times (the leapfrog scheme it
    # replaces grew like e^{+kappa t/2} here).
    n, kappa = 6000, 2e-3
    g, _ = evolve_pair(
        np.zeros(n), DispersiveRates(delta=0.01, kappa=kappa), start=(1 + 0j, 0j))
    exact = np.exp(-(1j * 0.01 + kappa / 2) * np.arange(n))
    assert np.abs(g - exact).max() < 1e-3


def _reference_branch(epsilon, s, rates):
    """High-accuracy α(t) for one branch, integrated with DOP853."""
    epsilon = np.asarray(epsilon, dtype=complex)
    n = len(epsilon)
    t = np.arange(n)
    env = interp1d(t, epsilon, kind="cubic", fill_value="extrapolate")

    def rhs(time, a):
        alpha = a[0]
        photons = abs(alpha) ** 2
        pull = -rates.delta + rates.kerr * photons + s * (
            rates.chi + rates.chi_prime * photons)
        return [(1j * pull - rates.kappa / 2) * alpha - 1j * env(time)]

    return solve_ivp(
        rhs, (0, n - 1), [0j], method="DOP853", t_eval=t, rtol=1e-12, atol=1e-12
    ).y[0]


def test_matches_high_accuracy_reference_nonlinear():
    chi = 2 * np.pi * 32.8e-6
    rates = DispersiveRates(
        delta=chi / 2, chi=chi,
        chi_prime=2 * np.pi * 3.0e-9, kerr=2 * np.pi * 1.0e-9)
    t = np.arange(600)
    eps = 0.02 * np.exp(-((t - 300) / 80) ** 2) * np.exp(0.3j)
    g, e = evolve_pair(eps, rates)
    assert np.abs(g - _reference_branch(eps, 0, rates)).max() < 1e-4
    assert np.abs(e - _reference_branch(eps, 1, rates)).max() < 1e-4


def test_evolve_with_echoes_cancels_differential_phase():
    # A single mid-way echo swaps the g/e labels, so the conditional phase built
    # in the first half is undone in the second: the branches reconverge instead
    # of diverging. (chi is exaggerated to make the effect visible.)
    chi = 0.01
    rates = DispersiveRates(delta=chi / 2, chi=chi)
    n, a0 = 200, 1 + 0j
    no_echo_g, no_echo_e = evolve_pair(np.zeros(n), rates, start=(a0, a0))
    echo_g, echo_e = evolve_with_echoes(np.zeros(n), [n // 2], rates, start=(a0, a0))
    # before the echo the echoed run tracks the plain one
    np.testing.assert_allclose(echo_g[: n // 2], no_echo_g[: n // 2])
    # without an echo the branches diverge in phase; the echo brings them back
    assert np.abs(no_echo_g[-1] - no_echo_e[-1]) > 1.0
    assert np.abs(echo_g[-1] - echo_e[-1]) < 0.05
