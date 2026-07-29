"""Tests whether the noise model reproduces analytic T1, T2, and cavity decay.
"""

import numpy as np
import pytest
import qutip as qt

from metriq_qudits.pulses.drive_envelopes import CavityMode
from metriq_qudits.simulation.displaced_frame import DisplacedFrameSimulator

N_CAV = 16
T_NS = 2000
_ZERO = np.zeros(T_NS + 1)  # no drive: displaced frame is the physical frame


def _mode(kappa_kHz=0.0):
    return CavityMode(chi_kHz=32.8, chi_prime_Hz=3.0, kerr_Hz=1.0, kappa_kHz=kappa_kHz)


def _evolve(sim, psi0, **noise):
    result, alpha = sim.simulate(_ZERO, _ZERO, psi0=psi0, **noise)
    return sim.to_physical_frame(result.states[-1], alpha)


def test_t1_decay_matches_exponential():
    T1_us = 10.0
    sim = DisplacedFrameSimulator(cavity_dim=N_CAV, mode=_mode(), qubit_dim=3)
    psi0 = qt.tensor(qt.basis(N_CAV, 0), qt.basis(3, 1))  # excited qubit
    rho = _evolve(sim, psi0, T1_us=T1_us)
    p_excited = float(rho.ptrace([1]).full()[1, 1].real)
    assert p_excited == pytest.approx(np.exp(-T_NS / (T1_us * 1e3)), abs=1e-3)


def test_t2_coherence_decay_matches_exponential():
    T1_us, T2_us = 10.0, 12.0
    sim = DisplacedFrameSimulator(cavity_dim=N_CAV, mode=_mode(), qubit_dim=3)
    psi0 = qt.tensor(qt.basis(N_CAV, 0), (qt.basis(3, 0) + qt.basis(3, 1)).unit())
    rho = _evolve(sim, psi0, T1_us=T1_us, T2_us=T2_us)
    coherence = abs(rho.ptrace([1]).full()[0, 1])
    assert coherence == pytest.approx(0.5 * np.exp(-T_NS / (T2_us * 1e3)), abs=1e-3)


def test_cavity_photon_loss_matches_exponential():
    mode = _mode(kappa_kHz=5.0)
    kappa = mode.angular_rates()[3]
    sim = DisplacedFrameSimulator(cavity_dim=N_CAV, mode=mode, qubit_dim=3)
    gamma = 2.0
    psi0 = qt.tensor(qt.coherent(N_CAV, gamma), qt.basis(3, 0))
    rho = _evolve(sim, psi0)
    n_final = float((rho.ptrace([0]) * qt.num(N_CAV)).tr().real)
    assert n_final == pytest.approx(gamma**2 * np.exp(-kappa * T_NS), abs=2e-3)
