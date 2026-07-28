"""Unit tests for the ECD pulse builder stack: drive envelopes, the conditional
cavity integrator, the weak-dispersive analytics, and gate/circuit calibration.
"""

import numpy as np
import pytest

from metriq_qudits.pulses.drive_envelopes import CavityMode, TransmonAncilla
from metriq_qudits.pulses.ecd_pulse_builder import ECDPulseBuilder, detect_echo_centers


def _builder():
    return ECDPulseBuilder(modes=[CavityMode()], ancilla=TransmonAncilla(sigma_ns=10))


def test_ancilla_rotation_zero_at_edges_with_expected_area():
    rot = TransmonAncilla(sigma_ns=10).rotation(angle=np.pi, axis_phase=0.3)
    assert abs(rot[0]) < 1e-12 and abs(rot[-1]) < 1e-12
    # a theta rotation integrates to (theta / 2) exp(-i phi)
    assert np.allclose(np.sum(rot), 0.5 * np.pi * np.exp(-1j * 0.3))


def test_cavity_displacement_reaches_requested_amplitude():
    disp = CavityMode().displacement(1.7 * np.exp(1j * 0.4))
    # a drive eps displaces the cavity to alpha = -i * sum(eps)
    assert np.allclose(-1j * np.sum(disp), 1.7 * np.exp(1j * 0.4))


def test_weak_dispersive_summary_matches_direct_sum():
    builder = _builder()
    rng = np.random.default_rng(7)
    n, echo_at = 400, 173
    eps = (rng.normal(size=n) + 1j * rng.normal(size=n)) * 0.01
    summary = builder.weak_dispersive_summary(eps, echo_samples=[echo_at])

    chi = CavityMode().angular_rates()[0]
    z = np.ones(n)
    z[echo_at:] = -1.0
    phi = -0.5 * chi * np.cumsum(z)
    delta_ref = np.array([
        -np.sum(np.sin(phi[: i + 1] - phi[i]) * eps[: i + 1]) for i in range(n)
    ])
    gamma_ref = np.array([
        -1j * np.sum(np.cos(phi[: i + 1] - phi[i]) * eps[: i + 1]) for i in range(n)
    ])
    theta_ref = -2.0 * np.cumsum(np.real(np.conj(eps) * delta_ref))
    theta_prime_ref = theta_ref[-1] + 2.0 * np.imag(gamma_ref[-1] * delta_ref[-1])

    assert np.max(np.abs(summary.running_conditional - delta_ref)) < 1e-10
    assert np.max(np.abs(summary.running_displacement - gamma_ref)) < 1e-10
    assert abs(summary.theta_prime - theta_prime_ref) < 1e-10
    assert np.allclose(summary.beta, 2 * delta_ref[-1])


def test_gate_calibration_reaches_target_beta():
    gate = _builder().compile_gate(beta=1.0, peak_amplitude=20.0)
    assert abs(gate.achieved_beta - 1.0) <= 1e-3
    # the deterministic displacement closes back to the origin
    assert gate.residual_displacement < 1e-2
    # exactly one echo, at the tracked sample
    detected = detect_echo_centers(gate.ancilla_drive)
    assert len(detected) == 1 and abs(detected[0] - gate.echo_sample) <= 1


def test_circuit_reaches_target_betas_with_synchronized_drives():
    circuit = _builder().compile_circuit(
        betas=np.array([[1.0], [1.5]]),
        rotations=np.array([[np.pi / 2, 0.0], [np.pi / 2, np.pi / 2], [0.0, 0.0]]),
        peak_amplitude=20.0,
    )
    assert len(circuit.cavity_drives[0]) == len(circuit.ancilla_drive)
    for target, got in zip([1.0, 1.5], circuit.achieved_betas):
        assert abs(abs(got) - target) / target < 0.05
