"""Tests whether transmon leakage falls as the anharmonicity |K_q| grows.
"""

import numpy as np
import pytest

from metriq_qudits.pulses.ecd_pulse_builder import ECDPulseBuilder
from metriq_qudits.pulses.build import make_ancilla, make_modes
from metriq_qudits.simulation.displaced_frame_dynamiqs import DisplacedFrameSimulatorDQ

pytestmark = pytest.mark.slow

N_CAV, DEPTH = 16, 6
KQ_MHZ = [0.0, -100.0, -200.0, -400.0, -800.0]  # increasing anharmonicity


def _random_pulse(seed=0):
    rng = np.random.default_rng(seed)
    betas = (rng.normal(size=(DEPTH, 1)) + 1j * rng.normal(size=(DEPTH, 1))) * 0.7
    rotations = np.column_stack([
        rng.uniform(0, np.pi, DEPTH + 1), rng.uniform(0, 2 * np.pi, DEPTH + 1),
    ])
    mode = make_modes(1)[0]
    pulse = ECDPulseBuilder([mode], make_ancilla()).compile_circuit(
        betas=betas, rotations=rotations, peak_amplitude=10, correct_cavity_phases=True,
    )
    return mode, pulse


def test_leakage_falls_with_anharmonicity():
    mode, pulse = _random_pulse()
    leakages, states = [], []
    for kq in KQ_MHZ:
        sim = DisplacedFrameSimulatorDQ(cavity_dim=N_CAV, mode=mode, K_q_MHz=kq)
        result, alpha = sim.simulate(
            epsilon=pulse.cavity_drives[0], omega=pulse.ancilla_drive,
            T1_us=None, T2_us=None,
        )
        final = sim.to_physical_frame(
            result.states[-1], alpha, cavity_phase=pulse.final_cavity_phases[0],
        )
        leakages.append(float(np.real(final.ptrace([1]).full()[2, 2])))  # |f> population
        states.append(np.asarray(final.ptrace([0]).full()))

    leakages = np.array(leakages)
    # as leakage falls, the cavity state converges to the leakage-free limit
    distance = np.array([np.abs(s - states[-1]).max() for s in states])
    assert np.all(np.diff(leakages) < 0)
    assert np.all(np.diff(distance) < 0)
    assert leakages[0] > 0.1     # harmonic transmon leaks heavily
    assert leakages[-1] < 1e-6   # strong anharmonicity suppresses leakage
