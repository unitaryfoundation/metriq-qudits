"""Tests whether displaced-frame and rotating-frame simulators agree.
"""

import numpy as np
import pytest
import qutip as qt

from metriq_qudits.pulses.pulse_stage import make_modes
from metriq_qudits.simulation.displaced_frame_simulator import DisplacedFrameSimulator
from metriq_qudits.simulation.rotating_frame_simulator import RotatingFrameSimulator

N_DISP = 20  # displaced frame stays near vacuum
N_ROT = 30   # rotating frame holds the full displacement
_T = np.arange(250)


def _gauss(center, width, area):
    g = np.exp(-((_T - center) / width) ** 2)
    return area * g / g.sum()


# (id, omega qubit drive, epsilon cavity drive)
_PULSES = [
    ("half_pi", _gauss(50, 12, np.pi / 2) * np.exp(0j), _gauss(150, 20, 1.8) * np.exp(0.4j)),
    ("pi", _gauss(50, 12, np.pi) * np.exp(0.5j), _gauss(150, 20, 1.6) * np.exp(-0.7j)),
    ("third_pi", _gauss(60, 12, np.pi / 3) * np.exp(1j), _gauss(160, 22, 2.2) * np.exp(0.2j)),
]


def _final_cavity(sim, epsilon, omega):
    result, alpha = sim.simulate(epsilon, omega)
    physical = sim.to_physical_frame(result.states[-1], alpha, cavity_phase=0.0)
    return physical.ptrace([0])


def _renormalized_block(cavity, m):
    block = np.asarray(cavity.full())[:m, :m]
    return qt.Qobj(block / np.trace(block))


@pytest.mark.parametrize(
    "omega, epsilon", [(p[1], p[2]) for p in _PULSES], ids=[p[0] for p in _PULSES],
)
def test_displaced_and_rotating_frames_agree(omega, epsilon):
    mode = make_modes(1)[0]
    disp = _final_cavity(
        DisplacedFrameSimulator(cavity_dim=N_DISP, mode=mode, qubit_dim=3),
        epsilon, omega,
    )
    rot = _final_cavity(
        RotatingFrameSimulator(cavity_dim=N_ROT, mode=mode, qubit_dim=3),
        epsilon, omega,
    )

    pops_d = np.real(np.diag(disp.full()))
    pops_r = np.real(np.diag(rot.full()))
    # both must stay off the top Fock levels, else the comparison is truncation-limited
    assert pops_d[-3:].sum() < 1e-3
    assert pops_r[-3:].sum() < 1e-3
    assert np.max(np.abs(pops_d[:12] - pops_r[:12])) < 1e-4

    m = min(N_DISP, N_ROT)
    fidelity = qt.fidelity(_renormalized_block(disp, m), _renormalized_block(rot, m))
    assert fidelity > 0.999
