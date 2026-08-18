"""End-to-end pipeline test on a non-trivial circuit via backend agreement.

Hand-builds one compiled circuit (substituting the optimizer), then runs the REAL
build -> simulate stages with the two independent displaced-frame backends (QuTiP
and dynamiqs) and asserts the per-circuit metrics agree. Cross-backend agreement is
the correctness anchor for a real ECD+R circuit.
"""

import numpy as np

from metriq_qudits.benchmarks.helpers import build_circuit_pulse, simulate_circuit
from metriq_qudits.compilation.circuit_io import CompiledCircuit
from metriq_qudits.metrics import eval_circuit


def _nontrivial_circuit(d, depth=4):
    rng = np.random.default_rng(7)
    betas = (rng.normal(size=(depth, 1)) + 1j * rng.normal(size=(depth, 1))) * 0.7
    rotations = np.column_stack([rng.uniform(0, np.pi, depth + 1),
                                 rng.uniform(0, 2 * np.pi, depth + 1)])
    target = rng.normal(size=d) + 1j * rng.normal(size=d)
    target /= np.linalg.norm(target)
    return CompiledCircuit(
        betas=betas, rotations=rotations, target_state=target,
        infidelity=0.0, boundary_leakage=0.0,
    )


def test_qutip_and_dynamiqs_pipelines_agree():
    d, depth, n_cav = 4, 4, 20
    circuit = _nontrivial_circuit(d, depth)
    pulse = build_circuit_pulse(circuit)  # pulses depend only on the compiled circuit
    metrics = {}
    for backend in ("qutip", "dynamiqs"):
        state = simulate_circuit(pulse, n_cavity=n_cav, backend=backend)
        m = eval_circuit(state, circuit.target_state, d, 1, n_cav)
        metrics[backend] = np.array([m.hog, m.xeb, m.fid])
    np.testing.assert_allclose(metrics["qutip"], metrics["dynamiqs"], atol=1e-2)
