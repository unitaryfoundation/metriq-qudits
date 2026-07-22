"""End-to-end correctness test for a single ECD gate (dynamiqs backend).

A single ECD from |g>|0> followed by a pi-pulse deterministically prepares a
coherent state with the ancilla disentangled back to |g>. The ideal cavity state
is computed independently by ``run_circuit`` (ideal ECD+R unitaries, no pulses);
the real build_circuit_pulses -> run_noiseless (dynamiqs) pipeline must reproduce
it to high fidelity.
"""

import numpy as np
import pytest

from metriq_qudits.benchmark.circuit_io import CompiledCircuit, save_circuits
from metriq_qudits.compilation.ecd_parameter_finder import run_circuit
from metriq_qudits.pulses.pulse_stage import build_circuit_pulses
from metriq_qudits.simulation.sweep import run_noiseless


def test_dq_pipeline_reproduces_single_ecd_state(tmp_path, monkeypatch):
    monkeypatch.setenv("METRIQ_QUDITS_OUTPUT_DIR", str(tmp_path))
    d, n_cav = 4, 16
    beta = 0.6 + 0.75j  # |beta| ~ 0.96, the median of the compiled ensemble

    betas = np.array([[beta]])                         # depth-1 ECD
    rotations = np.array([[0.0, 0.0], [np.pi, 0.0]])   # R0 = I, then pi-flip -> ancilla |g>

    # Independent ground truth: ideal ECD+R state vector (no pulses, no noise).
    psi_g, _, _ = run_circuit(betas, rotations, n_cav)
    psi_g = np.asarray(psi_g)
    assert np.linalg.norm(psi_g) == pytest.approx(1.0, abs=1e-3)  # ancilla disentangled
    target = psi_g[:d] / np.linalg.norm(psi_g[:d])

    circuit = CompiledCircuit(
        betas=betas, rotations=rotations, target_state=target,
        infidelity=0.0, boundary_leakage=0.0,
    )
    compiled_path = str(tmp_path / "compiled.npz")
    meta = {"d": d, "k": 1, "num_modes": 1, "N_cav": n_cav,
            "N_unitaries": 1, "seed": 42}
    save_circuits([circuit], meta, compiled_path)

    pulse_path = build_circuit_pulses(compiled_path, n_jobs=1)
    out_path = run_noiseless(pulse_path, compiled_path, backend="dynamiqs")

    with np.load(out_path) as res:
        assert float(res["fid_mean"]) > 0.99
