"""End-to-end pipeline test with a trivial circuit (known answer).

Substitutes only the slow JAX optimizer by hand-building one compiled circuit,
then runs the REAL stages: build_circuit_pulses -> run_noiseless. The betas are
below the ECD threshold, so the cavity stays in vacuum and fidelity to |0> ~ 1.
"""

import numpy as np
import pytest

from metriq_qudits.compilation.circuit_io import CompiledCircuit, save_circuits
from metriq_qudits.pulses.build import build_circuit_pulses
from metriq_qudits.simulation.sweep import run_noiseless


def _trivial_circuit(d, depth=4):
    rng = np.random.default_rng(0)
    betas = np.full((depth, 1), 1e-6 + 0j)  # below ECD threshold -> cavity untouched
    rotations = np.column_stack([rng.uniform(0, np.pi, depth + 1),
                                 rng.uniform(0, 2 * np.pi, depth + 1)])
    target = np.zeros(d, dtype=complex)
    target[0] = 1.0  # prepare Fock |0>
    return CompiledCircuit(
        betas=betas, rotations=rotations, target_state=target,
        infidelity=0.0, boundary_leakage=0.0,
    )


def test_noiseless_pipeline_prepares_vacuum(tmp_path, monkeypatch):
    monkeypatch.setenv("METRIQ_QUDITS_OUTPUT_DIR", str(tmp_path))
    d, depth, n_cav = 4, 4, 16

    compiled_path = str(tmp_path / "compiled.npz")
    meta = {"d": d, "k": depth, "num_modes": 1, "N_cav": n_cav,
            "N_unitaries": 1, "seed": 42}
    save_circuits([_trivial_circuit(d, depth)], meta, compiled_path)

    pulse_path = build_circuit_pulses(compiled_path, n_jobs=1)
    out_path = run_noiseless(pulse_path, compiled_path, backend="qutip")

    with np.load(out_path) as res:
        assert res["fid_mean"] == pytest.approx(1.0, abs=1e-3)
        assert 0.0 <= float(res["hog_mean"]) <= 1.0
        assert float(res["xeb_mean"]) <= 1.0 + 1e-9
        assert res["fid_per_circuit"].shape == (1,)
        assert int(res["d"]) == d
