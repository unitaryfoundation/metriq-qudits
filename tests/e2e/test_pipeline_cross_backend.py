"""End-to-end pipeline test on a non-trivial circuit via backend agreement.

Hand-builds one compiled circuit (substituting the optimizer), then runs the REAL
build_circuit_pulses -> run_noiseless stages with the two independent
displaced-frame backends (QuTiP and dynamiqs) and asserts the per-circuit metrics
agree. Cross-backend agreement is the correctness anchor for a real ECD+R circuit.
"""

import numpy as np

from metriq_qudits.benchmark.circuit_io import CompiledCircuit, save_circuits
from metriq_qudits.pulses.pulse_stage import build_circuit_pulses
from metriq_qudits.simulation.sweep import run_noiseless

_METRICS = ("hog", "xeb", "fid")


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


def _run(monkeypatch, out_dir, backend, pulse_path, compiled_path):
    monkeypatch.setenv("METRIQ_QUDITS_OUTPUT_DIR", str(out_dir))
    out = run_noiseless(pulse_path, compiled_path, backend=backend)
    with np.load(out) as res:
        return {m: np.asarray(res[f"{m}_per_circuit"]) for m in _METRICS}


def test_qutip_and_dynamiqs_pipelines_agree(tmp_path, monkeypatch):
    d, depth, n_cav = 4, 4, 20

    # Pulses depend only on the compiled circuit, so build them once (shared dir).
    monkeypatch.setenv("METRIQ_QUDITS_OUTPUT_DIR", str(tmp_path / "build"))
    compiled_path = str(tmp_path / "compiled.npz")
    meta = {"d": d, "k": depth, "num_modes": 1, "N_cav": n_cav,
            "N_unitaries": 1, "seed": 42}
    save_circuits([_nontrivial_circuit(d, depth)], meta, compiled_path)
    pulse_path = build_circuit_pulses(compiled_path, n_jobs=1)

    qutip = _run(monkeypatch, tmp_path / "q", "qutip", pulse_path, compiled_path)
    dynamiqs = _run(monkeypatch, tmp_path / "dq", "dynamiqs", pulse_path, compiled_path)

    for metric in _METRICS:
        np.testing.assert_allclose(qutip[metric], dynamiqs[metric], atol=1e-2)
