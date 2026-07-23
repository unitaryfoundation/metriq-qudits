import json

import numpy as np
import pytest

from metriq_qudits.benchmark.circuit_io import CompiledCircuit
from metriq_qudits.benchmark.ir import (
    SCHEMA_VERSION,
    compiled_circuit_to_ir,
    save_ir,
)


def _circuit():
    """A depth-2 single-mode circuit with known betas and rotations."""
    betas = np.array([[0.5 + 0.1j], [-0.2 + 0.3j]])
    rotations = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
    psi = np.zeros(3, dtype=complex)
    psi[0] = 1.0
    return CompiledCircuit(
        betas=betas,
        rotations=rotations,
        target_state=psi,
        infidelity=1e-3,
        boundary_leakage=1e-4,
    )


def test_schema_and_meta():
    ir = compiled_circuit_to_ir(_circuit(), n_cav=7)
    assert ir["schema"] == SCHEMA_VERSION
    assert ir["meta"] == {"d": 3, "depth": 2, "N_cav": 7}


def test_n_cav_omitted_when_not_given():
    ir = compiled_circuit_to_ir(_circuit())
    assert "N_cav" not in ir["meta"]


def test_sequence_ordering_and_values():
    ir = compiled_circuit_to_ir(_circuit())
    seq = ir["sequence"]

    # depth ECD gates and depth + 1 rotations, starting and ending with R.
    assert [g["op"] for g in seq] == ["R", "ECD", "R", "ECD", "R"]

    assert seq[0] == {"op": "R", "theta": 0.1, "phi": 0.2}
    assert seq[1] == {"op": "ECD", "beta": {"re": 0.5, "im": 0.1}}
    assert seq[3] == {"op": "ECD", "beta": {"re": -0.2, "im": 0.3}}
    assert seq[4] == {"op": "R", "theta": 0.5, "phi": 0.6}


def test_json_roundtrip():
    ir = compiled_circuit_to_ir(_circuit(), n_cav=7)
    assert json.loads(json.dumps(ir)) == ir


def test_save_ir(tmp_path):
    ir = compiled_circuit_to_ir(_circuit(), n_cav=7)
    path = tmp_path / "decomp.json"
    save_ir(ir, str(path))
    assert json.loads(path.read_text()) == ir


def test_rejects_multimode():
    betas = np.array([[0.5 + 0.1j, 0.0], [-0.2 + 0.3j, 0.0]])
    rotations = np.zeros((5, 2))
    circuit = CompiledCircuit(
        betas=betas,
        rotations=rotations,
        target_state=np.ones(4, dtype=complex) / 2.0,
        infidelity=1e-3,
        boundary_leakage=1e-4,
    )
    with pytest.raises(ValueError):
        compiled_circuit_to_ir(circuit)
