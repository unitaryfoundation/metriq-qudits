"""Unit tests for metriq_qudits.benchmark.circuit_io."""

import numpy as np
import pytest

from metriq_qudits.benchmark.circuit_io import (
    CompiledCircuit,
    load_circuits,
    save_circuits,
)


def _circuit(depth, d=4, trace_len=3, sweep_rows=2):
    rng = np.random.default_rng(depth)
    betas = rng.normal(size=(depth, 1)) + 1j * rng.normal(size=(depth, 1))
    rotations = rng.normal(size=(depth + 1, 2))
    target = rng.normal(size=d) + 1j * rng.normal(size=d)
    target /= np.linalg.norm(target)
    return CompiledCircuit(
        betas=betas,
        rotations=rotations,
        target_state=target,
        infidelity=0.01 * depth,
        boundary_leakage=1e-3 * depth,
        optimization_trace=rng.normal(size=trace_len),
        depth_sweep=np.column_stack(
            [np.arange(1, sweep_rows + 1), rng.normal(size=sweep_rows)]
        ),
    )


def test_properties():
    c = _circuit(2, d=4)
    assert c.depth == 2
    assert c.num_modes == 1
    np.testing.assert_allclose(c.target_probabilities, np.abs(c.target_state) ** 2)


def test_post_init_rejects_non_2d_betas():
    with pytest.raises(ValueError):
        CompiledCircuit(np.zeros(3), np.zeros((2, 2)), np.zeros(4), 0.0, 0.0)


def test_post_init_rejects_wrong_rotation_shape():
    with pytest.raises(ValueError):
        CompiledCircuit(np.zeros((2, 1), complex), np.zeros((5, 2)), np.zeros(4), 0.0, 0.0)


def test_post_init_rejects_non_1d_target():
    with pytest.raises(ValueError):
        CompiledCircuit(np.zeros((2, 1), complex), np.zeros((3, 2)), np.zeros((4, 1)), 0.0, 0.0)


def test_roundtrip_preserves_variable_depth_circuits(tmp_path):
    circuits = [_circuit(2), _circuit(3)]  # different depths exercise the padding
    config = {"num_modes": 1, "d": 4, "N_cav": 20}
    path = str(tmp_path / "circuits.npz")

    save_circuits(circuits, config, path)
    loaded, loaded_config = load_circuits(path)

    assert len(loaded) == 2
    for orig, got in zip(circuits, loaded):
        assert got.depth == orig.depth
        np.testing.assert_allclose(got.betas, orig.betas)
        np.testing.assert_allclose(got.rotations, orig.rotations)
        np.testing.assert_allclose(got.target_state, orig.target_state)
        assert got.infidelity == pytest.approx(orig.infidelity)
        assert got.boundary_leakage == pytest.approx(orig.boundary_leakage)
        np.testing.assert_allclose(got.optimization_trace, orig.optimization_trace)
        np.testing.assert_allclose(got.depth_sweep, orig.depth_sweep)

    assert loaded_config == {"num_modes": 1, "d": 4, "N_cav": 20}
