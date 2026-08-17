from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CompiledCircuit:
    """Optimized circuit parameters and the target state they prepare."""

    betas: np.ndarray              
    rotations: np.ndarray          
    target_state: np.ndarray       
    infidelity: float             
    boundary_leakage: float # maximum top-Fock-level population throughout entire circuit execution
    optimization_trace: np.ndarray | None = None  # best opt cost per check-in (check-in is performed every `check_every' steps)
    depth_sweep: np.ndarray | None = None  # rows: [depth, best infidelity] (here best infidelity refers to best over all adam branches at the final step)

    def __post_init__(self) -> None:
        if self.betas.ndim != 2:
            raise ValueError("betas must have shape (depth, num_modes)")
        expected_rotations = (self.depth * self.num_modes + 1, 2)
        if self.rotations.shape != expected_rotations:
            raise ValueError(f"rotations must have shape {expected_rotations}")
        if self.target_state.ndim != 1:
            raise ValueError("target_state must be one-dimensional")

    @property
    def depth(self) -> int:
        return self.betas.shape[0]

    @property
    def num_modes(self) -> int:
        return self.betas.shape[1]

    @property
    def target_probabilities(self) -> np.ndarray:
        return np.abs(self.target_state) ** 2


def save_circuit(path: str, circuit: CompiledCircuit) -> None:
    """Write one compiled circuit to its own npz.

    Per-unitary files need no batch padding: a single circuit's arrays are all
    regular, so this is a plain dump of its fields."""
    arrays = {
        "betas": circuit.betas,
        "rotations": circuit.rotations,
        "target_state": circuit.target_state,
        "infidelity": circuit.infidelity,
        "boundary_leakage": circuit.boundary_leakage,
    }
    if circuit.optimization_trace is not None:
        arrays["optimization_trace"] = circuit.optimization_trace
    if circuit.depth_sweep is not None:
        arrays["depth_sweep"] = circuit.depth_sweep
    np.savez(path, **arrays)


def load_circuit(path: str) -> CompiledCircuit:
    """Load one compiled circuit written by save_circuit."""
    with np.load(path, allow_pickle=False) as data:
        return CompiledCircuit(
            betas=data["betas"],
            rotations=data["rotations"],
            target_state=data["target_state"],
            infidelity=float(data["infidelity"]),
            boundary_leakage=float(data["boundary_leakage"]),
            optimization_trace=(
                data["optimization_trace"]
                if "optimization_trace" in data.files else None
            ),
            depth_sweep=(
                data["depth_sweep"] if "depth_sweep" in data.files else None
            ),
        )


def save_circuits(paths, circuits: list[CompiledCircuit | None]) -> None:
    """Save each compiled circuit to its own path, skipping targets that did not converge."""
    for path, circuit in zip(paths, circuits):
        if circuit is not None:
            save_circuit(str(path), circuit)


def load_circuits(paths) -> list[CompiledCircuit | None]:
    """Load circuits by path, kept index-aligned (None where a file is missing)."""
    return [load_circuit(str(path)) if path.exists() else None for path in paths]
