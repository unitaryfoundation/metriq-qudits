from __future__ import annotations

from dataclasses import dataclass

import numpy as np

_ARRAY_KEYS = {"betas", "rotations", "q", "psi", "boundary_leakage", "err",
               "k_per_circuit", "opt_trace", "k_sweep"}


@dataclass(frozen=True)
class CompiledCircuit:
    """Optimized circuit parameters and the target they prepare."""

    betas: np.ndarray              # complex (depth, num_modes)
    rotations: np.ndarray          # (depth*num_modes + 1, 2) [theta, phi]
    target_state: np.ndarray       # target amplitudes, shape (d**num_modes,)
    infidelity: float              # 1 - |<target|prepared>|
    boundary_leakage: float        # maximum top-Fock-level population
    optimization_trace: np.ndarray | None = None  # best objective per check-in
    depth_sweep: np.ndarray | None = None  # rows: [depth, best infidelity]

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


def save_circuits(circuits: list[CompiledCircuit], config: dict, path: str) -> None:
    num_modes = config["num_modes"]
    k_per     = np.array([c.depth for c in circuits], dtype=np.int32)
    k_max     = int(k_per.max())

    betas     = np.zeros((len(circuits), k_max, num_modes), dtype=complex)
    rotations = np.zeros((len(circuits), k_max * num_modes + 1, 2))
    for i, c in enumerate(circuits):
        betas[i, :c.depth] = c.betas
        rotations[i, :c.depth * num_modes + 1] = c.rotations

    n_ckpt = max((len(c.optimization_trace) for c in circuits
                  if c.optimization_trace is not None), default=0)
    opt_trace = np.full((len(circuits), n_ckpt), np.nan)
    for i, c in enumerate(circuits):
        if c.optimization_trace is not None:
            opt_trace[i, :len(c.optimization_trace)] = c.optimization_trace

    n_k = max((len(c.depth_sweep) for c in circuits
               if c.depth_sweep is not None), default=0)
    k_sweep = np.full((len(circuits), n_k, 2), np.nan)
    for i, c in enumerate(circuits):
        if c.depth_sweep is not None:
            k_sweep[i, :len(c.depth_sweep)] = c.depth_sweep

    np.savez(
        path,
        **config,
        betas=betas,
        rotations=rotations,
        q=np.array([c.target_probabilities for c in circuits]),
        psi=np.array([c.target_state for c in circuits], dtype=complex),
        boundary_leakage=np.array([c.boundary_leakage for c in circuits]),
        err=np.array([c.infidelity for c in circuits]),
        k_per_circuit=k_per,
        opt_trace=opt_trace,
        k_sweep=k_sweep,
    )


def load_circuits(path: str) -> tuple[list[CompiledCircuit], dict]:
    data      = np.load(path, allow_pickle=False)
    num_modes = int(data["num_modes"].item())

    def _opt_trace(i):
        row = data["opt_trace"][i]
        return row[~np.isnan(row)]

    def _k_sweep(i):
        rows = data["k_sweep"][i]
        return rows[~np.isnan(rows[:, 0])]

    circuits = [
        CompiledCircuit(
            betas=data["betas"][i, :ki],
            rotations=data["rotations"][i, :ki * num_modes + 1],
            target_state=data["psi"][i],
            boundary_leakage=float(data["boundary_leakage"][i]),
            infidelity=float(data["err"][i]),
            optimization_trace=_opt_trace(i),
            depth_sweep=_k_sweep(i),
        )
        for i, ki in enumerate(data["k_per_circuit"])
    ]
    config = {key: data[key].item() for key in data.files if key not in _ARRAY_KEYS}
    return circuits, config
