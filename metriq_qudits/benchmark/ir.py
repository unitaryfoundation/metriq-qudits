"""Hardware-neutral intermediate representation (IR) for ECD+rotation circuits.

Serializes a :class:`CompiledCircuit` into a provider-neutral, versioned gate
list. This is the hand-off point to a hardware adapter. The IR carries only the
abstract gate parameters, namely the conditional displacement β and the ancilla
rotation angles θ and φ. Waveforms, calibration constants (α_CD, χ, χ′, K), and
the spurious cavity phase corrections all live on the device side, downstream of
this boundary.

Schema ``metriq.ecd_rot.v1``::

    {
      "schema": "metriq.ecd_rot.v1",
      "meta": {"d": int, "depth": int, "N_cav": int | absent},
      "sequence": [
        {"op": "R",   "theta": float, "phi": float},
        {"op": "ECD", "beta": {"re": float, "im": float}},
        ...
      ]
    }

The sequence follows the k-layer ansatz: a leading rotation, then alternating
ECD and rotation, giving ``depth`` ECD gates and ``depth + 1`` rotations.
"""

from __future__ import annotations

import json

import numpy as np

from metriq_qudits.benchmark.circuit_io import CompiledCircuit

SCHEMA_VERSION = "metriq.ecd_rot.v1"


def _rotation_op(row: np.ndarray) -> dict:
    """One ancilla rotation op. theta in [0, π] rad, phi in [0, 2π) rad."""
    return {"op": "R", "theta": float(row[0]), "phi": float(row[1])}


def _ecd_op(beta: complex) -> dict:
    """One ECD op. beta is the conditional displacement as {re, im} [dimensionless]."""
    return {"op": "ECD", "beta": {"re": float(beta.real), "im": float(beta.imag)}}


def compiled_circuit_to_ir(circuit: CompiledCircuit, *, n_cav: int | None = None) -> dict:
    """Serialize a single-qudit compiled circuit to the ``metriq.ecd_rot.v1`` IR.

    Parameters
    ----------
    circuit : CompiledCircuit
        A compiled single-mode ECD circuit. Its ``betas`` (depth, 1) and
        ``rotations`` (depth + 1, 2) supply the gate parameters.
    n_cav : int, optional
        Fock truncation the circuit was compiled in. Recorded under
        ``meta.N_cav`` when given. Informational only, since a device does not
        need it to play the sequence.

    Returns
    -------
    dict
        The hardware-neutral IR, ready for :func:`json.dump`.
    """
    betas = np.asarray(circuit.betas)
    rotations = np.asarray(circuit.rotations)
    if betas.ndim != 2 or betas.shape[1] != 1:
        raise ValueError("IR is defined for a single cavity mode (betas shape (depth, 1))")

    depth = circuit.depth
    d = int(circuit.target_state.shape[0])
    beta_col = betas[:, 0]

    sequence = [_rotation_op(rotations[0])]
    for i in range(depth):
        sequence.append(_ecd_op(complex(beta_col[i])))
        sequence.append(_rotation_op(rotations[i + 1]))

    meta = {"d": d, "depth": int(depth)}
    if n_cav is not None:
        meta["N_cav"] = int(n_cav)
    return {"schema": SCHEMA_VERSION, "meta": meta, "sequence": sequence}


def save_ir(ir: dict, path: str) -> None:
    """Write an IR dict to ``path`` as indented JSON."""
    with open(path, "w") as handle:
        json.dump(ir, handle, indent=2)
