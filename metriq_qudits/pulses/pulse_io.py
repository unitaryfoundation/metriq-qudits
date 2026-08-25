from __future__ import annotations

import numpy as np

from metriq_qudits.pulses.ecd_pulse_builder import CircuitWaveforms


def save_pulse(path: str, pulse: CircuitWaveforms) -> None:
    """Write one circuit's waveforms to its own npz.

    Within a single circuit every drive has the same length, so cavity_drives
    stacks into a regular (num_modes, length) array with no object arrays."""
    np.savez(
        path,
        cavity_drives=np.stack(pulse.cavity_drives),
        ancilla_drive=pulse.ancilla_drive,
        final_cavity_phases=pulse.final_cavity_phases,
        peak_displacements=np.asarray(pulse.peak_displacements, dtype=float),
    )


def load_pulse(path: str) -> CircuitWaveforms:
    """Load one circuit's waveforms written by save_pulse."""
    with np.load(path, allow_pickle=False) as data:
        return CircuitWaveforms(
            cavity_drives=tuple(data["cavity_drives"]),
            ancilla_drive=data["ancilla_drive"],
            final_cavity_phases=data["final_cavity_phases"],
            peak_displacements=tuple(data["peak_displacements"].tolist()),
        )


def save_pulses(paths, pulses: list[CircuitWaveforms | None]) -> None:
    """Save each pulse to its own path, skipping circuits that produced none."""
    for path, pulse in zip(paths, pulses):
        if pulse is not None:
            save_pulse(str(path), pulse)


def load_pulses(paths) -> list[CircuitWaveforms | None]:
    """Load pulses by path, kept index-aligned (None where a file is missing)."""
    return [load_pulse(str(path)) if path.exists() else None for path in paths]
