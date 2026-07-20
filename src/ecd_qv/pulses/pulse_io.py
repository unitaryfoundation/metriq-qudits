from __future__ import annotations

import numpy as np

from ecd_qv.pulses.pulse_models import CircuitPulse


def save_pulses(path: str, pulses: list[CircuitPulse], metadata: dict) -> None:
    """Write variable-length circuit-pulse waveforms as an NPZ cache."""
    count = len(pulses)
    num_modes = pulses[0].num_modes if pulses else int(metadata["num_modes"])
    if any(pulse.num_modes != num_modes for pulse in pulses):
        raise ValueError("all circuit pulses must have the same number of modes")

    epsilon = np.empty((count, num_modes), dtype=object)
    for circuit_index, pulse in enumerate(pulses):
        mode_drives = pulse.cavity_drives
        for mode_index, drive in enumerate(mode_drives):
            epsilon[circuit_index, mode_index] = drive

    ancilla_drive = np.empty(count, dtype=object)
    ancilla_drive[:] = [pulse.ancilla_drive for pulse in pulses]
    phases = (np.stack([pulse.final_cavity_phases for pulse in pulses])
              if pulses else np.empty((0, num_modes)))
    peaks = np.array([pulse.peak_displacement for pulse in pulses])

    metadata = dict(metadata)
    metadata["N_unitaries"] = count
    metadata["num_modes"] = num_modes
    np.savez(
        path,
        **metadata,
        epsilon=epsilon,
        ancilla_drive=ancilla_drive,
        final_cavity_phase=phases,
        peak_alpha=peaks,
    )


def load_pulses(path: str) -> tuple[list[CircuitPulse], dict]:
    """Load pulse caches into ordinary circuit-pulse lists."""
    data = np.load(path, allow_pickle=True)
    count = int(data["N_unitaries"])
    num_modes = int(data["num_modes"])
    cavity_drives = [
        tuple(data["epsilon"][i, j] for j in range(num_modes))
        for i in range(count)
    ]
    phases = data["final_cavity_phase"]
    peaks = data["peak_alpha"]
    ancilla_drives = [data["ancilla_drive"][i] for i in range(count)]

    pulses = [
        CircuitPulse(
            cavity_drives=cavity_drives[i],
            ancilla_drive=ancilla_drives[i],
            final_cavity_phases=np.asarray(phases[i]),
            # Only the circuit-level maximum displacement is stored.
            peak_displacements=(float(peaks[i]),),
        )
        for i in range(count)
    ]
    array_keys = {"epsilon", "ancilla_drive", "final_cavity_phase", "peak_alpha"}
    metadata = {key: data[key].item() for key in data.files
                if key not in array_keys and data[key].ndim == 0}
    return pulses, metadata
