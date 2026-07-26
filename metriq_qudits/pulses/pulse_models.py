from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class ECDPulse:
    """Physical waveform and achieved displacement for one ECD gate."""

    cavity_drive: np.ndarray        # complex ε(t), one sample per ns [rad/ns]
    ancilla_drive: np.ndarray       # complex Ω(t), same length as cavity_drive [rad/ns]
    peak_displacement: float        # max |α| reached mid-gate (peak photons ≈ |α|²) [dimensionless]
    wait_time_ns: int               # idle time between displacement sub-pulses [ns]
    achieved_beta: complex          # realized conditional displacement β [dimensionless]
    residual_displacement: complex  # leftover state-independent displacement λ (ideally ≈ 0)

    def __post_init__(self) -> None:
        if len(self.cavity_drive) != len(self.ancilla_drive):
            raise ValueError("cavity and ancilla waveforms must have equal length")


@dataclass(frozen=True)
class CircuitPulse:
    """Synchronized physical waveforms for one compiled circuit."""

    cavity_drives: tuple[np.ndarray, ...]   # complex cavity drive ε(t) [rad/ns]
    ancilla_drive: np.ndarray               # shared complex Ω(t), matches ε(t) length [rad/ns]
    # spurious cavity phase to undo when returning to the physical frame [rad]
    final_cavity_phases: np.ndarray = field(default_factory=lambda: np.array([]))
    peak_displacements: tuple[float, ...] = ()  # peak |α| reached in each ECD gate [dimensionless]
    wait_times_ns: tuple[int, ...] = ()         # idle time in each ECD gate [ns]
    ancilla_phases: tuple[float, ...] = ()      # accumulated virtual-Z transmon phase from each ECD gate [rad]
    achieved_betas: tuple[complex, ...] = ()    # realized conditional displacement β of each ECD gate [dimensionless]

    def __post_init__(self) -> None:
        lengths = {len(self.ancilla_drive), *(len(x) for x in self.cavity_drives)}
        if len(lengths) != 1:
            raise ValueError("all circuit drive arrays must have the same length")
        if self.final_cavity_phases.size not in {0, len(self.cavity_drives)}:
            raise ValueError("one final cavity phase is required per mode")

    @property
    def num_modes(self) -> int:
        return len(self.cavity_drives)

    @property
    def peak_displacement(self) -> float:
        return max(self.peak_displacements, default=0.0)
