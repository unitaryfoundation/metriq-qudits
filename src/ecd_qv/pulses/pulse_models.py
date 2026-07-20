from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class ECDPulse:
    """Physical waveform and achieved displacement for one ECD gate."""

    cavity_drive: np.ndarray
    ancilla_drive: np.ndarray
    peak_displacement: float
    wait_time_ns: int
    achieved_beta: complex
    residual_displacement: complex

    def __post_init__(self) -> None:
        if len(self.cavity_drive) != len(self.ancilla_drive):
            raise ValueError("cavity and ancilla waveforms must have equal length")


@dataclass(frozen=True)
class CircuitPulse:
    """Synchronized physical waveforms for one compiled circuit."""

    cavity_drives: tuple[np.ndarray, ...]
    ancilla_drive: np.ndarray
    final_cavity_phases: np.ndarray = field(default_factory=lambda: np.array([]))
    peak_displacements: tuple[float, ...] = ()
    wait_times_ns: tuple[int, ...] = ()
    ancilla_phases: tuple[float, ...] = ()
    achieved_betas: tuple[complex, ...] = ()

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
