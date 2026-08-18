"""Unit tests for metriq_qudits.pulses.pulse_io."""

import numpy as np
import pytest

from metriq_qudits.pulses.pulse_io import load_pulses, save_pulses
from metriq_qudits.pulses.ecd_pulse_builder import CircuitWaveforms


def _pulse(length, phase, peaks):
    rng = np.random.default_rng(length)
    cav = rng.normal(size=length) + 1j * rng.normal(size=length)
    anc = rng.normal(size=length) + 1j * rng.normal(size=length)
    return CircuitWaveforms(
        cavity_drives=(cav,),
        ancilla_drive=anc,
        final_cavity_phases=np.array([phase]),
        peak_displacements=peaks,
    )


def test_roundtrip_preserves_variable_length_pulses(tmp_path):
    pulses = [_pulse(5, 0.1, (3.0,)), _pulse(8, 0.2, (5.0, 2.0))]
    paths = [tmp_path / f"{i:04d}.npz" for i in range(len(pulses))]

    save_pulses(paths, pulses)
    loaded = load_pulses(paths)

    assert len(loaded) == 2
    for orig, got in zip(pulses, loaded):
        assert got.num_modes == 1
        np.testing.assert_allclose(got.cavity_drives[0], orig.cavity_drives[0])
        np.testing.assert_allclose(got.ancilla_drive, orig.ancilla_drive)
        np.testing.assert_allclose(got.final_cavity_phases, orig.final_cavity_phases)
        assert got.peak_displacement == pytest.approx(orig.peak_displacement)


def test_peak_displacements_roundtrip(tmp_path):
    # the full per-gate peak tuple is preserved; peak_displacement is its maximum
    pulse = _pulse(6, 0.0, (5.0, 2.0, 4.0))
    path = tmp_path / "0000.npz"

    save_pulses([path], [pulse])
    (got,) = load_pulses([path])

    assert got.peak_displacements == (5.0, 2.0, 4.0)
    assert got.peak_displacement == pytest.approx(5.0)
