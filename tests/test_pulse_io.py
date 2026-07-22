"""Unit tests for metriq_qudits.pulses.pulse_io."""

import numpy as np
import pytest

from metriq_qudits.pulses.pulse_io import load_pulses, save_pulses
from metriq_qudits.pulses.pulse_models import CircuitPulse


def _pulse(length, phase, peaks):
    rng = np.random.default_rng(length)
    cav = rng.normal(size=length) + 1j * rng.normal(size=length)
    anc = rng.normal(size=length) + 1j * rng.normal(size=length)
    return CircuitPulse(
        cavity_drives=(cav,),
        ancilla_drive=anc,
        final_cavity_phases=np.array([phase]),
        peak_displacements=peaks,
    )


def test_roundtrip_preserves_variable_length_pulses(tmp_path):
    pulses = [_pulse(5, 0.1, (3.0,)), _pulse(8, 0.2, (5.0, 2.0))]
    metadata = {"num_modes": 1, "d": 4, "N_cav": 20}
    path = str(tmp_path / "pulses.npz")

    save_pulses(path, pulses, metadata)
    loaded, loaded_meta = load_pulses(path)

    assert len(loaded) == 2
    for orig, got in zip(pulses, loaded):
        assert got.num_modes == 1
        np.testing.assert_allclose(got.cavity_drives[0], orig.cavity_drives[0])
        np.testing.assert_allclose(got.ancilla_drive, orig.ancilla_drive)
        np.testing.assert_allclose(got.final_cavity_phases, orig.final_cavity_phases)
        assert got.peak_displacement == pytest.approx(orig.peak_displacement)

    assert loaded_meta["num_modes"] == 1
    assert loaded_meta["d"] == 4
    assert loaded_meta["N_unitaries"] == 2


def test_only_circuit_level_peak_is_persisted(tmp_path):
    # save/load keeps a single per-circuit maximum; the per-gate tuple is lossy
    pulse = _pulse(6, 0.0, (5.0, 2.0, 4.0))
    path = str(tmp_path / "p.npz")

    save_pulses(path, [pulse], {"num_modes": 1})
    (got,), _ = load_pulses(path)

    assert got.peak_displacement == pytest.approx(5.0)
    assert got.peak_displacements == (5.0,)
