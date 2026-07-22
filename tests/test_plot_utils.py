"""Smoke tests for the cache-driven result plots in metriq_qudits.plot_utils.

These check that each plot consumes the cached NPZ layout and writes a
non-empty figure; they do not assert on pixel content.
"""

import matplotlib

matplotlib.use("Agg")

import numpy as np

from metriq_qudits.plot_utils import (
    plot_compile_summary,
    plot_min_depth_curve,
    plot_stability_calibration,
)


def _write_compiled_npz(path, n=6, k_max=4):
    """Write a compiled-circuits NPZ with the fields plot_compile_summary reads."""
    rng = np.random.default_rng(0)
    k_per = rng.integers(3, k_max + 1, size=n).astype(np.int32)
    betas = np.zeros((n, k_max, 1), dtype=complex)  # (circuit, layer, single mode)
    for i, ki in enumerate(k_per):
        betas[i, :ki, 0] = rng.uniform(0.5, 2.0, size=ki)
    opt_trace = np.tile(np.log([1e-1, 1e-2, 5e-3]), (n, 1))  # log-objective per check-in
    np.savez(
        path,
        d=4,
        num_modes=1,
        n_attempted=n + 2,
        trace_stride=100,
        betas=betas,
        err=rng.uniform(1e-3, 1e-2, size=n),
        boundary_leakage=rng.uniform(1e-4, 1e-3, size=n),
        k_per_circuit=k_per,
        opt_trace=opt_trace,
    )


def test_plot_compile_summary_creates_figure(tmp_path):
    npz = tmp_path / "compiled.npz"
    _write_compiled_npz(npz)
    out = tmp_path / "compile.png"
    plot_compile_summary(str(npz), str(out))
    assert out.exists() and out.stat().st_size > 0


def test_plot_compile_summary_without_opt_trace(tmp_path):
    # Exercises the "no opt_trace in cache" branch of the convergence panel.
    npz = tmp_path / "compiled.npz"
    _write_compiled_npz(npz)
    data = dict(np.load(npz))
    data.pop("opt_trace")
    np.savez(npz, **data)
    out = tmp_path / "compile_no_trace.png"
    plot_compile_summary(str(npz), str(out))
    assert out.exists() and out.stat().st_size > 0


def _write_k_sweep_npz(path, n=5):
    """Write a compiled NPZ carrying a per-circuit depth sweep [depth, infidelity]."""
    k_sweep = np.full((n, 3, 2), np.nan)
    for i in range(n):
        k_sweep[i, :, 0] = [3, 4, 5]
        k_sweep[i, :, 1] = [0.05, 0.008, 0.004]
    np.savez(path, d=4, k_sweep=k_sweep)


def test_plot_min_depth_curve_creates_figure(tmp_path):
    npz = tmp_path / "compiled.npz"
    _write_k_sweep_npz(npz)
    out = tmp_path / "min_depth.png"
    assert plot_min_depth_curve(str(npz), str(out)) is True
    assert out.exists() and out.stat().st_size > 0


def test_plot_min_depth_curve_skips_without_k_sweep(tmp_path):
    npz = tmp_path / "compiled.npz"
    np.savez(npz, d=4)  # no k_sweep field
    out = tmp_path / "min_depth.png"
    assert plot_min_depth_curve(str(npz), str(out)) is False
    assert not out.exists()


def _write_cal_npz(path, buffers=(3, 4, 5), n_extra=6, n_circ=4):
    """Write a calibration NPZ with the fields plot_stability_calibration reads."""
    rng = np.random.default_rng(2)
    m = len(buffers)
    curves = rng.uniform(1e-3, 5e-2, size=(m, n_circ, n_extra))
    max_stab = np.nanmax(curves.reshape(m, -1), axis=1)
    np.savez(
        path,
        d=4,
        k=8,
        stability_th=0.01,
        N_test_extra=np.arange(1, n_extra + 1),
        num_buffers_tried=np.array(buffers, dtype=np.int32),
        n_pen_tried=np.array([b // 2 for b in buffers], dtype=np.int32),
        max_stab=max_stab,
        stable=np.array([False] * (m - 1) + [True]),
        curves=curves,
        best_buffers=buffers[-1],
        best_n_pen=buffers[-1] // 2,
    )


def test_plot_stability_calibration_creates_figure(tmp_path):
    npz = tmp_path / "cal.npz"
    _write_cal_npz(npz)
    out = tmp_path / "calibration.png"
    plot_stability_calibration(str(npz), str(out))
    assert out.exists() and out.stat().st_size > 0


def test_plot_stability_calibration_tolerates_failed_buffer(tmp_path):
    # A buffer count where every calibration circuit failed shows up as all-NaN;
    # it must be skipped rather than raising.
    npz = tmp_path / "cal.npz"
    _write_cal_npz(npz)
    data = dict(np.load(npz))
    data["curves"][0] = np.nan
    data["max_stab"][0] = np.nan
    np.savez(npz, **data)
    out = tmp_path / "calibration_failed.png"
    plot_stability_calibration(str(npz), str(out))
    assert out.exists() and out.stat().st_size > 0
