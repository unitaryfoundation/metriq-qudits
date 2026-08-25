"""Smoke tests for the cache-driven result plots in metriq_qudits.plotting.utils.

These check that each plot consumes the cached NPZ layout and writes a
non-empty figure; they do not assert on pixel content.
"""

import matplotlib

matplotlib.use("Agg")

import numpy as np

from metriq_qudits.plotting.utils import (
    plot_compile_summary,
    plot_min_depth_curve,
    plot_stability_calibration,
    plot_t1_sweep,
    plot_t2_sweep,
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


def _sweep_record(d, T1=(5, 10, 20, 50, 100), T2=(10, 20, 40, 100, 200), ceiling=True):
    """A noise-sweep record shaped like plot_results._load_noise_sweeps output."""
    rng = np.random.default_rng(d)
    T1, T2 = np.array(T1, float), np.array(T2, float)
    shape = (len(T1), len(T2))
    rec = {
        "d": d,
        "T1_us": T1,
        "T2_us": T2,
        "N_u": 25,
        "hog": rng.uniform(0.6, 0.8, shape),
        "xeb": rng.uniform(0.7, 0.98, shape),
        "fid": rng.uniform(0.7, 0.99, shape),
        "hog_std": rng.uniform(0.01, 0.05, shape),
        "xeb_std": rng.uniform(0.01, 0.05, shape),
        "fid_std": rng.uniform(0.01, 0.05, shape),
    }
    if ceiling:
        rec["hog_nl"], rec["xeb_nl"], rec["fid_nl"] = 0.77, 0.97, 0.98
    return rec


def test_plot_t1_sweep_creates_figure(tmp_path):
    out = tmp_path / "t1_sweep.png"
    assert plot_t1_sweep([_sweep_record(4), _sweep_record(6)], str(out)) is True
    assert out.exists() and out.stat().st_size > 0


def test_plot_t1_sweep_without_diagonal_writes_nothing(tmp_path):
    # A grid where no T2 sits within tolerance of 2*T1 has no diagonal points.
    rec = _sweep_record(4, T1=(5, 10), T2=(100, 200), ceiling=False)
    out = tmp_path / "t1_sweep_none.png"
    assert plot_t1_sweep([rec], str(out)) is False
    assert not out.exists()


def test_plot_t2_sweep_writes_one_figure_per_distinct_t1(tmp_path):
    # Default grid: T1=50 (T2 40 vs 100) and T1=100 (T2 40 vs 200) each contrast
    # a low/high T2; the smaller T1 values collapse to a single T2 and are skipped.
    out_dir = tmp_path / "t2"
    n = plot_t2_sweep([_sweep_record(4), _sweep_record(6)], str(out_dir))
    assert n == 2
    assert len(list(out_dir.glob("t2_sweep_T1_*us.png"))) == 2


def test_plot_t2_sweep_single_config(tmp_path):
    out_dir = tmp_path / "t2_single"
    assert plot_t2_sweep([_sweep_record(4)], str(out_dir)) == 2
    assert len(list(out_dir.glob("*.png"))) == 2


def test_plot_t2_sweep_no_distinct_slices_writes_nothing(tmp_path):
    out_dir = tmp_path / "t2_none"
    rec = _sweep_record(4, T1=(5, 10), T2=(10, 20), ceiling=False)
    assert plot_t2_sweep([rec], str(out_dir)) == 0
    assert not list(out_dir.glob("*.png"))
