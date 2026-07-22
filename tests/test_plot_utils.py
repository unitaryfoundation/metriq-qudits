"""Smoke tests for the cache-driven result plots in metriq_qudits.plot_utils.

These check that each plot consumes the cached NPZ layout and writes a
non-empty figure; they do not assert on pixel content.
"""

import matplotlib

matplotlib.use("Agg")

import numpy as np

from metriq_qudits.plot_utils import plot_compile_summary


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
