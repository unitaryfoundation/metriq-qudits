"""Generate figures from cached benchmark results."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np

from metriq_qudits.benchmarks.helpers import load_result
from metriq_qudits.benchmarks.quantum_volume import QVResult
from metriq_qudits.compilation.circuit_io import load_circuit
from metriq_qudits.paths import data_dir, plots_dir
from metriq_qudits.plotting.utils import (
    METRICS,
    plot_compile_summary,
    plot_metrics_vs_dim,
    plot_min_depth_curve,
    plot_noise_heatmaps,
    plot_stability_calibration,
    plot_t1_sweep,
    plot_t2_sweep,
)


def _dimension(run_dir: Path) -> int:
    return int(run_dir.name[1:])


def _metric_values(result: QVResult) -> dict[str, float]:
    return {"hog": result.hog_mean, "xeb": result.xeb_normalized, "fid": result.fid_mean}


def _noiseless_record(run_dir: Path, device: str) -> dict | None:
    path = run_dir / "metrics" / device / "noiseless.npz"
    if not path.exists():
        return None
    return {"d": _dimension(run_dir), **_metric_values(load_result(path, QVResult))}


def _sweep_record(run_dir: Path, device: str) -> dict | None:
    sweep_dir = run_dir / "metrics" / device / "sweep"
    if not sweep_dir.exists():
        return None
    points = []
    for path in sorted(sweep_dir.glob("*.npz")):
        with np.load(path) as data:
            t1, t2 = float(data["t1_us"]), float(data["t2_us"])
        points.append((t1, t2, load_result(path, QVResult)))
    if not points:
        return None

    t1_axis = sorted({t1 for t1, _, _ in points})
    t2_axis = sorted({t2 for _, t2, _ in points})
    row = {t1: i for i, t1 in enumerate(t1_axis)}
    col = {t2: i for i, t2 in enumerate(t2_axis)}
    grids = {m: np.full((len(t1_axis), len(t2_axis)), np.nan) for m in METRICS}
    stds = {m: np.full((len(t1_axis), len(t2_axis)), np.nan) for m in METRICS}
    n_u = 1
    for t1, t2, result in points:
        values = _metric_values(result)
        n_u = max(n_u, len(result.hog))
        for m in METRICS:
            grids[m][row[t1], col[t2]] = values[m]
            stds[m][row[t1], col[t2]] = float(np.std(getattr(result, m)))

    record = {"d": _dimension(run_dir), "T1_us": np.array(t1_axis),
              "T2_us": np.array(t2_axis), "N_u": n_u}
    for m in METRICS:
        record[m] = grids[m]
        record[f"{m}_std"] = stds[m]
    return record


def _compile_aggregate(run_dir: Path) -> dict | None:
    paths = sorted((run_dir / "circuits").glob("*.npz"))
    circuits = [load_circuit(str(p)) for p in paths]
    if not circuits:
        return None
    max_k = max(c.depth for c in circuits)
    betas = np.zeros((len(circuits), max_k, circuits[0].num_modes), dtype=complex)
    for i, circuit in enumerate(circuits):
        betas[i, :circuit.depth] = circuit.betas
    aggregate = {
        "err": np.array([c.infidelity for c in circuits]),
        "boundary_leakage": np.array([c.boundary_leakage for c in circuits]),
        "k_per_circuit": np.array([c.depth for c in circuits]),
        "betas": betas,
    }
    sweeps = [c.depth_sweep for c in circuits if c.depth_sweep is not None]
    if sweeps:
        rows = max(s.shape[0] for s in sweeps)
        k_sweep = np.full((len(sweeps), rows, 2), np.nan)
        for i, sweep in enumerate(sweeps):
            k_sweep[i, :sweep.shape[0]] = sweep
        aggregate["k_sweep"] = k_sweep
    return aggregate


def _plot_compile(run_dir: Path, out_dir: Path) -> None:
    aggregate = _compile_aggregate(run_dir)
    if aggregate is None:
        return
    d = _dimension(run_dir)
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp = os.path.join(tmp_dir, f"compile_d{d}.npz")
        np.savez(tmp, **aggregate)
        summary_out = out_dir / f"compile_d{d}.png"
        plot_compile_summary(tmp, str(summary_out))
        print(f"  compile quality d={d} -> {summary_out}")
        depth_out = out_dir / f"min_depth_d{d}.png"
        if plot_min_depth_curve(tmp, str(depth_out)):
            print(f"  min-depth curve d={d} -> {depth_out}")


def _plot_calibration(run_dir: Path, out_dir: Path) -> None:
    cal_path = run_dir / "calibration.npz"
    if not cal_path.exists():
        return
    with np.load(cal_path) as data:
        if "curves" not in data.files:
            return
    d = _dimension(run_dir)
    out = out_dir / f"calibration_d{d}.png"
    plot_stability_calibration(str(cal_path), str(out))
    print(f"  stability calibration d={d} -> {out}")


def _devices(benchmark_dir: Path) -> list[str]:
    names: set[str] = set()
    for run_dir in benchmark_dir.glob("d*"):
        metrics = run_dir / "metrics"
        if metrics.exists():
            names.update(p.name for p in metrics.iterdir() if p.is_dir())
    return sorted(names)


def _plot_device(benchmark_dir: Path, device: str) -> None:
    run_dirs = sorted(benchmark_dir.glob("d*"), key=_dimension)
    out_dir = plots_dir(benchmark_dir.name, device)

    noiseless = [r for r in (_noiseless_record(rd, device) for rd in run_dirs) if r]
    if noiseless:
        out = out_dir / "metrics_vs_dim.png"
        plot_metrics_vs_dim(noiseless, str(out))
        print(f"  [{device}] metrics vs dim ({len(noiseless)} config(s)) -> {out}")
    else:
        print(f"  [{device}] no noiseless results found")

    sweeps = [r for r in (_sweep_record(rd, device) for rd in run_dirs) if r]
    if not sweeps:
        return

    heatmap_dir = out_dir / "heatmaps"
    for record in sweeps:
        out = heatmap_dir / f"heatmap_d{record['d']}.png"
        plot_noise_heatmaps(record, str(out))
        print(f"  [{device}] T1/T2 heatmaps d={record['d']} -> {out}")

    ceilings = {r["d"]: (r["hog"], r["xeb"]) for r in noiseless}
    for record in sweeps:
        record["hog_nl"], record["xeb_nl"] = ceilings.get(record["d"], (None, None))
    t1_out = out_dir / "t1_sweep.png"
    if plot_t1_sweep(sweeps, str(t1_out)):
        print(f"  [{device}] T1 sweep ({len(sweeps)} config(s)) -> {t1_out}")

    t2_dir = out_dir / "t2_sweep"
    n_t2 = plot_t2_sweep(sweeps, str(t2_dir))
    if n_t2:
        print(f"  [{device}] T2 sweep ({n_t2} figure(s)) -> {t2_dir}")


def main() -> None:
    runs_root = data_dir("runs")
    if not runs_root.exists():
        print(f"No results found under {runs_root}")
        return
    for benchmark_dir in sorted(p for p in runs_root.iterdir() if p.is_dir()):
        for run_dir in sorted(benchmark_dir.glob("d*"), key=_dimension):
            _plot_compile(run_dir, plots_dir(benchmark_dir.name, "compile"))
            _plot_calibration(run_dir, plots_dir(benchmark_dir.name, "calibration"))
        for device in _devices(benchmark_dir):
            _plot_device(benchmark_dir, device)
    print("Done.")


if __name__ == "__main__":
    main()
