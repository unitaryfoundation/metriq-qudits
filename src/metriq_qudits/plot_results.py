"""Generate figures from cached ECD-QV pipeline results."""

from __future__ import annotations

import glob

import matplotlib

matplotlib.use("Agg")
import numpy as np

from metriq_qudits.plot_utils import (
    METRICS,
    plot_compile_summary,
    plot_metrics_vs_dim,
    plot_min_depth_curve,
    plot_noise_heatmaps,
    plot_stability_calibration,
    plot_t1_sweep,
    plot_t2_sweep,
)
from metriq_qudits.paths import data_dir, plots_dir


def _load_noiseless() -> list[dict]:
    records = []
    for path in sorted(glob.glob(str(data_dir("noiseless") / "noiseless_d*.npz"))):
        data = np.load(path)
        records.append(
            {
                "d": int(data["d"]),
                **{metric: float(data[f"{metric}_mean"]) for metric in METRICS},
            }
        )
    return records


def _load_noise_sweeps() -> list[dict]:
    records = []
    for path in sorted(
        glob.glob(str(data_dir("noise_sweeps") / "results_d*.npz"))
    ):
        data = np.load(path)
        records.append(
            {
                "d": int(data["d"]),
                "T1_us": data["T1_values"] * 1e6,
                "T2_us": data["T2_values"] * 1e6,
                "N_u": int(data["N_unitaries"]),
                **{metric: data[metric][:, :, 0] for metric in METRICS},
                **{f"{metric}_std": data[f"{metric}_std"][:, :, 0] for metric in METRICS},
            }
        )
    return records


def main() -> None:
    output_plots = plots_dir()
    output_plots.mkdir(parents=True, exist_ok=True)

    compiled_paths = sorted(
        glob.glob(str(data_dir("compiled_circuits") / "*.npz"))
    )
    if compiled_paths:
        compile_dir = output_plots / "compile"
        for path in compiled_paths:
            with np.load(path) as data:
                d = int(data["d"])
            out = compile_dir / f"compile_d{d}.png"
            plot_compile_summary(path, str(out))
            print(f"  compile quality d={d} -> {out}")

            depth_out = compile_dir / f"min_depth_d{d}.png"
            if plot_min_depth_curve(path, str(depth_out)):
                print(f"  min-depth curve d={d} -> {depth_out}")
    else:
        print("  no compiled circuits found")

    cal_paths = sorted(glob.glob(str(data_dir("calibration") / "cal_*.npz")))
    if cal_paths:
        calibration_dir = output_plots / "calibration"
        for path in cal_paths:
            with np.load(path) as data:
                d = int(data["d"])
            out = calibration_dir / f"calibration_d{d}.png"
            plot_stability_calibration(path, str(out))
            print(f"  stability calibration d={d} -> {out}")

    noiseless = _load_noiseless()
    if noiseless:
        out = output_plots / "metrics_vs_dim.png"
        plot_metrics_vs_dim(noiseless, str(out))
        print(f"  metrics vs dim ({len(noiseless)} config(s)) -> {out}")
    else:
        print("  no noiseless results found")

    sweeps = _load_noise_sweeps()
    if sweeps:
        heatmap_dir = output_plots / "heatmaps"
        for record in sweeps:
            out = heatmap_dir / f"heatmap_d{record['d']}.png"
            plot_noise_heatmaps(record, str(out))
            print(f"  T1/T2 heatmaps d={record['d']} -> {out}")

        ceilings = {r["d"]: (r["hog"], r["xeb"]) for r in noiseless}
        for record in sweeps:
            record["hog_nl"], record["xeb_nl"] = ceilings.get(record["d"], (None, None))
        t1_out = output_plots / "t1_sweep.png"
        if plot_t1_sweep(sweeps, str(t1_out)):
            print(f"  T1 sweep ({len(sweeps)} config(s)) -> {t1_out}")

        t2_dir = output_plots / "t2_sweep"
        n_t2 = plot_t2_sweep(sweeps, str(t2_dir))
        if n_t2:
            print(f"  T2 sweep ({n_t2} figure(s)) -> {t2_dir}")
    else:
        print("  no noise-sweep results found (run without --skip-sweep)")

    print("Done.")


if __name__ == "__main__":
    main()
