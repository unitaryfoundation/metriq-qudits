"""Generate figures from cached ECD-QV pipeline results."""

from __future__ import annotations

import glob

import matplotlib

matplotlib.use("Agg")
import numpy as np

from metriq_qudits.plot_utils import METRICS, plot_metrics_vs_dim, plot_noise_heatmaps
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
                **{metric: data[metric][:, :, 0] for metric in METRICS},
            }
        )
    return records


def main() -> None:
    output_plots = plots_dir()
    output_plots.mkdir(parents=True, exist_ok=True)

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
    else:
        print("  no noise-sweep results found (run without --skip-sweep)")

    print("Done.")


if __name__ == "__main__":
    main()
