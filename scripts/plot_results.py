"""
Generate result figures from saved pipeline output.
"""

from __future__ import annotations

import glob
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from ecd_qv.plot_utils import METRICS, plot_metrics_vs_dim, plot_noise_heatmaps

CACHE = REPO_ROOT / ".cache"
PLOTS = REPO_ROOT / "plots"


def _load_noiseless() -> list[dict]:
    records = []
    for path in sorted(glob.glob(str(CACHE / "noiseless" / "noiseless_d*.npz"))):
        data = np.load(path)
        records.append({
            "d": int(data["d"]),
            **{metric: float(data[f"{metric}_mean"]) for metric in METRICS},
        })
    return records


def _load_noise_sweeps() -> list[dict]:
    records = []
    for path in sorted(glob.glob(str(CACHE / "noise_sweep_pulse" / "results_d*.npz"))):
        data = np.load(path)
        records.append({
            "d": int(data["d"]),
            "T1_us": data["T1_values"] * 1e6,
            "T2_us": data["T2_values"] * 1e6,
            # stored as (T1, T2, 1); drop the trailing singleton axis
            **{metric: data[metric][:, :, 0] for metric in METRICS},
        })
    return records


def main() -> None:
    PLOTS.mkdir(parents=True, exist_ok=True)

    noiseless = _load_noiseless()
    if noiseless:
        out = PLOTS / "metrics_vs_dim.png"
        plot_metrics_vs_dim(noiseless, str(out))
        print(f"  metrics vs dim ({len(noiseless)} config(s)) -> {out}")
    else:
        print("  no noiseless results found")

    sweeps = _load_noise_sweeps()
    if sweeps:
        heatmap_dir = PLOTS / "heatmaps"
        for record in sweeps:
            out = heatmap_dir / f"heatmap_d{record['d']}.png"
            plot_noise_heatmaps(record, str(out))
            print(f"  T1/T2 heatmaps d={record['d']} -> {out}")
    else:
        print("  no noise-sweep results found (run without --skip-sweep)")

    print("Done.")


if __name__ == "__main__":
    main()
