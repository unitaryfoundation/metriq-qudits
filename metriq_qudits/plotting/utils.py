"""Plotting helpers for the single-qudit ECD benchmark.

Two groups of functions:

- Per-circuit *diagnostics* written during a pipeline run (pulse/trajectory/
  Wigner sanity checks), used by the optional ``diagnostics_dir`` hooks in the
  pulse and simulation stages.
- *Results* plots built after a run from the cached ``outputs/*`` NPZ files,
  driven by ``scripts/plot_results.py``.
"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np

METRICS = ("hog", "xeb", "fid")


def _save_diag_fig(out_dir: str, name: str, title: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    fig = plt.gcf()
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, name), dpi=120, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Pulse / trajectory diagnostics
# --------------------------------------------------------------------------- #

def plot_pulse(epsilon, ancilla_drive) -> None:
    """Cavity drive ε(t) and ancilla drive Ω(t), real/imag parts."""
    _, axs = plt.subplots(2, 1, figsize=(6, 5))
    axs[0].plot(np.real(epsilon), label="Real")
    axs[0].plot(np.imag(epsilon), label="Imag")
    axs[0].set_ylabel(r"$\epsilon(t)$", fontsize=16)
    axs[0].legend()
    axs[1].plot(np.real(ancilla_drive), label="I")
    axs[1].plot(np.imag(ancilla_drive), label="Q")
    axs[1].set_ylabel(r"$\Omega(t)$", fontsize=16)
    axs[1].set_xlabel("ns", fontsize=16)
    axs[1].legend()


def plot_trajectory_complex(alpha_g, alpha_e=None, ax=None, symmetric=False) -> None:
    """Semiclassical g/e conditional trajectory in the complex α plane."""
    if ax is None:
        _, ax = plt.subplots()
    ax.plot(np.real(alpha_g), np.imag(alpha_g), label="g")
    ax.fill_between(np.real(alpha_g), np.imag(alpha_g), alpha=0.2)

    radius = np.max(np.abs(alpha_g))
    if alpha_e is not None:
        ax.plot(np.real(alpha_e), np.imag(alpha_e), label="e")
        ax.fill_between(np.real(alpha_e), np.imag(alpha_e), alpha=0.2)
        radius = max(radius, np.max(np.abs(alpha_e)))

    theta = np.linspace(0, 2 * np.pi, 100)
    ax.plot(radius * np.sin(theta), radius * np.cos(theta), "--")
    if symmetric:
        lim = np.ceil(radius)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False)
    ax.set_title("Semi-classical trajectory")


def save_conditional_trajectory_diag(alpha_g, alpha_e, out_dir, name, title) -> None:
    """One ECD gate's g/e trajectory: complex-plane path plus components vs time."""
    _, (ax_c, ax_r) = plt.subplots(1, 2, figsize=(11, 4.6))
    plot_trajectory_complex(alpha_g, alpha_e, ax=ax_c, symmetric=True)
    ts = np.arange(len(alpha_g))
    ax_r.plot(ts, np.real(alpha_g), label="Re α_g")
    ax_r.plot(ts, np.imag(alpha_g), label="Im α_g")
    ax_r.plot(ts, np.real(alpha_e), "--", label="Re α_e")
    ax_r.plot(ts, np.imag(alpha_e), "--", label="Im α_e")
    ax_r.set_xlabel("time [ns]")
    ax_r.grid(True, alpha=0.3)
    ax_r.legend(fontsize=8, frameon=False)
    ax_r.set_title("trajectory components")
    _save_diag_fig(out_dir, name, title)


def _wigner_on_ax(ax, state, rg=3.0, num=100):
    import qutip as qt
    import matplotlib

    vec = np.linspace(-rg, rg, num)
    w = qt.wigner(state, vec, vec)
    lim = np.abs(w).max()
    norm = matplotlib.colors.Normalize(-lim, lim)
    pl = ax.contourf(vec, vec, w, 100, cmap=matplotlib.cm.RdBu, norm=norm)
    ax.set_aspect("equal")
    ax.set_xlabel("Re α")
    ax.set_ylabel("Im α")
    return pl


def _target_state(psi_target, N_cav):
    """Embed a d-level target amplitude vector into an N_cav Fock ket."""
    import qutip as qt

    amps = np.zeros(N_cav, dtype=complex)
    amps[: len(psi_target)] = psi_target
    amps /= np.linalg.norm(amps)
    return qt.Qobj(amps.reshape(N_cav, 1))


def save_state_diagnostics(alpha, rho_final, out_dir,
                           psi_target=None, N_cav=None, circuit_idx=0) -> None:
    """Per-circuit noiseless sanity visuals: the drive-frame trajectory α(t) with
    its transient photon number, and the final cavity Wigner function (target vs
    achieved when a target is given)."""
    ci = circuit_idx
    _, (ax_t, ax_n) = plt.subplots(1, 2, figsize=(10, 4.5))
    ax_t.plot(np.real(alpha), np.imag(alpha), lw=1)
    ax_t.set_aspect("equal")
    ax_t.grid(True, alpha=0.3)
    ax_t.set_xlabel("Re α")
    ax_t.set_ylabel("Im α")
    ax_t.set_title("drive-frame trajectory α(t)")
    ax_n.plot(np.abs(alpha) ** 2)
    ax_n.set_xlabel("time [ns]")
    ax_n.set_ylabel("|α(t)|²")
    ax_n.grid(True, alpha=0.3)
    ax_n.set_title("transient photon number")
    _save_diag_fig(os.path.join(out_dir, "trajectories"), f"c{ci:02d}.png",
                   f"circuit {ci}")

    rho_cav = rho_final.ptrace(0)
    n_bar = float(np.real(rho_cav.diag() @ np.arange(rho_cav.shape[0])))
    rg = max(3.0, np.sqrt(max(n_bar, 0.0)) + 2)
    if psi_target is not None and N_cav is not None:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        _wigner_on_ax(axes[0], _target_state(psi_target, N_cav), rg=rg)
        axes[0].set_title("target (ideal)")
        pl = _wigner_on_ax(axes[1], rho_cav, rg=rg)
        axes[1].set_title("achieved (noiseless)")
        fig.colorbar(pl, ax=axes, shrink=0.8)
        _save_diag_fig(os.path.join(out_dir, "wigner"), f"c{ci:02d}.png",
                       f"Wigner: target vs achieved, circuit {ci}")
    else:
        fig, ax = plt.subplots(figsize=(6, 5))
        pl = _wigner_on_ax(ax, rho_cav, rg=rg)
        fig.colorbar(pl, ax=ax)
        _save_diag_fig(os.path.join(out_dir, "wigner"), f"c{ci:02d}.png",
                       f"final state Wigner, circuit {ci}")


# --------------------------------------------------------------------------- #
# Results plots (from cached npz)
# --------------------------------------------------------------------------- #

def plot_metrics_vs_dim(records, out_path: str) -> None:
    """Noiseless HOG/XEB/FID versus qudit dimension d. ``records`` is a list of
    dicts with keys d, hog, xeb, fid."""
    records = sorted(records, key=lambda r: r["d"])
    d = np.array([r["d"] for r in records])
    fig, ax = plt.subplots(figsize=(7, 5))
    for metric, marker in zip(METRICS, ("o", "s", "^")):
        ax.plot(d, [r[metric] for r in records], marker + "-", label=metric.upper())
    ax.set_xlabel("qudit dimension d")
    ax.set_ylabel("metric")
    ax.set_title("Noiseless benchmark metrics vs dimension")
    ax.set_xticks(d)
    ax.grid(True, alpha=0.3)
    ax.legend()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_compile_summary(compiled_path: str, out_path: str, err_th: float = 0.01) -> None:
    """Stage-1 compile quality for one config, from the compiled-circuits NPZ.

    Four panels: per-circuit final infidelity and boundary leakage (both colored
    by the accepted depth k), the |β| distribution across all circuits/layers
    (the ECD amplitudes that drive pulse duration), and the batched-optimizer
    convergence traces. Skips silently for an empty cache.
    """
    data = np.load(compiled_path)
    err = data["err"]
    leakage = data["boundary_leakage"]
    k_per = data["k_per_circuit"]
    n_circuits = len(err)
    if n_circuits == 0:
        return
    n_attempted = int(data["n_attempted"]) if "n_attempted" in data.files else None

    beta_mags = np.concatenate([
        np.abs(data["betas"][i, :ki].ravel()) for i, ki in enumerate(k_per)
    ])
    beta_mags = beta_mags[beta_mags > 1e-3]  # drop zero-padding and skipped gates

    k_values = sorted({int(k) for k in k_per})
    palette = plt.cm.tab10.colors
    k_color = {k: palette[i % len(palette)] for i, k in enumerate(k_values)}
    index = np.arange(n_circuits)

    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    ax_err, ax_leak, ax_beta, ax_conv = axes.flat

    for k in k_values:
        mask = k_per == k
        ax_err.semilogy(index[mask], err[mask], "o", ms=5, color=k_color[k],
                        label=f"k={k}" if len(k_values) > 1 else None)
    ax_err.axhline(err_th, color="darkred", ls="--", lw=1.2, label=f"err_th={err_th:g}")
    ax_err.set_xlabel("circuit index")
    ax_err.set_ylabel("compile infidelity")
    ax_err.set_title("optimizer final error per circuit")
    ax_err.legend(fontsize=8)
    ax_err.grid(True, which="both", alpha=0.3)

    for k in k_values:
        mask = k_per == k
        ax_leak.semilogy(index[mask], np.maximum(leakage[mask], 1e-12), "o", ms=5,
                         color=k_color[k])
    ax_leak.set_xlabel("circuit index")
    ax_leak.set_ylabel("max top-Fock population")
    ax_leak.set_title("boundary leakage per circuit")
    ax_leak.grid(True, which="both", alpha=0.3)

    ax_beta.hist(beta_mags, bins=30, color="tab:blue", alpha=0.75)
    ax_beta.set_xlabel(r"$|\beta|$")
    ax_beta.set_ylabel("count")
    ax_beta.set_title(f"ECD amplitudes, all circuits/layers "
                      f"(median {np.median(beta_mags):.2f})")
    ax_beta.grid(True, alpha=0.3)

    if "opt_trace" in data.files and data["opt_trace"].size:
        stride = int(data["trace_stride"]) if "trace_stride" in data.files else 1
        for i, row in enumerate(data["opt_trace"]):
            trace = np.exp(row[~np.isnan(row)])  # stored log-objective -> objective
            steps = stride * np.arange(1, len(trace) + 1)
            ax_conv.semilogy(steps, trace, color=k_color[int(k_per[i])], alpha=0.4, lw=1)
        ax_conv.axhline(err_th, color="darkred", ls="--", lw=1.2)
        ax_conv.set_xlabel("optimizer step")
        ax_conv.set_ylabel("batch-best objective")
        ax_conv.set_title("optimizer convergence")
        ax_conv.grid(True, which="both", alpha=0.3)
    else:
        ax_conv.text(0.5, 0.5, "no opt_trace in cache\n(recompile with --overwrite)",
                     ha="center", va="center", fontsize=9, color="dimgray")
        ax_conv.set_axis_off()

    survived = f"{n_circuits}/{n_attempted}" if n_attempted else str(n_circuits)
    fig.suptitle(f"compile quality  ({os.path.basename(compiled_path)})  "
                 f"circuits: {survived}", fontsize=11)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_min_depth_curve(compiled_path: str, out_path: str, err_th: float = 0.01) -> bool:
    """Minimum-depth sweep: best compile infidelity versus ECD depth k for every
    circuit (from the ``k_sweep`` recorded during compilation). The filled marker
    is each circuit's accepted minimum k. Returns False (writing nothing) when the
    cache carries no depth sweep.
    """
    data = np.load(compiled_path)
    if "k_sweep" not in data.files or not data["k_sweep"].size:
        return False

    fig, ax = plt.subplots(figsize=(7, 5))
    plotted = False
    for rows in data["k_sweep"]:
        rows = rows[~np.isnan(rows[:, 0])]
        if not len(rows):
            continue
        ax.semilogy(rows[:, 0], np.maximum(rows[:, 1], 1e-12), "-o",
                    color="tab:blue", ms=3, lw=1, alpha=0.35)
        ax.semilogy(rows[-1, 0], max(rows[-1, 1], 1e-12), "o",
                    color="tab:blue", ms=6, zorder=3)
        plotted = True
    if not plotted:
        plt.close(fig)
        return False

    ax.axhline(err_th, color="darkred", ls="--", lw=1.2, label=f"err_th={err_th:g}")
    ax.set_xlabel("circuit depth k")
    ax.set_ylabel("best infidelity at depth k")
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    ax.set_title(f"minimum-depth sweep  ({os.path.basename(compiled_path)})")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return True


def plot_stability_calibration(cal_path: str, out_path: str) -> None:
    """Fock-truncation calibration for one config, from a calibration NPZ.

    Left: max (over calibration circuits) stability infidelity versus the test
    Fock truncation N_test, one curve per candidate buffer count (the shaded band
    spans min..max across circuits). Right: the summary max stability infidelity
    that drove the choice, versus buffer count. The stability threshold and the
    selected buffer count are marked on both.
    """
    data = np.load(cal_path)
    d = int(data["d"])
    depth = int(data["k"])
    stab_th = float(data["stability_th"])
    n_test_extra = np.asarray(data["N_test_extra"])
    buffers = data["num_buffers_tried"]
    max_stab = data["max_stab"]
    curves = data["curves"]
    best_buffers = int(data["best_buffers"])

    order = np.argsort(buffers)
    palette = plt.cm.viridis(np.linspace(0.15, 0.9, len(buffers)))
    buf_color = {int(b): palette[i] for i, b in enumerate(buffers[order])}

    fig, (ax_curve, ax_summary) = plt.subplots(1, 2, figsize=(12, 4.6),
                                               constrained_layout=True)

    for row in order:
        nb = int(buffers[row])
        circuit_curves = curves[row]
        valid = ~np.all(np.isnan(circuit_curves), axis=1)
        if not valid.any():
            continue
        x = (d + nb) + n_test_extra
        max_per_point = np.nanmax(circuit_curves[valid], axis=0)
        is_best = nb == best_buffers
        ax_curve.semilogy(x, max_per_point, color=buf_color[nb],
                          marker="o" if is_best else ".",
                          ms=6 if is_best else 4, lw=1.8 if is_best else 1.1,
                          label=f"{nb} buf" + ("  (chosen)" if is_best else ""))
        if valid.sum() > 1:
            ax_curve.fill_between(x, np.nanmin(circuit_curves[valid], axis=0),
                                  max_per_point, color=buf_color[nb], alpha=0.15)

    ax_curve.axhline(stab_th, color="black", ls="--", lw=0.8,
                     label=f"threshold {stab_th:.0e}")
    ax_curve.set_xlabel(r"$N_\mathrm{test}$ (Fock truncation)")
    ax_curve.set_ylabel("stability infidelity")
    ax_curve.set_title("stability vs test truncation")
    ax_curve.grid(True, which="both", alpha=0.3)
    ax_curve.legend(fontsize=7, ncol=2)

    b_sorted = buffers[order]
    s_sorted = max_stab[order]
    finite = np.isfinite(s_sorted)
    ax_summary.semilogy(b_sorted[finite], np.maximum(s_sorted[finite], 1e-12),
                        "-o", color="tab:blue", ms=5)
    ax_summary.axhline(stab_th, color="black", ls="--", lw=0.8,
                       label=f"threshold {stab_th:.0e}")
    ax_summary.axvline(best_buffers, color="tab:green", ls=":", lw=1.5,
                       label=f"chosen: {best_buffers} buf")
    ax_summary.set_xlabel("num_buffers")
    ax_summary.set_ylabel("max stability infidelity")
    ax_summary.set_title("calibration summary")
    ax_summary.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax_summary.grid(True, which="both", alpha=0.3)
    ax_summary.legend(fontsize=8)

    fig.suptitle(f"Fock-buffer calibration  d={d}  k={depth}  "
                 f"({os.path.basename(cal_path)})", fontsize=11)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


HOG_PASS = 2.0 / 3.0        # heavy-output-generation pass threshold
_T2_TOL_US = 0.5            # µs tolerance when matching the T2 = 2*T1 diagonal


def _diagonal_indices(T1_us, T2_us):
    """(it1, it2) index pairs along the physical T2 = 2*T1 diagonal, when the
    grid actually contains a T2 within tolerance of 2*T1."""
    pairs = []
    for it1, t1 in enumerate(T1_us):
        it2 = int(np.argmin(np.abs(T2_us - 2 * t1)))
        if abs(T2_us[it2] - 2 * t1) <= _T2_TOL_US:
            pairs.append((it1, it2))
    return pairs


def plot_t1_sweep(records, out_path: str) -> bool:
    """HOG and XEB along the physical T2 = 2*T1 diagonal versus qubit T1, one line
    per config (mean ± SEM). The noiseless ceilings are drawn as dotted lines in
    the matching color and the HOG pass threshold as a dashed line. Returns False
    (writing nothing) when no config has any point on the diagonal.

    records: list of dicts with keys d, T1_us, T2_us, hog, xeb, hog_std, xeb_std,
    N_u, and optional hog_nl / xeb_nl noiseless ceilings.
    """
    records = sorted(records, key=lambda r: r["d"])
    fig, (ax_hog, ax_xeb) = plt.subplots(1, 2, figsize=(12, 4.5))
    palette = plt.cm.tab10.colors
    plotted = False

    for i, rec in enumerate(records):
        T1 = np.asarray(rec["T1_us"])
        T2 = np.asarray(rec["T2_us"])
        pairs = _diagonal_indices(T1, T2)
        if not pairs:
            continue
        N_u = max(int(rec.get("N_u", 1)), 1)
        t1_pts, hog_pts, hog_err, xeb_pts, xeb_err = [], [], [], [], []
        for it1, it2 in pairs:
            h, x = rec["hog"][it1, it2], rec["xeb"][it1, it2]
            if np.isnan(h) or np.isnan(x):
                continue
            t1_pts.append(T1[it1])
            hog_pts.append(h)
            xeb_pts.append(x)
            hog_err.append(rec["hog_std"][it1, it2] / np.sqrt(N_u))
            xeb_err.append(rec["xeb_std"][it1, it2] / np.sqrt(N_u))
        if not t1_pts:
            continue

        color = palette[i % len(palette)]
        label = f"d={rec['d']}"
        ax_hog.errorbar(t1_pts, hog_pts, yerr=hog_err, color=color, marker="o",
                        ms=5, lw=1.8, capsize=3, label=label)
        ax_xeb.errorbar(t1_pts, xeb_pts, yerr=xeb_err, color=color, marker="s",
                        ms=5, lw=1.8, capsize=3, label=label)
        if rec.get("hog_nl") is not None:
            ax_hog.axhline(rec["hog_nl"], color=color, ls=":", lw=1.2, alpha=0.8)
        if rec.get("xeb_nl") is not None:
            ax_xeb.axhline(rec["xeb_nl"], color=color, ls=":", lw=1.2, alpha=0.8)
        plotted = True

    if not plotted:
        plt.close(fig)
        return False

    ax_hog.axhline(HOG_PASS, color="darkred", ls="--", lw=1.5, label="HOG pass (2/3)")
    ax_hog.set_xlabel("qubit T₁ [µs]")
    ax_hog.set_ylabel("HOG fraction")
    ax_hog.set_title("HOG at T₂ = 2T₁")
    ax_hog.grid(True, alpha=0.3)
    ax_hog.legend(fontsize=8)
    ax_xeb.set_xlabel("qubit T₁ [µs]")
    ax_xeb.set_ylabel("XEB$_n$")
    ax_xeb.set_title("XEB$_n$ at T₂ = 2T₁")
    ax_xeb.grid(True, alpha=0.3)
    ax_xeb.legend(fontsize=8)

    fig.suptitle("Coherence sweep: metrics vs T₁ along T₂ = 2T₁  "
                 "(dotted = noiseless ceiling)", fontsize=10)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return True


T2_LOW_TARGET_US = 35.0     # µs — "strong dephasing" reference for the low-T2 slice


def plot_t2_sweep(records, out_dir: str) -> int:
    """For each fixed T₁, HOG (left axis) and XEB (right twin axis) versus qudit
    dimension d, comparing a low-T₂ (strong dephasing) slice against the
    T₂ = 2·T₁ slice. One figure per T₁ value that has two *distinct* T₂ levels to
    compare; T₁ values whose low/high slices collapse to the same T₂ are skipped
    (that contrast is already carried by the T₁ sweep). Configs must share the
    reference T₁/T₂ grid. Returns the number of figures written.

    records: list of dicts with keys d, T1_us, T2_us, hog, xeb (T1×T2 arrays).
    """
    if not records:
        return 0
    ref = records[0]
    T1 = np.asarray(ref["T1_us"])
    T2 = np.asarray(ref["T2_us"])
    compatible = sorted(
        (r for r in records
         if np.array_equal(np.asarray(r["T1_us"]), T1)
         and np.array_equal(np.asarray(r["T2_us"]), T2)),
        key=lambda r: r["d"],
    )
    d_ticks = sorted({r["d"] for r in compatible})
    os.makedirs(out_dir, exist_ok=True)

    written = 0
    for it1, t1 in enumerate(T1):
        it2_hi = int(np.argmin(np.abs(T2 - 2 * t1)))
        it2_lo = int(np.argmin(np.abs(T2 - min(T2_LOW_TARGET_US, 2 * t1))))
        if it2_lo == it2_hi:
            continue  # no distinct low/high T2 to contrast at this T1

        fig, ax_hog = plt.subplots(figsize=(7, 4.5))
        ax_xeb = ax_hog.twinx()
        ax_hog.set_xlabel("qudit dimension d")
        ax_hog.set_ylabel("HOG fraction")
        ax_xeb.set_ylabel("XEB$_n$")
        ax_hog.set_title(f"T₂ sweep (T₁ = {t1:.0f} µs)")
        ax_hog.axhline(HOG_PASS, color="darkred", ls="--", lw=1.5, label="HOG pass (2/3)")
        ax_hog.grid(True, alpha=0.3)

        slices = (
            (it2_lo, T2[it2_lo], "tab:blue", "tab:purple"),
            (it2_hi, T2[it2_hi], "tab:green", "tab:orange"),
        )
        any_plotted = False
        for it2, t2_val, hog_color, xeb_color in slices:
            d_pts, hog_pts, xeb_pts = [], [], []
            for r in compatible:
                h, x = r["hog"][it1, it2], r["xeb"][it1, it2]
                if not (np.isnan(h) or np.isnan(x)):
                    d_pts.append(r["d"])
                    hog_pts.append(h)
                    xeb_pts.append(x)
            if not d_pts:
                continue
            label = f"T₂ = {t2_val:.0f} µs"
            ax_hog.plot(d_pts, hog_pts, "o-", color=hog_color, lw=1.8, ms=6,
                        label=f"HOG: {label}")
            ax_xeb.plot(d_pts, xeb_pts, "s:", color=xeb_color, lw=1.8, ms=6,
                        mfc="none", label=f"XEB$_n$: {label}")
            any_plotted = True

        if not any_plotted:
            plt.close(fig)
            continue

        ax_hog.set_xticks(d_ticks)
        handles = ax_hog.get_legend_handles_labels()
        extra = ax_xeb.get_legend_handles_labels()
        ax_hog.legend(handles[0] + extra[0], handles[1] + extra[1],
                      fontsize=8, loc="best")
        out_path = os.path.join(out_dir, f"t2_sweep_T1_{t1:.0f}us.png")
        fig.tight_layout()
        fig.savefig(out_path, dpi=120)
        plt.close(fig)
        written += 1

    return written


def plot_noise_heatmaps(record, out_path: str) -> None:
    """HOG/XEB/FID heatmaps over the T1×T2 grid for one config. ``record`` has
    keys d, T1_us, T2_us, and a T1×T2 array per metric."""
    T1, T2 = record["T1_us"], record["T2_us"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, metric in zip(axes, METRICS):
        grid = np.asarray(record[metric], dtype=float)
        im = ax.imshow(grid.T, origin="lower", aspect="auto", cmap="viridis",
                       vmin=0, vmax=1,
                       extent=[T1[0], T1[-1], T2[0], T2[-1]])
        ax.set_xlabel("T1 [µs]")
        ax.set_ylabel("T2 [µs]")
        ax.set_title(f"{metric.upper()}  (d={record['d']})")
        fig.colorbar(im, ax=ax, shrink=0.85)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
